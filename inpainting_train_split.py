"""Inpainting training with observed train/val split.

This script is intentionally separate from inpainting_train.py. It keeps the
same Gabor/Gaussian image parameterization, but splits the observed pixels into
train and validation subsets so overfitting can be monitored without relying on
ground-truth PSNR.
"""

import argparse
import copy
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import yaml
from PIL import Image
from tqdm import tqdm

from gaussianimage_cholesky import GaussianImage_Cholesky
from inpainting_utils import (
    cholesky_l2_reg,
    compute_error_map,
    compute_inpainting_psnrs,
    compute_ms_ssim,
    compute_psnr,
    compute_region_error_map,
    compute_masked_psnr,
    coverage_map_to_image_tensor,
    gabor_weights_l1_reg,
    generate_mask,
    get_missing_mask,
    mask_to_image_tensor,
    masked_loss_fn,
    nuclear_norm_reg,
    save_tensor_as_image,
    tv_loss,
)
from utils import LogWriter
from init_strategies import apply_obs_first_init


def normalize_image_tensor(img_tensor, normalize_mode="auto"):
    """Normalize image tensor to [0, 1]."""
    img_tensor = img_tensor.float()
    if normalize_mode == "none":
        return img_tensor

    min_value = img_tensor.amin()
    max_value = img_tensor.amax()
    if normalize_mode == "auto":
        if min_value >= 0.0 and max_value <= 1.0:
            return img_tensor
        if min_value >= 0.0 and max_value <= 255.0:
            return img_tensor / 255.0
    if max_value > min_value:
        return (img_tensor - min_value) / (max_value - min_value)
    return torch.zeros_like(img_tensor)


def image_path_to_tensor(image_path, normalize_mode="auto"):
    img = Image.open(image_path).convert("RGB")
    transform = transforms.ToTensor()
    img_tensor = transform(img).unsqueeze(0)
    return normalize_image_tensor(img_tensor, normalize_mode=normalize_mode)


def build_dataset_image_list(dataset_path, data_name):
    dataset_path = Path(dataset_path)
    if data_name == "kodak":
        return [dataset_path / f"kodim{i + 1:02}.png" for i in range(24)]
    if data_name == "DIV2K_valid_LRX2":
        return [dataset_path / f"{i + 1:04}x2.png" for i in range(800, 900)]

    image_paths = []
    for pattern in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"):
        image_paths.extend(dataset_path.glob(pattern))
    return sorted(image_paths)


def save_run_config(log_dir, args_text):
    config_path = log_dir / "config.yaml"
    with open(config_path, "w") as f:
        f.write(args_text)


def split_observed_mask(observed_mask, val_ratio):
    """Split observed entries into train and validation subsets.

    Args:
        observed_mask: (1, 1, H, W) or (1, C, H, W), values in {0, 1}.
        val_ratio: fraction of observed entries reserved for validation.

    Returns:
        train_mask, val_mask with the same shape as observed_mask.
    """
    if val_ratio < 0.0 or val_ratio >= 1.0:
        raise ValueError("val_ratio must be in [0, 1).")

    train_mask = observed_mask.clone()
    val_mask = torch.zeros_like(observed_mask)
    if val_ratio == 0.0:
        return train_mask, val_mask

    flat_observed = observed_mask.reshape(-1) > 0.5
    observed_indices = torch.nonzero(flat_observed, as_tuple=False).squeeze(1)
    num_observed = int(observed_indices.numel())
    if num_observed == 0:
        return train_mask, val_mask
    if num_observed == 1:
        return train_mask, val_mask

    num_val = int(round(num_observed * val_ratio))
    num_val = max(1, min(num_val, num_observed - 1))
    perm = torch.randperm(num_observed, device=observed_mask.device)
    val_indices = observed_indices[perm[:num_val]]

    train_flat = train_mask.reshape(-1)
    val_flat = val_mask.reshape(-1)
    train_flat[val_indices] = 0.0
    val_flat[val_indices] = 1.0
    return train_mask, val_mask


class SplitInpaintingTrainer:
    """Inpainting trainer with observed train/val split."""

    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda:0")

        self.gt_image = image_path_to_tensor(
            args.image_path,
            normalize_mode=args.normalize_mode,
        ).to(self.device)
        self.H, self.W = self.gt_image.shape[2], self.gt_image.shape[3]
        self.image_name = Path(args.image_path).stem

        self.observed_mask = generate_mask(
            self.H,
            self.W,
            mask_type=args.mask_type,
            mask_ratio=args.mask_ratio,
            block_size=args.block_size,
            num_blocks=args.num_blocks,
            C=self.gt_image.shape[1],
            device=self.device,
        )
        self.train_mask, self.val_mask = split_observed_mask(self.observed_mask, args.val_ratio)
        self.missing_mask = get_missing_mask(self.observed_mask)
        self.observed_image = self.gt_image * self.observed_mask
        self.train_observed_image = self.gt_image * self.train_mask
        self.val_observed_image = self.gt_image * self.val_mask

        self.psnr_masked = compute_psnr(self.observed_image, self.gt_image)
        self.psnr_train_masked = compute_masked_psnr(self.observed_image, self.gt_image, self.train_mask)
        self.has_val = bool((self.val_mask > 0.5).any().item())
        self.psnr_val_masked = (
            compute_masked_psnr(self.observed_image, self.gt_image, self.val_mask)
            if self.has_val else float("nan")
        )

        init_tag = args.init_strategy if hasattr(args, 'init_strategy') else 'random'
        if getattr(args, 'init_strategy', 'random') == 'obs_first':
            init_tag = f"obs_first_{args.obs_init_ratio}"

        self.log_dir = Path(
            f"./checkpoints_inpainting_split/{args.mask_type}_{args.mask_ratio}_val{args.val_ratio}/"
            f"GaussianImage_Cholesky_{args.iterations}_{args.num_points}_{args.num_gabor}"
            f"_{init_tag}/"
            f"{self.image_name}"
        )
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logwriter = LogWriter(self.log_dir)

        self.model = GaussianImage_Cholesky(
            loss_type="L2",
            opt_type="adan",
            num_points=args.num_points,
            H=self.H,
            W=self.W,
            BLOCK_H=16,
            BLOCK_W=16,
            device=self.device,
            lr=args.lr,
            num_gabor=args.num_gabor,
            quantize=False,
        ).to(self.device)

        if args.model_path is not None:
            checkpoint = torch.load(args.model_path, map_location=self.device)
            model_dict = self.model.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)

        # Apply initialization strategy
        if getattr(args, 'init_strategy', 'random') == 'obs_first':
            obs_ratio = getattr(args, 'obs_init_ratio', 0.8)
            print(f"Applying obs_first init (obs_init_ratio={obs_ratio})")
            apply_obs_first_init(
                self.model, self.observed_mask, self.gt_image,
                obs_init_ratio=obs_ratio,
            )

        self.best_val_psnr = float("-inf")
        self.best_iteration = 0
        self.best_state_dict = None

    def compute_regularization(self, image):
        reg_loss = torch.tensor(0.0, device=self.device)
        if self.args.lambda_gabor_l1 > 0:
            reg_loss = reg_loss + self.args.lambda_gabor_l1 * gabor_weights_l1_reg(self.model)
        if self.args.lambda_cholesky_l2 > 0:
            reg_loss = reg_loss + self.args.lambda_cholesky_l2 * cholesky_l2_reg(self.model)
        if self.args.lambda_tv > 0:
            reg_loss = reg_loss + self.args.lambda_tv * tv_loss(image)
        if self.args.lambda_nuclear > 0:
            reg_loss = reg_loss + self.args.lambda_nuclear * nuclear_norm_reg(image)
        return reg_loss

    def train_step(self):
        image = self.model.forward()["render"]
        data_fidelity = masked_loss_fn(
            image,
            self.gt_image,
            self.train_mask,
            loss_type=self.args.loss_type,
        )
        loss = data_fidelity + self.compute_regularization(image)
        loss.backward()
        self.model.optimizer.step()
        self.model.optimizer.zero_grad(set_to_none=True)
        self.model.scheduler.step()

        with torch.no_grad():
            region_metrics = compute_inpainting_psnrs(image, self.gt_image, self.observed_mask)
            psnr_gt = region_metrics["psnr_full"]
            psnr_train = compute_masked_psnr(image, self.gt_image, self.train_mask)
            psnr_val = (
                compute_masked_psnr(image, self.gt_image, self.val_mask)
                if self.has_val else float("nan")
            )
            psnr_missing = region_metrics["psnr_missing"]
        return loss, psnr_gt, psnr_train, psnr_val, psnr_missing

    def save_results(self, reconstruction):
        save_tensor_as_image(self.observed_image, self.log_dir / f"{self.image_name}_observed.png")
        save_tensor_as_image(self.train_observed_image, self.log_dir / f"{self.image_name}_train_observed.png")
        if self.has_val:
            save_tensor_as_image(self.val_observed_image, self.log_dir / f"{self.image_name}_val_observed.png")
        save_tensor_as_image(reconstruction, self.log_dir / f"{self.image_name}_reconstruction.png")
        save_tensor_as_image(mask_to_image_tensor(self.observed_mask), self.log_dir / f"{self.image_name}_mask.png")
        save_tensor_as_image(mask_to_image_tensor(self.train_mask), self.log_dir / f"{self.image_name}_train_mask.png")
        if self.has_val:
            save_tensor_as_image(mask_to_image_tensor(self.val_mask), self.log_dir / f"{self.image_name}_val_mask.png")
        error_map = compute_error_map(reconstruction, self.gt_image)
        observed_error_map = compute_region_error_map(reconstruction, self.gt_image, self.observed_mask)
        missing_error_map = compute_region_error_map(reconstruction, self.gt_image, self.missing_mask)
        error_vis = (error_map / 0.5).clamp(0, 1)
        observed_error_vis = (observed_error_map / 0.5).clamp(0, 1)
        missing_error_vis = (missing_error_map / 0.5).clamp(0, 1)
        save_tensor_as_image(error_vis, self.log_dir / f"{self.image_name}_error.png")
        save_tensor_as_image(observed_error_vis, self.log_dir / f"{self.image_name}_observed_error.png")
        save_tensor_as_image(missing_error_vis, self.log_dir / f"{self.image_name}_missing_error.png")
        if self.observed_mask.shape[1] > 1:
            save_tensor_as_image(
                coverage_map_to_image_tensor(self.observed_mask),
                self.log_dir / f"{self.image_name}_coverage.png",
            )

    def train(self):
        psnr_gt_list = []
        psnr_train_list = []
        psnr_val_list = []
        psnr_missing_list = []
        iter_list = []

        if self.args.save_observation:
            save_tensor_as_image(self.observed_image, self.log_dir / f"{self.image_name}_observed.png")
            save_tensor_as_image(mask_to_image_tensor(self.observed_mask), self.log_dir / f"{self.image_name}_mask.png")
            save_tensor_as_image(mask_to_image_tensor(self.train_mask), self.log_dir / f"{self.image_name}_train_mask.png")
            if self.has_val:
                save_tensor_as_image(mask_to_image_tensor(self.val_mask), self.log_dir / f"{self.image_name}_val_mask.png")
            if self.observed_mask.shape[1] > 1:
                save_tensor_as_image(
                    coverage_map_to_image_tensor(self.observed_mask),
                    self.log_dir / f"{self.image_name}_coverage.png",
                )

        progress_bar = tqdm(range(1, self.args.iterations + 1), desc="Inpainting split progress")
        self.model.train()
        start_time = time.time()
        for iteration in range(1, self.args.iterations + 1):
            loss, psnr_gt, psnr_train, psnr_val, psnr_missing = self.train_step()
            iter_list.append(iteration)
            psnr_gt_list.append(psnr_gt)
            psnr_train_list.append(psnr_train)
            psnr_val_list.append(psnr_val)
            psnr_missing_list.append(psnr_missing)

            if self.has_val and psnr_val > self.best_val_psnr:
                self.best_val_psnr = psnr_val
                self.best_iteration = iteration
                self.best_state_dict = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model.state_dict().items()
                }

            if iteration % self.args.log_interval == 0:
                postfix = {
                    "Loss": f"{loss.item():.7f}",
                    "PSNR(gt)": f"{psnr_gt:.4f}",
                    "PSNR(train)": f"{psnr_train:.4f}",
                    "PSNR(miss)": f"{psnr_missing:.4f}",
                }
                if self.has_val:
                    postfix["PSNR(val)"] = f"{psnr_val:.4f}"
                progress_bar.set_postfix(postfix)
                progress_bar.update(self.args.log_interval)

        if self.args.iterations % self.args.log_interval != 0:
            progress_bar.update(self.args.iterations % self.args.log_interval)
        progress_bar.close()
        training_time = time.time() - start_time

        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)
        elif not self.has_val:
            self.best_iteration = self.args.iterations

        self.model.eval()
        with torch.no_grad():
            reconstruction = self.model()["render"]

        region_metrics = compute_inpainting_psnrs(reconstruction, self.gt_image, self.observed_mask)
        psnr_full = region_metrics["psnr_full"]
        ms_ssim_full = compute_ms_ssim(reconstruction, self.gt_image)
        psnr_train_final = compute_masked_psnr(reconstruction, self.gt_image, self.train_mask)
        psnr_val_final = (
            compute_masked_psnr(reconstruction, self.gt_image, self.val_mask)
            if self.has_val else float("nan")
        )
        psnr_obs_final = region_metrics["psnr_observed"]
        psnr_missing_final = region_metrics["psnr_missing"]

        self.final_psnr_full = psnr_full
        self.final_psnr_observed = psnr_obs_final
        self.final_psnr_missing = psnr_missing_final
        self.final_ms_ssim = ms_ssim_full

        self.logwriter.write(
            f"Split Inpainting Complete in {training_time:.4f}s\n"
            f"  PSNR(masked vs gt):        {self.psnr_masked:.4f}\n"
            f"  PSNR(train_obs baseline):  {self.psnr_train_masked:.4f}\n"
            f"  PSNR(val_obs baseline):    {self.psnr_val_masked:.4f}\n"
            f"  Best iter by val:          {self.best_iteration}\n"
            f"  PSNR(recon vs gt):         {psnr_full:.4f}\n"
            f"  PSNR(recon train_obs):     {psnr_train_final:.4f}\n"
            f"  PSNR(recon val_obs):       {psnr_val_final:.4f}\n"
            f"  PSNR(recon obs):           {psnr_obs_final:.4f}\n"
            f"  PSNR(recon missing):       {psnr_missing_final:.4f}\n"
            f"  MS-SSIM(recon vs gt):      {ms_ssim_full:.6f}\n"
            f"  mask_type={self.args.mask_type}, mask_ratio={self.args.mask_ratio}, val_ratio={self.args.val_ratio}"
        )

        torch.save(self.model.state_dict(), self.log_dir / "gaussian_model_best.pth.tar")
        if self.args.save_imgs:
            self.save_results(reconstruction)

        np.save(self.log_dir / "inpainting_split_results.npy", {
            "iterations": iter_list,
            "training_psnr_gt": psnr_gt_list,
            "training_psnr_train": psnr_train_list,
            "training_psnr_val": psnr_val_list,
            "training_psnr_missing": psnr_missing_list,
            "training_time": training_time,
            "best_iteration": self.best_iteration,
            "psnr_masked": self.psnr_masked,
            "psnr_train_masked": self.psnr_train_masked,
            "psnr_val_masked": self.psnr_val_masked,
            "psnr_full": psnr_full,
            "psnr_train_final": psnr_train_final,
            "psnr_val_final": psnr_val_final,
            "psnr_obs_final": psnr_obs_final,
            "psnr_missing_final": psnr_missing_final,
            "ms_ssim_full": ms_ssim_full,
            "mask_type": self.args.mask_type,
            "mask_ratio": self.args.mask_ratio,
            "val_ratio": self.args.val_ratio,
        })

        return {
            "psnr_full": psnr_full,
            "ms_ssim_full": ms_ssim_full,
            "training_time": training_time,
            "psnr_train_final": psnr_train_final,
            "psnr_val_final": psnr_val_final,
            "psnr_obs_final": psnr_obs_final,
            "psnr_missing_final": psnr_missing_final,
            "best_iteration": self.best_iteration,
        }


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Split-based Gabor/Gaussian inpainting training.")

    parser.add_argument("--image_path", type=str, required=False,
                        help="Path to a single input image.")
    parser.add_argument("--dataset_path", type=str, required=False,
                        help="Path to a dataset.")
    parser.add_argument("--data_name", type=str, default="kodak",
                        help="Dataset name (for logging).")
    parser.add_argument("--normalize_mode", type=str, default="auto",
                        choices=["auto", "minmax", "none"],
                        help="Normalize images to [0, 1]: auto, minmax, or none.")

    parser.add_argument("--mask_type", type=str, default="random",
                        choices=["random", "elementwise", "block"],
                        help="Mask type: random, elementwise, or block.")
    parser.add_argument("--mask_ratio", type=float, default=0.5,
                        help="Fraction of pixels or elements to drop.")
    parser.add_argument("--block_size", type=int, default=64,
                        help="Block side length for block mask.")
    parser.add_argument("--num_blocks", type=int, default=4,
                        help="Number of blocks to drop for block mask.")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="Fraction of observed entries reserved for validation.")

    parser.add_argument("--num_points", type=int, default=50000,
                        help="Number of 2D Gaussian points.")
    parser.add_argument("--num_gabor", type=int, default=2,
                        help="Number of Gabor frequencies per Gaussian.")
    parser.add_argument("--iterations", type=int, default=30000,
                        help="Number of training iterations.")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate.")
    parser.add_argument("--loss_type", type=str, default="L2",
                        choices=["L1", "L2"],
                        help="Fidelity loss type on observed pixels.")

    parser.add_argument("--lambda_gabor_l1", type=float, default=0.0,
                        help="Weight for L1 sparsity regularization on gabor_weights.")
    parser.add_argument("--lambda_cholesky_l2", type=float, default=0.0,
                        help="Weight for L2 regularization on Cholesky parameters.")
    parser.add_argument("--lambda_tv", type=float, default=0.0,
                        help="Weight for TV regularization on rendered image.")
    parser.add_argument("--lambda_nuclear", type=float, default=0.0,
                        help="Weight for nuclear norm regularization on rendered image.")

    # Initialization strategy
    parser.add_argument("--init_strategy", type=str, default="random",
                        choices=["random", "obs_first"],
                        help="Initialization strategy: 'random' (default baseline) or "
                             "'obs_first' (concentrate initial Gaussians on observed pixels).")
    parser.add_argument("--obs_init_ratio", type=float, default=0.8,
                        help="Fraction of Gaussians placed on observed pixels "
                             "(only used when init_strategy=obs_first). Default: 0.8.")

    parser.add_argument("--save_imgs", action="store_true", default=True,
                        help="Save output images.")
    parser.add_argument("--save_observation", action="store_true", default=True,
                        help="Save observed image and masks at training start.")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to a pretrained checkpoint.")
    parser.add_argument("--seed", type=int, default=1,
                        help="Random seed for reproducibility.")
    parser.add_argument("--log_interval", type=int, default=10,
                        help="How often to log PSNR values.")

    return parser.parse_args(argv)


def run_single_image(args, args_text):
    trainer = SplitInpaintingTrainer(args)
    results = trainer.train()
    save_run_config(trainer.log_dir, args_text)

    print("\n=== Split Inpainting Result ===")
    print(f"Image: {args.image_path}")
    print(f"Mask: {args.mask_type}, ratio={args.mask_ratio}, val_ratio={args.val_ratio}")
    print(f"Best Iteration:           {results['best_iteration']}")
    print(f"PSNR(recon vs gt):        {results['psnr_full']:.4f}")
    print(f"PSNR(recon train_obs):    {results['psnr_train_final']:.4f}")
    print(f"PSNR(recon val_obs):      {results['psnr_val_final']:.4f}")
    print(f"PSNR(recon obs):          {results['psnr_obs_final']:.4f}")
    print(f"PSNR(recon missing):      {results['psnr_missing_final']:.4f}")
    print(f"MS-SSIM(recon vs gt):     {results['ms_ssim_full']:.6f}")
    print(f"Training Time:            {results['training_time']:.2f}s")


def run_dataset(args, args_text):
    init_tag = args.init_strategy
    if args.init_strategy == 'obs_first':
        init_tag = f"obs_first_{args.obs_init_ratio}"
    dataset_log_dir = Path(
        f"./checkpoints_inpainting_split/{args.data_name}/"
        f"{args.mask_type}_{args.mask_ratio}_val{args.val_ratio}/"
        f"GaussianImage_Cholesky_{args.iterations}_{args.num_points}_{args.num_gabor}_{init_tag}"
    )
    logwriter = LogWriter(dataset_log_dir)
    save_run_config(dataset_log_dir, args_text)

    image_paths = build_dataset_image_list(args.dataset_path, args.data_name)
    if not image_paths:
        raise ValueError(f"No images found under dataset path: {args.dataset_path}")

    psnrs_gt = []
    psnrs_val = []
    psnrs_missing = []
    ms_ssims = []
    training_times = []
    image_h = 0
    image_w = 0

    for image_path in image_paths:
        if not image_path.exists():
            print(f"Skipping missing image: {image_path}")
            continue

        trainer_args = copy.deepcopy(args)
        trainer_args.image_path = str(image_path)
        trainer = SplitInpaintingTrainer(trainer_args)
        results = trainer.train()
        save_run_config(trainer.log_dir, args_text)

        psnrs_gt.append(results["psnr_full"])
        psnrs_val.append(results["psnr_val_final"])
        psnrs_missing.append(results["psnr_missing_final"])
        ms_ssims.append(results["ms_ssim_full"])
        training_times.append(results["training_time"])
        image_h += trainer.H
        image_w += trainer.W

        logwriter.write(
            "{}: {}x{}, best_iter:{}, PSNR(gt):{:.4f}, PSNR(val_obs):{:.4f}, "
            "PSNR(missing):{:.4f}, MS-SSIM:{:.6f}, Training:{:.4f}s".format(
                image_path.stem,
                trainer.H,
                trainer.W,
                results["best_iteration"],
                results["psnr_full"],
                results["psnr_val_final"],
                results["psnr_missing_final"],
                results["ms_ssim_full"],
                results["training_time"],
            )
        )

    if not psnrs_gt:
        raise ValueError(f"No valid images were processed from dataset path: {args.dataset_path}")

    image_count = len(psnrs_gt)
    logwriter.write(
        "Average: {}x{}, PSNR(gt):{:.4f}, PSNR(val_obs):{:.4f}, PSNR(missing):{:.4f}, MS-SSIM:{:.6f}, Training:{:.4f}s".format(
            image_h // image_count,
            image_w // image_count,
            torch.tensor(psnrs_gt).mean().item(),
            torch.tensor(psnrs_val).mean().item(),
            torch.tensor(psnrs_missing).mean().item(),
            torch.tensor(ms_ssims).mean().item(),
            torch.tensor(training_times).mean().item(),
        )
    )


def main(argv):
    args = parse_args(argv)
    if (args.image_path is None) == (args.dataset_path is None):
        raise ValueError("Provide exactly one of --image_path or --dataset_path.")

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    args_text = yaml.safe_dump(vars(args), default_flow_style=False)
    if args.dataset_path is not None:
        run_dataset(args, args_text)
        return
    run_single_image(args, args_text)


if __name__ == "__main__":
    main(sys.argv[1:])