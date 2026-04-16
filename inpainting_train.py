"""
Inpainting training entry point using Gabor/Gaussian parameterization.

Solves the inverse problem: given a partially observed image (masked),
recover the full image by optimizing Gabor/Gaussian splatting parameters
with masked data-fidelity loss + optional regularization.

This file is independent from train.py. It reuses GaussianImage_Cholesky
as the differentiable renderer and adds inpainting-specific logic.
"""

import math
import time
from pathlib import Path
import argparse
import copy
import yaml
import numpy as np
import torch
import sys
import random
import torch.nn.functional as F
from PIL import Image
from pytorch_msssim import ms_ssim
import torchvision.transforms as transforms
from tqdm import tqdm

from utils import LogWriter, dwt_loss
from inpainting_utils import (
    coverage_map_to_image_tensor,
    generate_mask,
    masked_loss_fn,
    tv_loss,
    gabor_freq_l1_reg,
    save_tensor_as_image,
    compute_error_map,
    compute_inpainting_psnrs,
    compute_region_error_map,
    mask_to_image_tensor,
    compute_psnr,
    compute_ms_ssim,
    gabor_weights_l2_reg,
    nuclear_norm_reg,
    get_missing_mask,
)
from init_strategies import apply_obs_first_init


def normalize_image_tensor(img_tensor, normalize_mode="auto"):
    """Normalize image tensor to [0, 1].

    Modes:
        auto: keep tensors already in [0, 1]; divide by 255 if in [0, 255];
            otherwise apply min-max normalization.
        minmax: always apply min-max normalization.
        none: keep values unchanged.
    """
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
    img_tensor = transform(img).unsqueeze(0)  # [1, C, H, W]
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


class InpaintingTrainer:
    """Trains Gabor/Gaussian splatting to solve masked inpainting."""

    def __init__(self, args):
        self.device = torch.device("cuda:0")
        self.args = args

        # Load ground truth
        self.gt_image = image_path_to_tensor(
            args.image_path,
            normalize_mode=args.normalize_mode,
        ).to(self.device)
        self.H, self.W = self.gt_image.shape[2], self.gt_image.shape[3]
        self.image_name = Path(args.image_path).stem

        # Generate mask
        self.mask = generate_mask(
            self.H, self.W,
            mask_type=args.mask_type,
            mask_ratio=args.mask_ratio,
            block_size=args.block_size,
            num_blocks=args.num_blocks,
            C=self.gt_image.shape[1],
            device=self.device,
        )

        # Observed image: gt * mask (missing regions set to 0)
        self.observed_image = self.gt_image * self.mask
        self.missing_mask = get_missing_mask(self.mask)

        # Baseline PSNR: masked observation vs GT (constant, shows degradation severity)
        mse_masked = F.mse_loss(self.observed_image, self.gt_image)
        self.psnr_masked = 10 * math.log10(1.0 / mse_masked.item()) if mse_masked.item() > 0 else float('inf')

        # Output directory
        init_tag = args.init_strategy
        if args.init_strategy == 'obs_first':
            init_tag = f"obs_first_{args.obs_init_ratio}"
        self.log_dir = Path(
            f"./checkpoints_inpainting/{args.mask_type}_{args.mask_ratio}/"
            f"GaussianImage_Cholesky_{args.iterations}_{args.num_points}_{args.num_gabor}_{init_tag}/"
            f"{self.image_name}"
        )
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logwriter = LogWriter(self.log_dir)

        # Build model (reuse GaussianImage_Cholesky directly)
        BLOCK_H, BLOCK_W = 16, 16
        from gaussianimage_cholesky import GaussianImage_Cholesky
        self.model = GaussianImage_Cholesky(
            loss_type="L2",
            opt_type="adan",
            num_points=args.num_points,
            H=self.H,
            W=self.W,
            BLOCK_H=BLOCK_H,
            BLOCK_W=BLOCK_W,
            device=self.device,
            lr=args.lr,
            num_gabor=args.num_gabor,
            quantize=False,
        ).to(self.device)

        # Load pretrained model if provided
        if args.model_path is not None:
            print(f"Loading pretrained model: {args.model_path}")
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
                self.model, self.mask, self.gt_image,
                obs_init_ratio=obs_ratio,
            )

    def train_step(self, iteration):
        """Single inpainting training step.

        Returns:
            loss: total loss (data_fidelity + regularization)
            psnr_gt: PSNR of full reconstruction vs full GT
            psnr_obs: PSNR on observed region only
            psnr_missing: PSNR on missing region only
        """
        render_pkg = self.model.forward()
        image = render_pkg["render"]  # (1, C, H, W)

        # --- Data fidelity: only on observed pixels ---
        data_fidelity = masked_loss_fn(
            image, self.gt_image, self.mask, loss_type=self.args.loss_type
        )

        # --- Regularization ---
        reg_loss = torch.tensor(0.0, device=self.device)
        
        if self.args.lambda_gabor_l1 > 0:
            reg_loss = reg_loss + self.args.lambda_gabor_l1 * gabor_freq_l1_reg(self.model)
        if self.args.lambda_weights_l2 > 0:
            reg_loss = reg_loss + self.args.lambda_weights_l2 * gabor_weights_l2_reg(self.model)
        if self.args.lambda_tv > 0:
            reg_loss = reg_loss + self.args.lambda_tv * tv_loss(image)
        if self.args.lambda_nuclear > 0:
            reg_loss = reg_loss + self.args.lambda_nuclear * nuclear_norm_reg(image)
        if self.args.lambda_dwt > 0:
            reg_loss = reg_loss + self.args.lambda_dwt * dwt_loss(image)

        loss = data_fidelity + reg_loss

        # Backward
        loss.backward()
        self.model.optimizer.step()
        self.model.optimizer.zero_grad(set_to_none=True)
        self.model.scheduler.step()

        with torch.no_grad():
            psnr_metrics = compute_inpainting_psnrs(image, self.gt_image, self.mask)

        return (
            loss,
            psnr_metrics["psnr_full"],
            psnr_metrics["psnr_observed"],
            psnr_metrics["psnr_missing"],
        )

    def evaluate(self):
        """Evaluate on full ground truth image (not just observed region).

        Returns full / observed / missing PSNR, MS-SSIM, and reconstruction.
        """
        self.model.eval()
        with torch.no_grad():
            out = self.model()
            image = out["render"]
            psnr_metrics = compute_inpainting_psnrs(image, self.gt_image, self.mask)
            ms_ssim_value = compute_ms_ssim(image, self.gt_image)
        return psnr_metrics, ms_ssim_value, image

    def save_results(self, reconstruction):
        """Save ground truth, observation, reconstruction, mask, and error map."""
        save_tensor_as_image(self.observed_image, self.log_dir / f"{self.image_name}_observed.png")
        save_tensor_as_image(reconstruction, self.log_dir / f"{self.image_name}_reconstruction.png")
        save_tensor_as_image(mask_to_image_tensor(self.mask), self.log_dir / f"{self.image_name}_mask.png")
        if self.mask.shape[1] > 1:
            save_tensor_as_image(
                coverage_map_to_image_tensor(self.mask),
                self.log_dir / f"{self.image_name}_coverage.png",
            )

        # Error map (amplified for visibility)
        error_map = compute_error_map(reconstruction, self.gt_image)
        observed_error_map = compute_region_error_map(reconstruction, self.gt_image, self.mask)
        missing_error_map = compute_region_error_map(reconstruction, self.gt_image, self.missing_mask)
        # Scale error to [0, 1] for visualization (clamp max at 0.5 for contrast)
        error_vis = (error_map / 0.5).clamp(0, 1)
        observed_error_vis = (observed_error_map / 0.5).clamp(0, 1)
        missing_error_vis = (missing_error_map / 0.5).clamp(0, 1)
        save_tensor_as_image(error_vis, self.log_dir / f"{self.image_name}_error.png")
        save_tensor_as_image(observed_error_vis, self.log_dir / f"{self.image_name}_observed_error.png")
        save_tensor_as_image(missing_error_vis, self.log_dir / f"{self.image_name}_missing_error.png")

    def train(self):
        """Full inpainting training loop."""
        psnr_gt_list, psnr_obs_list, psnr_missing_list, iter_list = [], [], [], []
        progress_bar = tqdm(range(1, self.args.iterations + 1), desc="Inpainting progress")
        self.model.train()

        # Save observation image at start
        if self.args.save_observation:
            save_tensor_as_image(self.observed_image, self.log_dir / f"{self.image_name}_observed.png")
            save_tensor_as_image(mask_to_image_tensor(self.mask), self.log_dir / f"{self.image_name}_mask.png")

        start_time = time.time()
        for it in range(1, self.args.iterations + 1):
            loss, psnr_gt, psnr_obs, psnr_missing = self.train_step(it)
            psnr_gt_list.append(psnr_gt)
            psnr_obs_list.append(psnr_obs)
            psnr_missing_list.append(psnr_missing)
            iter_list.append(it)

            with torch.no_grad():
                if it % 10 == 0:
                    progress_bar.set_postfix({
                        "Loss": f"{loss.item():.7f}",
                        "PSNR(gt)": f"{psnr_gt:.4f}",
                        "PSNR(obs)": f"{psnr_obs:.4f}",
                        "PSNR(miss)": f"{psnr_missing:.4f}",
                    })
                    progress_bar.update(10)

        end_time = time.time() - start_time
        progress_bar.close()

        # Final evaluation on full image
        psnr_metrics, ms_ssim_full, reconstruction = self.evaluate()
        psnr_full = psnr_metrics["psnr_full"]
        psnr_recon_obs = psnr_metrics["psnr_observed"]
        psnr_recon_missing = psnr_metrics["psnr_missing"]

        self.final_psnr_full = psnr_full
        self.final_psnr_observed = psnr_recon_obs
        self.final_psnr_missing = psnr_recon_missing
        self.final_ms_ssim = ms_ssim_full

        self.logwriter.write(
            f"Inpainting Complete in {end_time:.4f}s\n"
            f"  PSNR(masked vs gt):      {self.psnr_masked:.4f}  (degradation baseline)\n"
            f"  PSNR(recon vs gt):       {psnr_full:.4f}  (reconstruction quality)\n"
            f"  PSNR(recon_obs vs gt_obs): {psnr_recon_obs:.4f}  (data consistency)\n"
            f"  PSNR(recon_missing vs gt_missing): {psnr_recon_missing:.4f}  (missing-region recovery)\n"
            f"  MS-SSIM(recon vs gt):    {ms_ssim_full:.6f}\n"
            f"  mask_type={self.args.mask_type}, mask_ratio={self.args.mask_ratio}"
        )

        # Save model and results
        torch.save(self.model.state_dict(), self.log_dir / "gaussian_model.pth.tar")
        if self.args.save_imgs:
            self.save_results(reconstruction)

        np.save(self.log_dir / "inpainting_results.npy", {
            "iterations": iter_list,
            "training_psnr_gt": psnr_gt_list,
            "training_psnr_obs": psnr_obs_list,
            "training_psnr_missing": psnr_missing_list,
            "training_time": end_time,
            "psnr_masked": self.psnr_masked,
            "psnr_full": psnr_full,
            "psnr_recon_obs": psnr_recon_obs,
            "psnr_recon_missing": psnr_recon_missing,
            "ms_ssim_full": ms_ssim_full,
            "mask_type": self.args.mask_type,
            "mask_ratio": self.args.mask_ratio,
        })

        return psnr_full, ms_ssim_full, end_time


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Gabor/Gaussian inpainting training.")

    # Data
    parser.add_argument("--image_path", type=str, required=False,
                        help="Path to a single input image.")
    parser.add_argument("--dataset_path", type=str, required=False,
                        help="Path to a dataset.")
    parser.add_argument("--data_name", type=str, default="kodak",
                        help="Dataset name (for logging).")
    parser.add_argument("--normalize_mode", type=str, default="auto",
                        choices=["auto", "minmax", "none"],
                        help="Normalize images to [0, 1]: auto, minmax, or none.")

    # Mask
    parser.add_argument("--mask_type", type=str, default="random",
                        choices=["random", "elementwise", "block"],
                        help="Mask type: 'random' (pixel-wise), "
                             "'elementwise' (per-channel independent, like GSLR), "
                             "or 'block' (square blocks).")
    parser.add_argument("--mask_ratio", type=float, default=0.5,
                        help="Fraction of pixels to drop (for random mask).")
    parser.add_argument("--block_size", type=int, default=64,
                        help="Block side length (for block mask).")
    parser.add_argument("--num_blocks", type=int, default=4,
                        help="Number of blocks to drop (for block mask).")

    # Model
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

    # # Regularization
    parser.add_argument("--lambda_gabor_l1", type=float, default=0.0,
                        help="Weight for L1 sparsity regularization on gabor_freqs.")
    
    parser.add_argument("--lambda_weights_l2", type=float, default=0.0,
                        help="Weight for L2 regularization on gabor weigths.")
    parser.add_argument("--lambda_tv", type=float, default=0.0,
                        help="Weight for total variation regularization on the rendered image.")
    parser.add_argument("--lambda_dwt", type=float, default=0.0,
                        help="Weight for high frequency details in the rendered image.")
    parser.add_argument("--lambda_nuclear", type=float, default=0.0,
                        help="Weight for nuclear norm regularization on rendered image (low-rank prior).")

    # Initialization strategy
    parser.add_argument("--init_strategy", type=str, default="random",
                        choices=["random", "obs_first"],
                        help="Initialization strategy: 'random' (default baseline) or "
                             "'obs_first' (concentrate initial Gaussians on observed pixels).")
    parser.add_argument("--obs_init_ratio", type=float, default=0.8,
                        help="Fraction of Gaussians placed on observed pixels "
                             "(only used when init_strategy=obs_first). Default: 0.8.")

    # Saving
    parser.add_argument("--save_imgs", action="store_true", default=True,
                        help="Save output images (gt, observed, reconstruction, mask, error).")
    parser.add_argument("--save_observation", action="store_true", default=True,
                        help="Save observed image and mask at training start.")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to a pretrained checkpoint (optional).")

    # Misc
    parser.add_argument("--seed", type=int, default=1,
                        help="Random seed for reproducibility.")

    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)

    if (args.image_path is None) == (args.dataset_path is None):
        raise ValueError("Provide exactly one of --image_path or --dataset_path.")

    # Reproducibility
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    # Save config
    args_text = yaml.safe_dump(vars(args), default_flow_style=False)

    if args.dataset_path is not None:
        init_tag = args.init_strategy
        if args.init_strategy == 'obs_first':
            init_tag = f"obs_first_{args.obs_init_ratio}"
        dataset_log_dir = Path(
            f"./checkpoints_inpainting/{args.data_name}/"
            f"{args.mask_type}_{args.mask_ratio}/"
            f"GaussianImage_Cholesky_{args.iterations}_{args.num_points}_{args.num_gabor}_{init_tag}"
        )
        logwriter = LogWriter(dataset_log_dir)
        save_run_config(dataset_log_dir, args_text)

        psnrs_gt, psnrs_missing, ms_ssims_gt, training_times, psnrs_masked = [], [], [], [], []
        image_h, image_w = 0, 0
        image_paths = build_dataset_image_list(args.dataset_path, args.data_name)
        if not image_paths:
            raise ValueError(f"No images found under dataset path: {args.dataset_path}")

        for image_path in image_paths:
            if not image_path.exists():
                print(f"Skipping missing image: {image_path}")
                continue

            trainer_args = copy.deepcopy(args)
            trainer_args.image_path = str(image_path)
            trainer = InpaintingTrainer(trainer_args)
            psnr, ms_ssim_val, training_time = trainer.train()
            save_run_config(trainer.log_dir, args_text)

            print(f"\n=== Inpainting Result ===")
            print(f"Image: {trainer_args.image_path}")
            print(f"Mask: {trainer_args.mask_type}, ratio={trainer_args.mask_ratio}")
            print(f"PSNR(masked vs gt):    {trainer.psnr_masked:.4f}  (degradation baseline)")
            print(f"PSNR(recon vs gt):     {psnr:.4f}  (reconstruction quality)")
            print(f"PSNR(recon miss vs gt miss): {trainer.final_psnr_missing:.4f}")
            print(f"MS-SSIM(recon vs gt):  {ms_ssim_val:.6f}")
            print(f"Training Time: {training_time:.2f}s")

            psnrs_gt.append(psnr)
            psnrs_missing.append(trainer.final_psnr_missing)
            ms_ssims_gt.append(ms_ssim_val)
            training_times.append(training_time)
            psnrs_masked.append(trainer.psnr_masked)
            image_h += trainer.H
            image_w += trainer.W
            logwriter.write(
                "{}: {}x{}, PSNR(masked vs gt):{:.4f}, PSNR(recon vs gt):{:.4f}, "
                "PSNR(recon missing):{:.4f}, MS-SSIM:{:.6f}, Training:{:.4f}s".format(
                    image_path.stem,
                    trainer.H,
                    trainer.W,
                    trainer.psnr_masked,
                    psnr,
                    trainer.final_psnr_missing,
                    ms_ssim_val,
                    training_time,
                )
            )

        if not psnrs_gt:
            raise ValueError(f"No valid images were processed from dataset path: {args.dataset_path}")

        image_count = len(psnrs_gt)
        avg_psnr_gt = torch.tensor(psnrs_gt).mean().item()
        avg_psnr_missing = torch.tensor(psnrs_missing).mean().item()
        avg_ms_ssim_gt = torch.tensor(ms_ssims_gt).mean().item()
        avg_training_time = torch.tensor(training_times).mean().item()
        avg_psnr_mask = torch.tensor(psnrs_masked).mean().item()
        avg_h = image_h // image_count
        avg_w = image_w // image_count

        logwriter.write(
            "Average: {}x{}, PSNR(masked vs gt):{:.4f}, PSNR(recon vs gt):{:.4f}, "
            "PSNR(recon missing):{:.4f}, MS-SSIM:{:.6f}, Training:{:.4f}s".format(
                avg_h,
                avg_w,
                avg_psnr_mask,
                avg_psnr_gt,
                avg_psnr_missing,
                avg_ms_ssim_gt,
                avg_training_time,
            )
        )
        return

    trainer = InpaintingTrainer(args)
    psnr, ms_ssim_val, training_time = trainer.train()

    print(f"\n=== Inpainting Result ===")
    print(f"Image: {args.image_path}")
    print(f"Mask: {args.mask_type}, ratio={args.mask_ratio}")
    print(f"PSNR(masked vs gt):    {trainer.psnr_masked:.4f}  (degradation baseline)")
    print(f"PSNR(recon vs gt):     {psnr:.4f}  (reconstruction quality)")
    print(f"PSNR(recon miss vs gt miss): {trainer.final_psnr_missing:.4f}")
    print(f"MS-SSIM(recon vs gt):  {ms_ssim_val:.6f}")
    print(f"Training Time: {training_time:.2f}s")

    save_run_config(trainer.log_dir, args_text)


if __name__ == "__main__":
    main(sys.argv[1:])
