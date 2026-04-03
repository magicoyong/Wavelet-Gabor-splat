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

from utils import LogWriter
from inpainting_utils import (
    generate_mask,
    masked_loss_fn,
    gabor_weights_l1_reg,
    position_l2_reg,
    cholesky_l2_reg,
    save_tensor_as_image,
    compute_error_map,
    mask_to_image_tensor,
    compute_psnr,
    compute_ms_ssim,
)


def image_path_to_tensor(image_path):
    img = Image.open(image_path)
    transform = transforms.ToTensor()
    img_tensor = transform(img).unsqueeze(0)  # [1, C, H, W]
    return img_tensor


class InpaintingTrainer:
    """Trains Gabor/Gaussian splatting to solve masked inpainting."""

    def __init__(self, args):
        self.device = torch.device("cuda:0")
        self.args = args

        # Load ground truth
        self.gt_image = image_path_to_tensor(args.image_path).to(self.device)
        self.H, self.W = self.gt_image.shape[2], self.gt_image.shape[3]
        self.image_name = Path(args.image_path).stem

        # Generate mask
        self.mask = generate_mask(
            self.H, self.W,
            mask_type=args.mask_type,
            mask_ratio=args.mask_ratio,
            block_size=args.block_size,
            num_blocks=args.num_blocks,
            device=self.device,
        )

        # Observed image: gt * mask (missing regions set to 0)
        self.observed_image = self.gt_image * self.mask

        # Output directory
        self.log_dir = Path(
            f"./checkpoints_inpainting/{args.mask_type}_{args.mask_ratio}/"
            f"GaussianImage_Cholesky_{args.iterations}_{args.num_points}_{args.num_gabor}/"
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

    def train_step(self, iteration):
        """Single inpainting training step.

        Returns:
            loss: total loss (data_fidelity + regularization)
            psnr_gt: PSNR of full reconstruction vs full GT
            psnr_obs: PSNR on observed region only
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
            reg_loss = reg_loss + self.args.lambda_gabor_l1 * gabor_weights_l1_reg(self.model)
        if self.args.lambda_position_l2 > 0:
            reg_loss = reg_loss + self.args.lambda_position_l2 * position_l2_reg(self.model)
        if self.args.lambda_cholesky_l2 > 0:
            reg_loss = reg_loss + self.args.lambda_cholesky_l2 * cholesky_l2_reg(self.model)

        loss = data_fidelity + reg_loss

        # Backward
        loss.backward()
        self.model.optimizer.step()
        self.model.optimizer.zero_grad(set_to_none=True)
        self.model.scheduler.step()

        with torch.no_grad():
            # PSNR on full image vs GT (real inpainting quality)
            mse_full = F.mse_loss(image, self.gt_image)
            psnr_gt = 10 * math.log10(1.0 / mse_full.item()) if mse_full.item() > 0 else float('inf')

            # PSNR on observed region only (data consistency)
            masked_pred = image * self.mask
            masked_gt = self.gt_image * self.mask
            num_obs = self.mask.sum().clamp(min=1.0)
            mse_obs = (masked_pred - masked_gt).pow(2).sum() / (num_obs * image.shape[1])
            psnr_obs = 10 * math.log10(1.0 / mse_obs.item()) if mse_obs.item() > 0 else float('inf')

        return loss, psnr_gt, psnr_obs

    def evaluate(self):
        """Evaluate on full ground truth image (not just observed region).

        Returns PSNR, MS-SSIM on the complete image.
        """
        self.model.eval()
        with torch.no_grad():
            out = self.model()
            image = out["render"]
            psnr = compute_psnr(image, self.gt_image)
            ms_ssim_value = compute_ms_ssim(image, self.gt_image)
        return psnr, ms_ssim_value, image

    def save_results(self, reconstruction):
        """Save ground truth, observation, reconstruction, mask, and error map."""
        save_tensor_as_image(self.observed_image, self.log_dir / f"{self.image_name}_observed.png")
        save_tensor_as_image(reconstruction, self.log_dir / f"{self.image_name}_reconstruction.png")
        save_tensor_as_image(mask_to_image_tensor(self.mask), self.log_dir / f"{self.image_name}_mask.png")

        # Error map (amplified for visibility)
        error_map = compute_error_map(reconstruction, self.gt_image)
        # Scale error to [0, 1] for visualization (clamp max at 0.5 for contrast)
        error_vis = (error_map / 0.5).clamp(0, 1)
        save_tensor_as_image(error_vis, self.log_dir / f"{self.image_name}_error.png")

    def train(self):
        """Full inpainting training loop."""
        psnr_gt_list, psnr_obs_list, iter_list = [], [], []
        progress_bar = tqdm(range(1, self.args.iterations + 1), desc="Inpainting progress")
        self.model.train()

        # Save observation image at start
        if self.args.save_observation:
            save_tensor_as_image(self.observed_image, self.log_dir / f"{self.image_name}_observed.png")
            save_tensor_as_image(mask_to_image_tensor(self.mask), self.log_dir / f"{self.image_name}_mask.png")

        start_time = time.time()
        for it in range(1, self.args.iterations + 1):
            loss, psnr_gt, psnr_obs = self.train_step(it)
            psnr_gt_list.append(psnr_gt)
            psnr_obs_list.append(psnr_obs)
            iter_list.append(it)

            with torch.no_grad():
                if it % 10 == 0:
                    progress_bar.set_postfix({
                        "Loss": f"{loss.item():.7f}",
                        "PSNR(gt)": f"{psnr_gt:.4f}",
                        "PSNR(obs)": f"{psnr_obs:.4f}",
                    })
                    progress_bar.update(10)

        end_time = time.time() - start_time
        progress_bar.close()

        # Final evaluation on full image
        psnr_full, ms_ssim_full, reconstruction = self.evaluate()
        self.logwriter.write(
            f"Inpainting Complete in {end_time:.4f}s | "
            f"Full PSNR: {psnr_full:.4f} | Full MS-SSIM: {ms_ssim_full:.6f} | "
            f"mask_type={self.args.mask_type}, mask_ratio={self.args.mask_ratio}"
        )

        # Save model and results
        torch.save(self.model.state_dict(), self.log_dir / "gaussian_model.pth.tar")
        if self.args.save_imgs:
            self.save_results(reconstruction)

        np.save(self.log_dir / "inpainting_results.npy", {
            "iterations": iter_list,
            "training_psnr_gt": psnr_gt_list,
            "training_psnr_obs": psnr_obs_list,
            "training_time": end_time,
            "psnr_full": psnr_full,
            "ms_ssim_full": ms_ssim_full,
            "mask_type": self.args.mask_type,
            "mask_ratio": self.args.mask_ratio,
        })

        return psnr_full, ms_ssim_full, end_time


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Gabor/Gaussian inpainting training.")

    # Data
    parser.add_argument("--image_path", type=str, required=True,
                        help="Path to a single input image.")
    parser.add_argument("--data_name", type=str, default="kodak",
                        help="Dataset name (for logging).")

    # Mask
    parser.add_argument("--mask_type", type=str, default="random",
                        choices=["random", "block"],
                        help="Mask type: 'random' (pixel-wise) or 'block' (square blocks).")
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

    # Regularization
    parser.add_argument("--lambda_gabor_l1", type=float, default=0.0,
                        help="Weight for L1 sparsity regularization on gabor_weights.")
    
    parser.add_argument("--lambda_cholesky_l2", type=float, default=0.0,
                        help="Weight for L2 regularization on Cholesky (covariance) parameters.")

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

    trainer = InpaintingTrainer(args)
    psnr, ms_ssim_val, training_time = trainer.train()

    print(f"\n=== Inpainting Result ===")
    print(f"Image: {args.image_path}")
    print(f"Mask: {args.mask_type}, ratio={args.mask_ratio}")
    print(f"Full Image PSNR: {psnr:.4f}")
    print(f"Full Image MS-SSIM: {ms_ssim_val:.6f}")
    print(f"Training Time: {training_time:.2f}s")

    # Save config to output dir
    config_path = trainer.log_dir / "config.yaml"
    with open(config_path, "w") as f:
        f.write(args_text)


if __name__ == "__main__":
    main(sys.argv[1:])
