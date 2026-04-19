"""
HSI Low-Rank Gabor Inpainting Training Entry Point.

Pipeline:
    1. Load HSI data, generate mask.
    2. Compute E0 via masked NMF on observed pixels only (NO GT leakage).
    3. Train GaussianImage_Cholesky_HSI with:
       - Abundance renderer (Gabor splatting)
       - Trainable endmember calibration:  E_hat = E0 + gamma * (U @ V)
       - Masked observation-consistency loss:
             L = || mask * (A_hat @ E_hat - Y_obs) ||
    4. Evaluate PSNR on full / observed / missing regions.

Usage:
    python inpainting_train_hsi.py --dataset JasperRidge --rank 10 \
        --mask_type random --mask_ratio 0.5 --num_points 600 --iterations 8000
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
import scipy.io
from tqdm import tqdm

from utils import LogWriter
from inpainting_utils import (
    generate_mask,
    compute_inpainting_psnrs,
    get_missing_mask,
    compute_ssim_hsi,
)
from endmember import masked_nmf_initialization


def load_hsi_dataset(name):
    """Load and normalize HSI dataset. Returns (H, W, C) numpy array."""
    name = name.lower()
    if name == "urban":
        I = scipy.io.loadmat("HSI/data/Urban_R162.mat")['Y'].astype(float)
        for i in range(162):
            I[i, :] /= np.max(I[i, :])
        I = I.reshape(162, 307, 307).transpose(2, 1, 0)  # (H, W, C)
    elif name == "salinas":
        I = scipy.io.loadmat("HSI/data/Salinas_crop.mat")['I'].astype(float)
        I = np.clip(I, 0, None)
        for i in range(I.shape[2]):
            I[:, :, i] /= np.max(I[:, :, i])
    elif name == "jasperridge":
        I = scipy.io.loadmat("HSI/data/jasperRidge2_R198.mat")['Y'].astype(float)
        for i in range(198):
            I[i, :] /= np.max(I[i, :])
        I = I.reshape(198, 100, 100).transpose(2, 1, 0)  # (H, W, C)
    elif name == "paviau":
        I = scipy.io.loadmat("HSI/data/PaviaU.mat")['paviaU'].astype(float)
        for i in range(103):
            I[:, :, i] /= np.max(I[:, :, i])
        I = I[-340:, :, :]
    else:
        raise ValueError(f"Unknown HSI dataset: {name}")
    return I  # (H, W, C)


def compute_sam(gt, pred):
    """Compute Spectral Angle Mapper (mean SAM in degrees)."""
    # gt, pred: (H, W, C) numpy arrays
    dot = np.sum(gt * pred, axis=-1)
    norm_gt = np.linalg.norm(gt, axis=-1)
    norm_pred = np.linalg.norm(pred, axis=-1)
    cos_angle = dot / (norm_gt * norm_pred + 1e-8)
    cos_angle = np.clip(cos_angle, -1, 1)
    angles = np.arccos(cos_angle) * 180 / np.pi
    return np.mean(angles)


class HSIInpaintingTrainer:
    """Trains Gabor splatting for HSI inpainting via low-rank decomposition.

    E0 is obtained from masked NMF (NO GT leakage).
    During training the model learns:
      - Abundance renderer (Gabor splatting)
      - Endmember low-rank correction U, V  →  E_hat = E0 + gamma*(U@V)
    """

    def __init__(self, args):
        self.device = torch.device("cuda:0")
        self.args = args

        # ── Load HSI data ───────────────────────────────────────────────
        I_np = load_hsi_dataset(args.dataset)  # (H, W, C)
        self.H, self.W, self.C = I_np.shape
        self.gt_image = torch.tensor(I_np, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        self.gt_image = torch.clamp(self.gt_image, 0, 1).to(self.device)

        # ── Generate mask ───────────────────────────────────────────────
        self.mask = generate_mask(
            self.H, self.W,
            mask_type=args.mask_type,
            mask_ratio=args.mask_ratio,
            block_size=getattr(args, 'block_size', 64),
            num_blocks=getattr(args, 'num_blocks', 4),
            C=self.C if args.mask_type == "elementwise" else 1,
            device=self.device,
        )
        self.observed_image = self.gt_image * self.mask
        self.missing_mask = get_missing_mask(self.mask)

        # Baseline PSNR (masked input vs GT)
        mse_masked = F.mse_loss(self.observed_image, self.gt_image)
        self.psnr_masked = (
            10 * math.log10(1.0 / mse_masked.item())
            if mse_masked.item() > 0 else float('inf')
        )

        # ── Masked NMF for E0  (NO GT leakage) ─────────────────────────
        print("Computing E0 via masked NMF (observed pixels only) ...")
        mask_np = self.mask.cpu().numpy()   # (1,1,H,W) or (1,C,H,W)
        E0, A0 = masked_nmf_initialization(
            I_np, mask_np, args.rank,
            dataset_name=args.dataset, save=False,
            max_iter=getattr(args, 'nmf_max_iter', 12000),
        )
        self.rank = E0.shape[0]
        assert E0.shape[1] == self.C

        # ── Output directory ────────────────────────────────────────────
        calib_tag = (
            f"calib{args.calib_rank}_g{args.gamma}"
            if not args.freeze_endmember_calibration
            else "E0only"
        )
        self.log_dir = Path(
            f"./checkpoints_inpainting_hsi/{args.dataset}/"
            f"{args.mask_type}_{args.mask_ratio}/"
            f"GaborHSI_{args.iterations}_{args.num_points}_{args.num_gabor}"
            f"_rank{args.rank}_{calib_tag}"
        )
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logwriter = LogWriter(self.log_dir)

        # ── Build model ─────────────────────────────────────────────────
        BLOCK_H, BLOCK_W = 16, 16
        from gaussianimage_cholesky_hsi import GaussianImage_Cholesky_HSI
        self.model = GaussianImage_Cholesky_HSI(
            loss_type=args.loss_type,
            opt_type=getattr(args, 'opt_type', 'adan'),
            num_points=args.num_points,
            H=self.H, W=self.W,
            rank=self.rank, C=self.C,
            E=E0,
            calib_rank=args.calib_rank,
            gamma=args.gamma,
            freeze_endmember_calibration=args.freeze_endmember_calibration,
            BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
            device=self.device,
            lr=args.lr,
            num_gabor=args.num_gabor,
        ).to(self.device)

        # Load pretrained model if provided
        if getattr(args, 'model_path', None) is not None:
            print(f"Loading pretrained model: {args.model_path}")
            checkpoint = torch.load(args.model_path, map_location=self.device)
            model_dict = self.model.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)

    def train_step(self, iteration):
        """Single HSI inpainting training step."""
        render_pkg = self.model.forward()
        image = render_pkg["render"]  # (1, C, H, W)

        # Masked data fidelity: ||M ⊙ (Y_hat - Y_obs)||
        masked_pred = image * self.mask
        masked_gt = self.gt_image * self.mask

        if self.args.loss_type == "L1":
            data_loss = (masked_pred - masked_gt).abs().sum() 
        else:
            data_loss = ((masked_pred - masked_gt) ** 2).sum() 

        loss = data_loss
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
            self.model.get_delta_E_norm(),
        )

    def evaluate(self):
        """Evaluate on full HSI."""
        self.model.eval()
        with torch.no_grad():
            out = self.model.forward()
            image = out["render"]  # (1, C, H, W)
            abundance = out["abundance"]  # (H, W, rank)
            psnr_metrics = compute_inpainting_psnrs(image, self.gt_image, self.mask)

            # SAM
            gt_np = self.gt_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
            pred_np = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
            sam_value = compute_sam(gt_np, pred_np)

            # SSIM (per-band mean)
            ssim_value = compute_ssim_hsi(image, self.gt_image)

            delta_norm = self.model.get_delta_E_norm()
        return psnr_metrics, sam_value, ssim_value, image, abundance, delta_norm

    def train(self):
        """Full HSI inpainting training loop."""
        psnr_gt_list, psnr_obs_list, psnr_missing_list = [], [], []
        progress_bar = tqdm(range(1, self.args.iterations + 1), desc="HSI Inpainting")
        self.model.train()
        start_time = time.time()

        best_psnr = 0
        best_model_dict = None

        for it in range(1, self.args.iterations + 1):
            loss, psnr_gt, psnr_obs, psnr_missing, delta_norm = self.train_step(it)
            psnr_gt_list.append(psnr_gt)
            psnr_obs_list.append(psnr_obs)
            psnr_missing_list.append(psnr_missing)

            if psnr_gt > best_psnr:
                best_psnr = psnr_gt
                best_model_dict = copy.deepcopy(self.model.state_dict())

            with torch.no_grad():
                if it % 10 == 0:
                    progress_bar.set_postfix({
                        "Loss": f"{loss.item():.7f}",
                        "PSNR(gt)": f"{psnr_gt:.4f}",
                        "PSNR(obs)": f"{psnr_obs:.4f}",
                        "PSNR(miss)": f"{psnr_missing:.4f}",
                        "||UV||": f"{delta_norm:.6f}",
                    })
                    progress_bar.update(10)

        end_time = time.time() - start_time
        progress_bar.close()

        # Final evaluation (last model)
        psnr_metrics, sam_value, ssim_value, reconstruction, abundance, dn = self.evaluate()

        # Also evaluate best model
        if best_model_dict is not None:
            self.model.load_state_dict(best_model_dict)
        best_metrics, best_sam, best_ssim, best_recon, best_abundance, best_dn = self.evaluate()

        calib_info = (
            f"  Calibration: freeze={self.args.freeze_endmember_calibration}, "
            f"calib_rank={self.args.calib_rank}, gamma={self.args.gamma}\n"
            f"  ||U@V|| (best):           {best_dn:.6f}\n"
        )

        self.logwriter.write(
            f"HSI Inpainting Complete in {end_time:.4f}s\n"
            f"  Dataset: {self.args.dataset}, Rank: {self.rank}, C: {self.C}\n"
            f"  Mask: {self.args.mask_type}, ratio={self.args.mask_ratio}\n"
            + calib_info +
            f"  PSNR(masked vs gt):        {self.psnr_masked:.4f}  (degradation baseline)\n"
            f"  PSNR(recon vs gt):         {psnr_metrics['psnr_full']:.4f}\n"
            f"  PSNR(recon_obs):           {psnr_metrics['psnr_observed']:.4f}\n"
            f"  PSNR(recon_missing):       {psnr_metrics['psnr_missing']:.4f}\n"
            f"  SAM:                       {sam_value:.4f}\n"
            f"  SSIM:                      {ssim_value:.4f}\n"
            f"  Best PSNR(recon vs gt):    {best_metrics['psnr_full']:.4f}\n"
            f"  Best PSNR(recon_missing):  {best_metrics['psnr_missing']:.4f}\n"
            f"  Best SAM:                  {best_sam:.4f}\n"
            f"  Best SSIM:                 {best_ssim:.4f}\n"
        )

        # Save models
        torch.save(self.model.state_dict(), self.log_dir / "gaussian_model.pth.tar")
        torch.save(best_model_dict, self.log_dir / "gaussian_model.best.pth.tar")

        # Save E0 and calibrated E_hat for analysis
        np.save(self.log_dir / "E0_maskedNMF.npy", self.model.E0.cpu().numpy())
        with torch.no_grad():
            E_hat_np = self.model.get_calibrated_endmember().cpu().numpy()
        np.save(self.log_dir / "E_hat_calibrated.npy", E_hat_np)

        # Save abundance map
        np.save(
            self.log_dir / "abundance.npy",
            best_abundance.cpu().numpy()
        )

        # Save results
        np.save(self.log_dir / "inpainting_results.npy", {
            "training_psnr_gt": psnr_gt_list,
            "training_psnr_obs": psnr_obs_list,
            "training_psnr_missing": psnr_missing_list,
            "training_time": end_time,
            "psnr_masked": self.psnr_masked,
            "psnr_full": psnr_metrics["psnr_full"],
            "psnr_obs": psnr_metrics["psnr_observed"],
            "psnr_missing": psnr_metrics["psnr_missing"],
            "sam": sam_value,
            "best_psnr_full": best_metrics["psnr_full"],
            "best_psnr_missing": best_metrics["psnr_missing"],
            "best_sam": best_sam,
            "ssim": ssim_value,
            "best_ssim": best_ssim,
            "delta_E_norm": best_dn,
            "dataset": self.args.dataset,
            "rank": self.rank,
            "mask_type": self.args.mask_type,
            "mask_ratio": self.args.mask_ratio,
            "calib_rank": self.args.calib_rank,
            "gamma": self.args.gamma,
            "freeze_endmember_calibration": self.args.freeze_endmember_calibration,
        })

        return best_metrics["psnr_full"], best_sam, best_ssim, end_time


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="HSI Low-Rank Gabor Inpainting")

    # Data
    parser.add_argument("--dataset", type=str, default="JasperRidge",
                        help="HSI dataset: Urban | Salinas | JasperRidge | PaviaU")
    parser.add_argument("--rank", type=int, default=10,
                        help="NMF rank (number of endmembers)")

    # Mask
    parser.add_argument("--mask_type", type=str, default="random",
                        choices=["random", "elementwise", "block"],
                        help="Mask type.")
    parser.add_argument("--mask_ratio", type=float, default=0.5,
                        help="Fraction of pixels to drop.")
    parser.add_argument("--block_size", type=int, default=16,
                        help="Block side length (for block mask).")
    parser.add_argument("--num_blocks", type=int, default=4,
                        help="Number of blocks to drop (for block mask).")

    # Model
    parser.add_argument("--num_points", type=int, default=600,
                        help="Number of 2D Gaussian points.")
    parser.add_argument("--num_gabor", type=int, default=2,
                        help="Number of Gabor frequencies per Gaussian.")
    parser.add_argument("--iterations", type=int, default=8000,
                        help="Number of training iterations.")
    parser.add_argument("--lr", type=float, default=5e-3,
                        help="Learning rate.")
    parser.add_argument("--loss_type", type=str, default="L2",
                        choices=["L1", "L2"],
                        help="Fidelity loss type.")
    parser.add_argument("--opt_type", type=str, default="adan",
                        choices=["adam", "adan"],
                        help="Optimizer type.")

    # Endmember calibration
    parser.add_argument("--calib_rank", type=int, default=2,
                        help="Rank of low-rank endmember correction (U@V).")
    parser.add_argument("--gamma", type=float, default=0.1,
                        help="Scaling factor for endmember correction: "
                             "E_hat = E0 + gamma*(U@V).")
    parser.add_argument("--freeze_endmember_calibration", action="store_true",
                        help="If set, only use E0 (no calibration). "
                             "Ablation switch.")
    parser.add_argument("--nmf_max_iter", type=int, default=12000,
                        help="Max iterations for masked NMF.")

    # Misc
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to pretrained checkpoint.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")

    args = parser.parse_args(argv)
    return args


def main(argv=None):
    args = parse_args(argv)

    # Reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Save config
    args_text = yaml.safe_dump(vars(args), default_flow_style=False)
    print(f"\n=== HSI Inpainting Configuration ===")
    print(args_text)

    trainer = HSIInpaintingTrainer(args)

    # Save config to log dir
    config_path = trainer.log_dir / "config.yaml"
    with open(config_path, "w") as f:
        f.write(args_text)

    psnr, sam, ssim, training_time = trainer.train()

    print(f"\n=== HSI Inpainting Result ===")
    print(f"Dataset: {args.dataset}, Rank: {args.rank}")
    print(f"Mask: {args.mask_type}, ratio={args.mask_ratio}")
    print(f"Calibration: freeze={args.freeze_endmember_calibration}, "
          f"calib_rank={args.calib_rank}, gamma={args.gamma}")
    print(f"PSNR(masked vs gt):    {trainer.psnr_masked:.4f}  (degradation baseline)")
    print(f"Best PSNR(recon vs gt): {psnr:.4f}")
    print(f"Best SAM:               {sam:.4f}")
    print(f"Best SSIM:              {ssim:.4f}")
    print(f"Training Time:          {training_time:.2f}s")


if __name__ == "__main__":
    main(sys.argv[1:])
