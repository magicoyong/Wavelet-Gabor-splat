"""
HSI Low-Rank Gabor Inpainting Model.

Renders abundance maps A (H, W, rank) via multi-channel Gabor splatting,
then reconstructs HSI via Y_hat = A @ E where E is the NMF endmember matrix.
Loss is computed in HSI space on observed pixels.
"""

from gsplat.project_gaussians_2d import project_gaussians_2d
from gsplat.rasterize_sum import rasterize_gabor_sum_nd
from utils import loss_fn
import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from optimizer import Adan


class GaussianImage_Cholesky_HSI(nn.Module):
    """Gabor splatting model for HSI inpainting via low-rank decomposition.

    Parameters are Gaussian positions, Cholesky covariance, per-Gaussian
    abundance features (rank channels), and Gabor frequency modulation.
    The endmember matrix E maps abundance to HSI channels.
    """

    def __init__(self, loss_type="L2", **kwargs):
        super().__init__()
        self.loss_type = loss_type
        self.init_num_points = kwargs["num_points"]
        self.H, self.W = kwargs["H"], kwargs["W"]
        self.rank = kwargs["rank"]
        self.C = kwargs["C"]  # number of HSI spectral channels
        self.BLOCK_W, self.BLOCK_H = kwargs["BLOCK_W"], kwargs["BLOCK_H"]
        self.tile_bounds = (
            (self.W + self.BLOCK_W - 1) // self.BLOCK_W,
            (self.H + self.BLOCK_H - 1) // self.BLOCK_H,
            1,
        )
        self.device = kwargs["device"]

        # Endmember matrix E: (rank, C), frozen (not optimized)
        E_np = kwargs["E"]  # numpy array (rank, C)
        self.register_buffer('endmember', torch.tensor(E_np, dtype=torch.float32))

        # Learnable Gaussian parameters
        self._xyz = nn.Parameter(torch.atanh(2 * (torch.rand(self.init_num_points, 2) - 0.5)))
        self._cholesky = nn.Parameter(torch.rand(self.init_num_points, 3))
        self.register_buffer('_opacity', torch.ones((self.init_num_points, 1)))
        # Abundance features: (N, rank) instead of (N, 3)
        self._features_dc = nn.Parameter(torch.rand(self.init_num_points, self.rank))

        # Gabor parameters
        self.num_gabor = kwargs.get("num_gabor", 2)
        self.gabor_freqs = nn.Parameter(
            (torch.rand(self.init_num_points * self.num_gabor, 2) - 0.5) * 4
        )
        self.gabor_weights = nn.Parameter(
            torch.rand(self.init_num_points * self.num_gabor, 1) * (-5)
        )

        self.last_size = (self.H, self.W)
        self.register_buffer('background', torch.ones(self.rank))
        self.register_buffer('bound', torch.tensor([0.5, 0.5]).view(1, 2))
        self.register_buffer('cholesky_bound', torch.tensor([0.5, 0, 0.5]).view(1, 3))

        # Optimizer
        self.lr = kwargs["lr"]
        if kwargs.get("opt_type", "adan") == "adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        else:
            self.optimizer = Adan(self.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=20000, gamma=0.5
        )

    @property
    def get_xyz(self):
        return torch.tanh(self._xyz)

    @property
    def get_features(self):
        return self._features_dc

    @property
    def get_opacity(self):
        return self._opacity

    @property
    def get_cholesky_elements(self):
        return self._cholesky + self.cholesky_bound

    @property
    def get_gabor_freqs(self):
        return self.gabor_freqs

    @property
    def get_gabor_weights(self):
        return torch.sigmoid(self.gabor_weights)

    @property
    def get_num_gabor(self):
        return self.num_gabor

    def forward(self):
        """Render abundance maps via Gabor splatting.

        Returns:
            dict with:
                "abundance": (H, W, rank) abundance map
                "render": (1, C, H, W) HSI reconstruction Y_hat = A @ E
        """
        self.xys, depths, self.radii, conics, num_tiles_hit = project_gaussians_2d(
            self.get_xyz, self.get_cholesky_elements, self.H, self.W, self.tile_bounds
        )
        # Multi-channel Gabor rendering: (H, W, rank)
        out_abundance = rasterize_gabor_sum_nd(
            self.xys, depths, self.radii, conics, num_tiles_hit,
            self.get_features, self._opacity,
            self.get_gabor_freqs[:, 0], self.get_gabor_freqs[:, 1],
            self.get_gabor_weights, self.get_num_gabor,
            self.H, self.W, self.BLOCK_H, self.BLOCK_W,
            background=self.background, return_alpha=False,
        )
        out_abundance = torch.clamp(out_abundance, min=0)  # (H, W, rank), non-negative only

        # HSI reconstruction: A @ E -> (H, W, C)
        A_flat = out_abundance.view(self.H * self.W, self.rank)  # (H*W, rank)
        hsi_flat = A_flat @ self.endmember  # (H*W, C) = (H*W, rank) @ (rank, C)
        # Reshape to (1, C, H, W)
        hsi_image = hsi_flat.view(1, self.H, self.W, self.C).permute(0, 3, 1, 2).contiguous()

        return {"abundance": out_abundance, "render": hsi_image}

    def train_iter(self, gt_image, mask, loss_type="L2"):
        """Single training iteration with masked HSI loss.

        Args:
            gt_image: (1, C, H, W) ground truth HSI
            mask: (1, 1, H, W) or (1, C, H, W) binary mask, 1=observed
            loss_type: "L1" or "L2"

        Returns:
            loss, psnr_full
        """
        render_pkg = self.forward()
        image = render_pkg["render"]  # (1, C, H, W)

        # Masked loss in HSI space: ||M ⊙ (X - A×E₀)||_F² / num_elements
        masked_pred = image * mask
        masked_gt = gt_image * mask
        num_observed = mask.expand_as(image).sum().clamp(min=1)
        if loss_type == "L1":
            data_loss = (masked_pred - masked_gt).abs().sum() / num_observed
        else:
            data_loss = ((masked_pred - masked_gt) ** 2).sum() / num_observed

        loss = data_loss
        loss.backward()

        with torch.no_grad():
            mse_full = F.mse_loss(image, gt_image)
            psnr_full = 10 * math.log10(1.0 / mse_full.item()) if mse_full.item() > 0 else 100.0

        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()

        return loss, psnr_full
