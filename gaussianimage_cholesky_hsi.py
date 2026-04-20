"""
HSI Low-Rank Gaussian Inpainting Model.

Renders abundance maps A (H, W, rank) via multi-channel Gaussian splatting,
then reconstructs HSI via Y_hat = A @ E_hat where:
    E_hat = E0 + gamma * (U @ V)
E0 is the masked-NMF initial endmember (frozen buffer).
U (rank, calib_rank) and V (calib_rank, C) are trainable low-rank
correction parameters.  gamma is a learnable scalar bounded by
max_calib_scale:  gamma = max_calib_scale * tanh(calib_gamma).
"""

from gsplat.project_gaussians_2d import project_gaussians_2d
from gsplat.rasterize_sum import rasterize_gaussians_sum_nd
from utils import loss_fn
import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from optimizer import Adan


class GaussianImage_Cholesky_HSI(nn.Module):
    """Gaussian splatting model for HSI inpainting via low-rank decomposition.

    Parameters are Gaussian positions, Cholesky covariance, per-Gaussian
    abundance features (rank channels).
    The endmember matrix E_hat = E0 + gamma * (U @ V) maps abundance to
    HSI channels.
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

        # ── Endmember calibration ───────────────────────────────────────
        # E0: frozen masked-NMF initial endmember (rank, C)
        E_np = kwargs["E"]  # numpy array (rank, C)
        self.register_buffer('E0', torch.tensor(E_np, dtype=torch.float32))

        # Low-rank correction: E_hat = E0 + gamma * tanh(U @ V)
        self.calib_rank = kwargs.get("calib_rank", 2)
        self.max_calib_scale = kwargs.get("gamma", 0.1)  # max |gamma|
        self.freeze_endmember_calibration = kwargs.get(
            "freeze_endmember_calibration", False
        )

        # U: (rank, calib_rank),  V: (calib_rank, C)
        # Non-zero init (like refine code) so gradients can flow.
        self.calib_U = nn.Parameter(
            torch.empty(self.rank, self.calib_rank, dtype=torch.float32)
        )
        self.calib_V = nn.Parameter(
            torch.empty(self.calib_rank, self.C, dtype=torch.float32)
        )
        # Learnable gamma: starts at 0 so E_hat ≈ E0 initially,
        # but U/V have non-zero init so gradients are non-zero.
        self.calib_gamma = nn.Parameter(torch.tensor(0.0))
        nn.init.kaiming_uniform_(self.calib_U, a=np.sqrt(5))
        nn.init.kaiming_uniform_(self.calib_V, a=np.sqrt(5))
        self.calib_V.data.mul_(1e-3)  # keep V small at init

        # ── Learnable Gaussian parameters ───────────────────────────────
        self._xyz = nn.Parameter(torch.atanh(2 * (torch.rand(self.init_num_points, 2) - 0.5)))
        self._cholesky = nn.Parameter(torch.rand(self.init_num_points, 3))
        self.register_buffer('_opacity', torch.ones((self.init_num_points, 1)))
        # Abundance features: (N, rank)
        self._features_dc = nn.Parameter(torch.rand(self.init_num_points, self.rank))

        self.last_size = (self.H, self.W)
        self.register_buffer('background', torch.ones(self.rank))
        self.register_buffer('bound', torch.tensor([0.5, 0.5]).view(1, 2))
        self.register_buffer('cholesky_bound', torch.tensor([0.5, 0, 0.5]).view(1, 3))

        # ── Optimizer ───────────────────────────────────────────────────
        self.lr = kwargs["lr"]
        if kwargs.get("opt_type", "adan") == "adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        else:
            self.optimizer = Adan(self.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=20000, gamma=0.5
        )

    # ── Endmember interface ─────────────────────────────────────────────

    def get_calibrated_endmember(self):
        """Return E_hat = clamp(E0 + gamma * U @ V, min=eps).

        If *freeze_endmember_calibration* is True, returns E0 directly
        (still clamped for safety).

        Returns:
            Tensor of shape (rank, C).
        """
        if self.freeze_endmember_calibration:
            return torch.clamp(self.E0, min=1e-6)
        delta_E = torch.tanh(self.calib_U @ self.calib_V)  # (rank, C), bounded [-1,1]
        gamma = self.max_calib_scale * torch.tanh(self.calib_gamma)  # learnable, bounded
        E_hat = self.E0 + gamma * delta_E
        return torch.clamp(E_hat, min=1e-6)

    def get_delta_E_norm(self):
        """Return Frobenius norm of the effective correction gamma * tanh(U @ V)."""
        with torch.no_grad():
            delta = torch.tanh(self.calib_U @ self.calib_V)
            gamma = self.max_calib_scale * torch.tanh(self.calib_gamma)
            return (gamma * delta).norm().item()

    # ── Properties ──────────────────────────────────────────────────────

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

    # ── Forward ─────────────────────────────────────────────────────────

    def forward(self):
        """Render abundance maps via Gaussian splatting, reconstruct HSI.

        Returns:
            dict with:
                "abundance": (H, W, rank) abundance map (non-negative)
                "render": (1, C, H, W) HSI reconstruction Y_hat = A_hat @ E_hat
                "E_hat": (rank, C) calibrated endmember used for this forward
        """
        self.xys, depths, self.radii, conics, num_tiles_hit = project_gaussians_2d(
            self.get_xyz, self.get_cholesky_elements, self.H, self.W, self.tile_bounds
        )
        # Multi-channel Gaussian rendering → abundance: (H, W, rank)
        out_abundance = rasterize_gaussians_sum_nd(
            self.xys, depths, self.radii, conics, num_tiles_hit,
            self.get_features, self._opacity,
            self.H, self.W, self.BLOCK_H, self.BLOCK_W,
            background=self.background, return_alpha=False,
        )
        out_abundance = torch.clamp(out_abundance, min=0)  # (H, W, rank), non-negative

        # Calibrated endmember
        E_hat = self.get_calibrated_endmember()  # (rank, C)

        # HSI reconstruction: A_hat @ E_hat → (H*W, C)
        A_flat = out_abundance.view(self.H * self.W, self.rank)  # (H*W, rank)
        hsi_flat = A_flat @ E_hat  # (H*W, C)
        hsi_image = hsi_flat.view(1, self.H, self.W, self.C).permute(0, 3, 1, 2).contiguous()

        return {"abundance": out_abundance, "render": hsi_image, "E_hat": E_hat}
