"""
HSI Low-Rank Gaussian Splatting Inpainting Model.

Renders abundance maps A (H, W, rank) via standard Gaussian splatting
(rasterize_gaussians_sum), then reconstructs HSI via Y_hat = A @ E
where E is the NMF endmember matrix.

Supports arbitrary rank determined by endmember.shape[0]; no hardcoded
feature dimension splits or quantization logic.
"""

from gsplat.project_gaussians_2d import project_gaussians_2d
from gsplat.rasterize_sum import rasterize_gaussians_sum
import torch
import torch.nn as nn
from optimizer import Adan


class GaussianImage_Cholesky_HSI(nn.Module):
    """Gaussian splatting model for HSI inpainting via low-rank decomposition.

    Core pipeline:
        1. Project 2D Gaussians: position (tanh-mapped) + Cholesky covariance
        2. Rasterize abundance maps: (H, W, rank) via rasterize_gaussians_sum
        3. Reconstruct HSI: Y_hat = A @ E  where A is (H*W, rank), E is (rank, C)

    Supports arbitrary rank. No quantization.
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
        self._xyz = nn.Parameter(
            torch.atanh(2 * (torch.rand(self.init_num_points, 2) - 0.5))
        )
        self._cholesky = nn.Parameter(torch.rand(self.init_num_points, 3))
        self.register_buffer('_opacity', torch.ones((self.init_num_points, 1)))
        # Abundance features: (N, rank) — arbitrary rank, no splits
        self._features_dc = nn.Parameter(
            0.5 * torch.rand(self.init_num_points, self.rank)
        )

        self.last_size = (self.H, self.W)
        self.register_buffer('background', torch.ones(self.rank))
        self.register_buffer('bound', torch.tensor([0.5, 0.5]).view(1, 2))
        self.register_buffer('cholesky_bound',
                             torch.tensor([0.5, 0, 0.5]).view(1, 3))

        # Optimizer
        self.lr = kwargs["lr"]
        if kwargs.get("opt_type", "adan") == "adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        else:
            self.optimizer = Adan(self.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=20000, gamma=0.5
        )

    # ------------------------------------------------------------------
    # Properties (same interface as the Gabor model for interoperability)
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self):
        """Render abundance maps via Gaussian splatting, then reconstruct HSI.

        Returns:
            dict with:
                "abundance": (H, W, rank) abundance map (non-negative)
                "render":    (1, C, H, W) HSI reconstruction  Y_hat = A @ E
        """
        self.xys, depths, self.radii, conics, num_tiles_hit = \
            project_gaussians_2d(
                self.get_xyz, self.get_cholesky_elements,
                self.H, self.W, self.tile_bounds,
            )

        # Gaussian splatting — render all rank channels at once
        out_abundance = rasterize_gaussians_sum(
            self.xys, depths, self.radii, conics, num_tiles_hit,
            self.get_features,          # (N, rank)
            self._opacity,              # (N, 1)
            self.H, self.W,
            self.BLOCK_H, self.BLOCK_W,
            background=self.background,  # (rank,)
            return_alpha=False,
        )  # -> (H, W, rank)

        out_abundance = torch.clamp(out_abundance, min=0)  # non-negative

        # HSI reconstruction: A @ E -> (H*W, C)
        A_flat = out_abundance.view(self.H * self.W, self.rank)
        hsi_flat = A_flat @ self.endmember                    # (H*W, C)
        hsi_image = (hsi_flat
                     .view(1, self.H, self.W, self.C)
                     .permute(0, 3, 1, 2)
                     .contiguous())                            # (1, C, H, W)

        return {"abundance": out_abundance, "render": hsi_image}
