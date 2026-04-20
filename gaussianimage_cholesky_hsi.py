"""
HSI Low-Rank Gaussian Splatting Inpainting Model.

Renders abundance maps A (H, W, rank) via standard Gaussian splatting
(rasterize_gaussians_sum), then reconstructs HSI via Y_hat = A @ E
where E is the NMF endmember matrix.

IMPORTANT: The CUDA kernels rasterize_sum_forward / nd_rasterize_sum_forward
only support exactly 3 (float3) or 4 (float4) channels. Features must be
split into chunks of 3 or 4 before rasterization.
"""

from gsplat.project_gaussians_2d import project_gaussians_2d
from gsplat.rasterize_sum import rasterize_gaussians_sum
import torch
import torch.nn as nn
from optimizer import Adan


def _safe_chunk_sizes(rank):
    """Return a tuple of chunk sizes (each 3 or 4) that sum to rank.

    The rasterize_gaussians_sum CUDA kernel only supports 3 (float3) or
    4 (float4) channels.  This function computes a valid partition.
    """
    if rank <= 0:
        return ()
    if rank in (1, 2):
        return (3,)  # must pad features to 3
    # 3a + 4b = rank, a >= 0, b >= 0
    b, r = divmod(rank, 4)
    if r == 0:
        return (4,) * b
    elif r == 3:
        return (4,) * b + (3,)
    elif r == 2:
        # 4 + 2 -> 3 + 3
        return (4,) * (b - 1) + (3, 3) if b >= 1 else (3,) * (rank // 3) + ((rank % 3,) if rank % 3 else ())
    else:  # r == 1
        if b >= 2:
            # 4 + 4 + 1 -> 3 + 3 + 3
            return (4,) * (b - 2) + (3, 3, 3)
        else:
            # rank = 5 (b=1,r=1): pad to 6 = 3+3 (caller must pad features)
            # rank = 1 (b=0,r=1): pad to 3 (caller must pad features)
            padded = rank + (3 - rank % 3) % 3
            return _safe_chunk_sizes(padded)


class GaussianImage_Cholesky_HSI(nn.Module):
    """Gaussian splatting model for HSI inpainting via low-rank decomposition.

    Core pipeline:
        1. Project 2D Gaussians: position (tanh-mapped) + Cholesky covariance
        2. Rasterize abundance maps in 3/4-channel chunks
        3. Reconstruct HSI: Y_hat = A @ E  where A is (H*W, rank), E is (rank, C)
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

        # Pre-compute safe chunk sizes for feature splitting
        self._chunk_sizes = _safe_chunk_sizes(self.rank)
        self._padded_rank = sum(self._chunk_sizes)  # may exceed rank if padding needed

        # Endmember matrix E: (rank, C), frozen (not optimized)
        E_np = kwargs["E"]  # numpy array (rank, C)
        self.register_buffer('endmember', torch.tensor(E_np, dtype=torch.float32))

        # Learnable Gaussian parameters
        self._xyz = nn.Parameter(
            torch.atanh(2 * (torch.rand(self.init_num_points, 2) - 0.5))
        )
        self._cholesky = nn.Parameter(torch.zeros(self.init_num_points, 3))
        self.register_buffer('_opacity', torch.ones((self.init_num_points, 1)))
        # Abundance features: (N, rank)
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
            self.optimizer, step_size=3000, gamma=0.5
        )

    # ------------------------------------------------------------------
    # Properties
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

        Features are split into chunks of 3 or 4 for the CUDA kernel,
        then concatenated.

        Returns:
            dict with:
                "abundance": (H, W, rank) abundance map (clamped to [0, 1])
                "render":    (1, C, H, W) HSI reconstruction  Y_hat = A @ E
        """
        self.xys, depths, self.radii, conics, num_tiles_hit = \
            project_gaussians_2d(
                self.get_xyz, self.get_cholesky_elements,
                self.H, self.W, self.tile_bounds,
            )

        # Pad features if needed (for ranks like 5 that need padding)
        features = self.get_features
        if self._padded_rank > self.rank:
            pad = torch.zeros(
                features.shape[0], self._padded_rank - self.rank,
                device=features.device, dtype=features.dtype,
            )
            features = torch.cat([features, pad], dim=1)

        # Split features into safe chunks (each 3 or 4)
        features_split = torch.split(features, list(self._chunk_sizes), dim=1)

        # Rasterize each chunk separately
        out_chunks = []
        for chunk in features_split:
            out = rasterize_gaussians_sum(
                self.xys, depths, self.radii, conics, num_tiles_hit,
                chunk,
                self._opacity,
                self.H, self.W,
                self.BLOCK_H, self.BLOCK_W,
                return_alpha=False,
            )
            out_chunks.append(out)

        # Concatenate along channel dimension: (H, W, padded_rank)
        out_abundance = torch.cat(out_chunks, dim=2)

        # Remove padding channels if any
        if self._padded_rank > self.rank:
            out_abundance = out_abundance[:, :, :self.rank]

        out_abundance = torch.clamp(out_abundance, 0, 1)  # [0, 1]

        # HSI reconstruction: A @ E -> (H*W, C)
        A_flat = out_abundance.view(self.H * self.W, self.rank)
        hsi_flat = A_flat @ self.endmember                    # (H*W, C)
        hsi_image = (hsi_flat
                     .view(1, self.H, self.W, self.C)
                     .permute(0, 3, 1, 2)
                     .contiguous())                            # (1, C, H, W)

        return {"abundance": out_abundance, "render": hsi_image}
