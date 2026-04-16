"""
Initialization strategies for Gabor/Gaussian inpainting.

Provides observed-first (obs_first) initialization and helpers.
This module is independent from the model definition and only performs
post-construction parameter re-initialization via `torch.no_grad()`.

Usage:
    model = GaussianImage_Cholesky(...)
    if init_strategy == "obs_first":
        apply_obs_first_init(model, mask, gt_image, obs_init_ratio=0.8)
"""

import torch
import torch.nn.functional as F
import math


def compute_coverage_map(mask):
    """Aggregate a possibly multi-channel mask into a single-channel coverage map.

    For pixel-wise mask (1, 1, H, W) this is a no-op.
    For element-wise mask (1, C, H, W) the per-channel masks are averaged
    to produce a soft coverage score per pixel in [0, 1].

    Args:
        mask: (1, 1, H, W) or (1, C, H, W) float tensor with values in {0, 1}.

    Returns:
        coverage: (H, W) float tensor in [0, 1].
    """
    # Average across batch and channel dims → (H, W)
    coverage = mask.mean(dim=(0, 1))  # (H, W)
    return coverage


def sample_positions_from_weights(weight_map, num_samples, device="cpu"):
    """Sample 2D pixel positions according to a weight map.

    Args:
        weight_map: (H, W) non-negative weight tensor.  Positions with
            higher weight are sampled more frequently.
        num_samples: number of positions to sample.
        device: torch device.

    Returns:
        positions: (num_samples, 2) float tensor of (x_norm, y_norm) in [0, 1],
            where x_norm = col / W, y_norm = row / H.
    """
    H, W = weight_map.shape
    flat_weights = weight_map.reshape(-1).float()

    # Ensure non-negative and at least some weight
    flat_weights = flat_weights.clamp(min=0.0)
    total = flat_weights.sum()
    if total <= 0:
        # Fallback: uniform sampling
        flat_weights = torch.ones_like(flat_weights)
        total = flat_weights.sum()

    # Sample with replacement
    indices = torch.multinomial(flat_weights, num_samples, replacement=True)

    rows = (indices // W).float()
    cols = (indices % W).float()

    # Add sub-pixel jitter to avoid grid-locking
    rows = rows + torch.rand(num_samples, device=device) - 0.5
    cols = cols + torch.rand(num_samples, device=device) - 0.5

    # Normalize to [0, 1]
    y_norm = rows.clamp(0, H - 1) / max(H - 1, 1)
    x_norm = cols.clamp(0, W - 1) / max(W - 1, 1)

    return torch.stack([x_norm, y_norm], dim=1)  # (N, 2)


def positions_norm_to_model(positions_norm):
    """Convert normalized [0,1] positions to the model's atanh parameter space.

    The model stores `_xyz` such that `tanh(_xyz)` is in [-1, 1].
    Pixel mapping: normalized [0, 1] → model space [-1, 1] → atanh.

    Args:
        positions_norm: (N, 2) float tensor with values in [0, 1].

    Returns:
        atanh_positions: (N, 2) float tensor in atanh space.
    """
    # Map [0, 1] → [-1, 1]
    xy = positions_norm * 2.0 - 1.0
    # Clamp to avoid atanh(±1) = ±inf
    xy = xy.clamp(-0.999, 0.999)
    return torch.atanh(xy)


def sample_colors_from_image(positions_norm, image, mask=None):
    """Sample RGB colors from the image at given normalized positions.

    Uses bilinear interpolation via grid_sample.  For positions that land
    on missing pixels (mask ≈ 0), falls back to random color initialization.

    Args:
        positions_norm: (N, 2) float tensor of (x_norm, y_norm) in [0, 1].
        image: (1, C, H, W) image tensor.
        mask: (1, 1, H, W) or (1, C, H, W) mask tensor, optional.

    Returns:
        colors: (N, C) float tensor.
    """
    N = positions_norm.shape[0]
    C = image.shape[1]

    # grid_sample expects grid in (x, y) order with range [-1, 1]
    x_norm = positions_norm[:, 0]
    y_norm = positions_norm[:, 1]
    grid_x = x_norm * 2.0 - 1.0
    grid_y = y_norm * 2.0 - 1.0
    grid = torch.stack([grid_x, grid_y], dim=1).view(1, 1, N, 2)  # (1, 1, N, 2)

    # Sample image values
    sampled = F.grid_sample(
        image, grid, mode="bilinear", padding_mode="border", align_corners=True
    )  # (1, C, 1, N)
    colors = sampled.squeeze(0).squeeze(1).permute(1, 0)  # (N, C)

    if mask is not None:
        # Aggregate mask to single channel for checking coverage
        coverage = mask.mean(dim=1, keepdim=True)  # (1, 1, H, W)
        sampled_cov = F.grid_sample(
            coverage, grid, mode="bilinear", padding_mode="border", align_corners=True
        )  # (1, 1, 1, N)
        cov_vals = sampled_cov.squeeze()  # (N,)

        # For positions with very low coverage, use random init
        low_coverage = cov_vals < 0.1
        if low_coverage.any():
            colors[low_coverage] = torch.rand(
                low_coverage.sum().item(), C, device=colors.device
            )

    return colors


def apply_obs_first_init(model, mask, gt_image, obs_init_ratio=0.8):
    """Apply observed-first initialization to a GaussianImage_Cholesky model.

    Replaces the model's default random position and color initialization
    with a strategy that concentrates most Gaussians on observed pixels
    while placing the rest uniformly in missing regions.

    Args:
        model: GaussianImage_Cholesky instance (already constructed).
        mask: (1, 1, H, W) or (1, C, H, W) observation mask.
        gt_image: (1, C, H, W) ground truth image used for color init.
        obs_init_ratio: fraction of Gaussians initialized at observed positions.
            The remaining (1 - obs_init_ratio) are sampled uniformly
            within the missing (mask==0) region. Default: 0.8.
    """
    device = model._xyz.device
    N = model.init_num_points
    H, W = model.H, model.W

    num_obs = int(round(N * obs_init_ratio))
    num_missing = N - num_obs

    # --- Coverage map ---
    coverage = compute_coverage_map(mask)  # (H, W)

    # --- Sample observed-biased positions ---
    obs_positions = sample_positions_from_weights(
        coverage, num_obs, device=device
    )  # (num_obs, 2)

    # --- Sample uniformly within the missing region ---
    missing_map = (1.0 - coverage).clamp(min=0.0)  # (H, W)
    # If no missing pixels (full observation), fall back to uniform
    if missing_map.sum() <= 0:
        missing_map = torch.ones(H, W, device=device)
    missing_positions = sample_positions_from_weights(
        missing_map, num_missing, device=device
    )  # (num_missing, 2)

    # --- Concatenate and convert to model space ---
    all_positions = torch.cat([obs_positions, missing_positions], dim=0)  # (N, 2)
    atanh_xy = positions_norm_to_model(all_positions)

    observed_image = gt_image * mask
    colors = sample_colors_from_image(all_positions, observed_image, mask=mask)

    # --- Apply to model parameters (no grad) ---
    with torch.no_grad():
        model._xyz.data.copy_(atanh_xy)
        model._features_dc.data.copy_(colors)

    return model
