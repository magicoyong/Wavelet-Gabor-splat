"""
Utility functions for Gabor/Gaussian-based image inpainting.

Provides mask generation, masked fidelity losses, regularization terms,
and visualization helpers. Does NOT modify the original utils.py.
"""

import math
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from pytorch_msssim import ms_ssim


# ---------------------------------------------------------------------------
# Mask generation
# ---------------------------------------------------------------------------

def generate_random_mask(H, W, mask_ratio=0.5, device="cpu"):
    """Generate a random pixel-wise binary mask.

    Args:
        H, W: image height and width.
        mask_ratio: fraction of pixels to *drop* (set to 0 in mask).
        device: torch device.

    Returns:
        mask: (1, 1, H, W) float tensor. 1 = observed, 0 = missing.
    """
    mask = (torch.rand(1, 1, H, W, device=device) >= mask_ratio).float()
    return mask


def generate_block_mask(H, W, block_size=64, num_blocks=4, device="cpu"):
    """Generate a block-occlusion binary mask.

    Places ``num_blocks`` square blocks of size ``block_size`` at random
    positions.  Blocked regions are 0 (missing), everything else is 1.

    Args:
        H, W: image height and width.
        block_size: side length of each square block.
        num_blocks: how many blocks to drop.
        device: torch device.

    Returns:
        mask: (1, 1, H, W) float tensor. 1 = observed, 0 = missing.
    """
    mask = torch.ones(1, 1, H, W, device=device)
    for _ in range(num_blocks):
        top = np.random.randint(0, max(H - block_size, 1))
        left = np.random.randint(0, max(W - block_size, 1))
        mask[:, :, top:top + block_size, left:left + block_size] = 0.0
    return mask


def generate_mask(H, W, mask_type="random", mask_ratio=0.5,
                  block_size=64, num_blocks=4, device="cpu"):
    """Unified mask generator.

    Args:
        mask_type: ``"random"`` for pixel-wise, ``"block"`` for block occlusion.
        Other args forwarded to the corresponding generator.

    Returns:
        mask: (1, 1, H, W) float tensor.
    """
    if mask_type == "random":
        return generate_random_mask(H, W, mask_ratio=mask_ratio, device=device)
    elif mask_type == "block":
        return generate_block_mask(H, W, block_size=block_size,
                                   num_blocks=num_blocks, device=device)
    else:
        raise ValueError(f"Unknown mask_type: {mask_type}")


# ---------------------------------------------------------------------------
# Masked fidelity loss
# ---------------------------------------------------------------------------

def masked_mse_loss(pred, target, mask):
    """MSE loss computed only on observed (mask==1) pixels.

    All inputs should be broadcastable: pred & target (1, C, H, W),
    mask (1, 1, H, W).
    """
    diff = (pred - target) ** 2
    # mask broadcasts over channel dim
    masked_diff = diff * mask
    # average over observed pixels only
    num_observed = mask.sum().clamp(min=1.0)
    return masked_diff.sum() / (num_observed * pred.shape[1])


def masked_l1_loss(pred, target, mask):
    """L1 loss computed only on observed pixels."""
    diff = (pred - target).abs()
    masked_diff = diff * mask
    num_observed = mask.sum().clamp(min=1.0)
    return masked_diff.sum() / (num_observed * pred.shape[1])


def masked_loss_fn(pred, target, mask, loss_type="L2"):
    """Masked fidelity loss matching the style of utils.loss_fn.

    Args:
        pred: (1, C, H, W)
        target: (1, C, H, W)
        mask: (1, 1, H, W)
        loss_type: ``"L2"`` or ``"L1"``.
    """
    if loss_type == "L2":
        return masked_mse_loss(pred, target, mask)
    elif loss_type == "L1":
        return masked_l1_loss(pred, target, mask)
    else:
        raise ValueError(f"Unsupported masked loss_type: {loss_type}")


# ---------------------------------------------------------------------------
# Regularization
# ---------------------------------------------------------------------------

def gabor_weights_l1_reg(model):
    """L1 sparsity regularization on gabor_weights (after sigmoid)."""
    return model.get_gabor_weights.abs().mean()


def position_l2_reg(model):
    """L2 regularization on Gaussian positions to prevent drift."""
    return (model.get_xyz ** 2).mean()


def cholesky_l2_reg(model):
    """L2 regularization on Cholesky parameters for covariance stability."""
    return (model._cholesky ** 2).mean()


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def compute_psnr(pred, target):
    """Compute PSNR between two (1, C, H, W) tensors."""
    mse = F.mse_loss(pred.float(), target.float())
    if mse.item() == 0:
        return float('inf')
    return 10 * math.log10(1.0 / mse.item())


def compute_ms_ssim(pred, target):
    """Compute MS-SSIM between two (1, C, H, W) tensors."""
    return ms_ssim(pred.float(), target.float(), data_range=1, size_average=True).item()


# ---------------------------------------------------------------------------
# Visualization / saving helpers
# ---------------------------------------------------------------------------

def save_tensor_as_image(tensor, path):
    """Save a (1, C, H, W) or (C, H, W) tensor as PNG image.

    Values are clamped to [0, 1] before conversion.
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    tensor = tensor.clamp(0, 1)
    img = transforms.ToPILImage()(tensor.cpu())
    img.save(str(path))


def compute_error_map(pred, target):
    """Per-pixel absolute error, returned as (1, C, H, W)."""
    return (pred - target).abs()


def mask_to_image_tensor(mask):
    """Convert (1, 1, H, W) mask to (1, 3, H, W) for saving."""
    return mask.expand(-1, 3, -1, -1)
