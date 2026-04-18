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


def generate_elementwise_mask(H, W, C=3, mask_ratio=0.5, device="cpu"):
    """Generate an element-wise random binary mask (per-channel independent).

    Each element (pixel, channel) is independently sampled. This matches
    the "random missing" pattern used in GSLR (arXiv:2511.14270) where
    the mask tensor M has the same shape as the image.

    Args:
        H, W: image height and width.
        C: number of channels.
        mask_ratio: fraction of elements to *drop* (set to 0 in mask).
                    Sample rate SR = 1 - mask_ratio.
        device: torch device.

    Returns:
        mask: (1, C, H, W) float tensor. 1 = observed, 0 = missing.
    """
    mask = (torch.rand(1, C, H, W, device=device) >= mask_ratio).float()
    return mask


def generate_mask(H, W, mask_type="elementwise", mask_ratio=0.5,
                  block_size=64, num_blocks=4, C=3, device="cpu"):
    """Unified mask generator.

    Args:
        mask_type: ``"random"`` for pixel-wise, ``"elementwise"`` for
            per-channel independent, ``"block"`` for block occlusion.
        C: number of channels (only used for ``"elementwise"``).
        Other args forwarded to the corresponding generator.

    Returns:
        mask: (1, 1, H, W) or (1, C, H, W) float tensor.
    """
    if mask_type == "random":
        return generate_random_mask(H, W, mask_ratio=mask_ratio, device=device)
    elif mask_type == "elementwise":
        return generate_elementwise_mask(H, W, C=C, mask_ratio=mask_ratio, device=device)
    elif mask_type == "block":
        return generate_block_mask(H, W, block_size=block_size,
                                   num_blocks=num_blocks, device=device)
    else:
        raise ValueError(f"Unknown mask_type: {mask_type}")


# ---------------------------------------------------------------------------
# Masked fidelity loss
# ---------------------------------------------------------------------------

def masked_mse_loss(pred, target, mask):
    """MSE loss computed only on observed (mask==1) elements.

    Handles both pixel-wise mask (1, 1, H, W) and element-wise
    mask (1, C, H, W) correctly by counting actual observed elements.
    """
    diff = (pred - target) ** 2
    masked_diff = diff * mask
    num_observed = mask.expand_as(pred).sum().clamp(min=1.0)
    return masked_diff.sum() / num_observed


def masked_l1_loss(pred, target, mask):
    """L1 loss computed only on observed elements."""
    diff = (pred - target).abs()
    masked_diff = diff * mask
    num_observed = mask.expand_as(pred).sum().clamp(min=1.0)
    return masked_diff.sum() / num_observed


def masked_loss_fn(pred, target, mask, loss_type="L2"):
    """Masked fidelity loss matching the style of utils.loss_fn.

    Args:
        pred: (1, C, H, W)
        target: (1, C, H, W)
        mask: (1, 1, H, W) or (1, C, H, W)
        loss_type: ``"L2"`` or ``"L1"``.
    """
    if loss_type == "L2":
        return masked_mse_loss(pred, target, mask)
    elif loss_type == "L1":
        return masked_l1_loss(pred, target, mask)
    else:
        raise ValueError(f"Unsupported masked loss_type: {loss_type}")


def get_missing_mask(mask):
    """Return the missing-region mask as the complement of the observed mask.

    Args:
        mask: (1, 1, H, W) or (1, C, H, W) float tensor with values in {0, 1}.

    Returns:
        missing_mask: same shape as ``mask`` where 1 indicates missing entries.
    """
    return 1.0 - mask


def compute_coverage_map(mask):
    """Aggregate a mask into a single-channel per-pixel coverage map.

    For pixel-wise masks, this returns the mask itself.
    For element-wise masks, this averages over channels so values in [0, 1]
    indicate how many channels are observed at each pixel.

    Args:
        mask: (1, 1, H, W) or (1, C, H, W).

    Returns:
        coverage_map: (1, 1, H, W) tensor.
    """
    return mask.mean(dim=1, keepdim=True)


def compute_masked_psnr(pred, target, mask):
    """Compute PSNR on any masked subset.

    The mask is applied element-wise. For a pixel-wise mask with shape
    ``(1, 1, H, W)``, the mask is broadcast to all channels. For an
    element-wise mask with shape ``(1, C, H, W)``, each channel is scored
    against its own observed/missing entries.
    """
    expanded_mask = mask.expand_as(pred)
    num_selected = expanded_mask.sum().clamp(min=1.0)
    mse = ((pred.float() - target.float()) * expanded_mask).pow(2).sum() / num_selected
    if mse.item() == 0:
        return float("inf")
    return 10 * math.log10(1.0 / mse.item())


def compute_inpainting_psnrs(pred, target, observed_mask):
    """Compute full / observed-only / missing-only PSNR for inpainting.

    Missing-only is defined strictly as the complement of the original
    observed mask, i.e. ``1 - observed_mask``.

    Returns:
        dict with keys ``psnr_full``, ``psnr_observed``, ``psnr_missing``.
    """
    missing_mask = get_missing_mask(observed_mask)
    return {
        "psnr_full": compute_psnr(pred, target),
        "psnr_observed": compute_masked_psnr(pred, target, observed_mask),
        "psnr_missing": compute_masked_psnr(pred, target, missing_mask),
    }


# ---------------------------------------------------------------------------
# Regularization
# ---------------------------------------------------------------------------

def gabor_weights_l2_reg(model):
    """L2 sparsity regularization on gabor_weights (after sigmoid)."""
    return model.get_gabor_weights.abs().mean()


def gabor_freq_l1_reg(model):
    """L1 regularization on gabor frequency for stable frequecy fitting."""
    return model.get_gabor_freqs.abs().mean()


def cholesky_l2_reg(model):
    """L2 regularization on Cholesky parameters for covariance stability."""
    return (model._cholesky ** 2).mean()


def tv_loss(image):
    """Anisotropic total variation loss on a rendered image.

    Args:
        image: (1, C, H, W) tensor.

    Returns:
        Scalar TV penalty normalized by the number of finite differences.
    """
    if image.shape[-2] < 2 and image.shape[-1] < 2:
        return torch.tensor(0.0, device=image.device, dtype=image.dtype)

    loss = torch.tensor(0.0, device=image.device, dtype=image.dtype)
    num_terms = 0
    if image.shape[-2] >= 2:
        loss = loss + (image[:, :, 1:, :] - image[:, :, :-1, :]).abs().mean()
        num_terms += 1
    if image.shape[-1] >= 2:
        loss = loss + (image[:, :, :, 1:] - image[:, :, :, :-1]).abs().mean()
        num_terms += 1
    return loss / max(num_terms, 1)


def nuclear_norm_reg(image, downsample=4):
    """Nuclear norm regularization on rendered image to promote low-rank.

    Computes  sum_{c=0}^{C-1} ||A_{[c]}||_*  where A_{[c]} is the c-th
    channel (H x W matrix) and ||X||_* = sum of singular values of X.

    The image is downsampled before SVD to keep the cost manageable.
    SVD on a full-res image (e.g. 768x512) is very expensive; downsampling
    by 4x reduces SVD cost by ~64x while preserving the low-rank prior.

    Args:
        image: (1, C, H, W) rendered image tensor (must be in autograd graph).
        downsample: spatial downsample factor before SVD. 1 = full resolution.

    Returns:
        Scalar nuclear norm summed over channels (normalized by matrix size).
    """
    if downsample > 1:
        image = F.avg_pool2d(image, kernel_size=downsample, stride=downsample)
    img = image.squeeze(0)  # (C, H, W)
    reg = torch.tensor(0.0, device=image.device)
    for c in range(img.shape[0]):
        sigmas = torch.linalg.svdvals(img[c])  # (min(h,w),)
        reg = reg + sigmas.sum()
    # Normalize by the downsampled matrix size so lambda doesn't need re-tuning
    h, w = img.shape[1], img.shape[2]
    reg = reg / (h * w)
    return reg


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


def compute_ssim_hsi(pred, target, window_size=11):
    """Compute mean SSIM over all spectral bands for HSI data.

    Args:
        pred: (1, C, H, W) tensor
        target: (1, C, H, W) tensor
        window_size: Gaussian window size for SSIM

    Returns:
        float: mean SSIM across all C bands
    """
    C = pred.shape[1]
    # Create 1D Gaussian kernel
    sigma = 1.5
    coords = torch.arange(window_size, dtype=torch.float32, device=pred.device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    # 2D window
    window = g.unsqueeze(1) * g.unsqueeze(0)  # (ws, ws)
    window = window.unsqueeze(0).unsqueeze(0)  # (1, 1, ws, ws)

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    pad = window_size // 2

    ssim_bands = []
    for c in range(C):
        p = pred[:, c:c+1, :, :].float()  # (1, 1, H, W)
        t = target[:, c:c+1, :, :].float()

        mu_p = F.conv2d(p, window, padding=pad)
        mu_t = F.conv2d(t, window, padding=pad)
        mu_p_sq = mu_p ** 2
        mu_t_sq = mu_t ** 2
        mu_pt = mu_p * mu_t

        sigma_p_sq = F.conv2d(p * p, window, padding=pad) - mu_p_sq
        sigma_t_sq = F.conv2d(t * t, window, padding=pad) - mu_t_sq
        sigma_pt = F.conv2d(p * t, window, padding=pad) - mu_pt

        num = (2 * mu_pt + C1) * (2 * sigma_pt + C2)
        den = (mu_p_sq + mu_t_sq + C1) * (sigma_p_sq + sigma_t_sq + C2)
        ssim_map = num / den
        ssim_bands.append(ssim_map.mean().item())

    return np.mean(ssim_bands)


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


def compute_region_error_map(pred, target, mask):
    """Absolute error restricted to a given region mask."""
    return compute_error_map(pred, target) * mask.expand_as(pred)


def mask_to_image_tensor(mask):
    """Convert mask to (1, 3, H, W) for saving.

    For pixel-wise mask (1, 1, H, W): expands to 3 channels.
    For element-wise mask (1, C, H, W): returns as-is (already has channels).
    """
    if mask.shape[1] == 1:
        return mask.expand(-1, 3, -1, -1)
    return mask


def coverage_map_to_image_tensor(mask):
    """Convert a coverage map derived from the mask into a 3-channel image."""
    coverage = compute_coverage_map(mask)
    return coverage.expand(-1, 3, -1, -1)
