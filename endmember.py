"""
Endmember initialization via NMF.

Provides two entry points:
  1. masked_nmf_initialization — uses only masked (observed) HSI pixels.
     Missing positions are filled differently depending on mask type:
       - elementwise: per-pixel spectral mean (observed bands → missing bands);
         fully-missing pixels fall back to local spatial same-band mean.
       - random / pixel-wise: local spatial same-band mean from neighbouring
         observed pixels (expanding window); global per-band mean only as
         last-resort fallback.
     This is the ONLY path that should be used for HSI inpainting.
  2. (legacy) nmf_initialization — uses full GT HSI.  DEPRECATED for
     inpainting because it leaks test information.
"""

import argparse
import numpy as np
import scipy.io
from sklearn.decomposition import NMF
from scipy.ndimage import uniform_filter
import time
import os


# ── canonical name map shared by all functions ──────────────────────────
_NAME_MAP = {
    "urban": "Urban",
    "salinas": "Salinas",
    "jasperridge": "JR",
    "paviau": "PaviaU",
}


def _canonical(dataset_name: str) -> str:
    return _NAME_MAP.get(dataset_name.lower(), dataset_name)


# ─────────────────────────────────────────────────────────────────────────
# Fill helpers for masked NMF pre-processing
# ─────────────────────────────────────────────────────────────────────────

def _local_spatial_band_fill(band, obs_mask, missing_mask,
                             window_sizes=(5, 11, 21, 41, 81, 161)):
    """Fill missing positions in a single 2D band via expanding-window local mean.

    For each missing pixel, computes the mean of observed values **in the same
    band** within a local spatial window.  Window size is increased progressively
    until all missing pixels are covered or the list is exhausted.

    Only truly observed pixels (``obs_mask``) contribute to the local mean —
    previously filled values are never used as sources.

    Falls back to the global observed-band mean only in extreme cases where no
    observed pixel exists within the largest window.

    Args:
        band:         (H, W) float64 array — one spectral band.
        obs_mask:     (H, W) boolean-like — True/1 at observed positions.
        missing_mask: (H, W) boolean array — True at positions to fill.
        window_sizes: Iterable of odd ints, tried in order.

    Returns:
        filled: (H, W) float64 array with missing positions filled.
    """
    filled = band.copy()
    still_missing = missing_mask.copy()
    obs_float = (np.asarray(obs_mask) > 0.5).astype(np.float64)
    obs_values = band * obs_float  # zero out unobserved positions

    for ws in window_sizes:
        if not still_missing.any():
            break
        local_sum = uniform_filter(obs_values, size=ws,
                                   mode='constant', cval=0.0)
        local_count = uniform_filter(obs_float, size=ws,
                                     mode='constant', cval=0.0)
        can_fill = still_missing & (local_count > 1e-10)
        if can_fill.any():
            filled[can_fill] = local_sum[can_fill] / local_count[can_fill]
            still_missing = still_missing & ~can_fill

    # Last-resort fallback: global observed-band mean
    if still_missing.any():
        obs = band[obs_float > 0.5]
        global_mean = obs.mean() if obs.size > 0 else 0.0
        filled[still_missing] = global_mean

    return filled


def _fill_elementwise_spectral(hsi_hwc, mask_hwc):
    """Fill missing entries for an **elementwise** mask.

    Primary strategy (spectral):
        For each pixel, compute the mean of its *observed* bands and use
        that value to fill its *missing* bands.  This is a purely spectral
        operation — no cross-pixel information is used.

    Fallback (spatial, same-band):
        Pixels where *all* bands are missing cannot use spectral fill.
        For these, each band is filled independently using the expanding-
        window local spatial mean of observed values in the same band
        (``_local_spatial_band_fill``).

    Args:
        hsi_hwc:  (H, W, C) float64 — GT HSI (only observed entries used).
        mask_hwc: (H, W, C) float64 — binary mask (1 = observed).

    Returns:
        filled: (H, W, C) float64 with all missing entries imputed.
    """
    H, W, C = hsi_hwc.shape
    filled = hsi_hwc.copy()

    obs_bool = mask_hwc > 0.5                          # (H, W, C) bool
    obs_count = obs_bool.sum(axis=2)                    # (H, W)
    obs_sum = (hsi_hwc * obs_bool).sum(axis=2)          # (H, W)

    has_some = obs_count > 0                            # pixels with >=1 obs band
    pixel_mean = np.zeros((H, W), dtype=np.float64)
    pixel_mean[has_some] = obs_sum[has_some] / obs_count[has_some]

    # Spectral fill: missing bands <- pixel's observed-band mean
    for c in range(C):
        fill_here = (~obs_bool[:, :, c]) & has_some
        filled[:, :, c] = np.where(fill_here, pixel_mean, filled[:, :, c])

    # Spatial fallback for fully-missing pixels
    all_missing = ~has_some                             # (H, W)
    if all_missing.any():
        for c in range(C):
            band_filled = _local_spatial_band_fill(
                hsi_hwc[:, :, c],
                obs_bool[:, :, c],
                all_missing,
            )
            filled[:, :, c] = np.where(all_missing, band_filled,
                                        filled[:, :, c])

    return filled


def _fill_pixelwise_local_spatial(hsi_hwc, mask_hwc):
    """Fill missing entries for a **pixel-wise** (random / block) mask.

    Because the mask is shared across all bands, a missing pixel has its
    *entire* spectrum absent — spectral self-fill is impossible.

    Strategy (spatial, band-by-band):
        For every band *c* independently, fill each missing pixel with the
        expanding-window local mean of observed pixels in **the same band**
        (``_local_spatial_band_fill``).  This keeps band-to-band alignment
        and avoids cross-band mixing.

    Args:
        hsi_hwc:  (H, W, C) float64 — GT HSI (only observed entries used).
        mask_hwc: (H, W, C) float64 — binary mask (1 = observed, same for
                  all bands at each pixel).

    Returns:
        filled: (H, W, C) float64 with all missing entries imputed.
    """
    H, W, C = hsi_hwc.shape
    filled = hsi_hwc.copy()

    pixel_obs = mask_hwc[:, :, 0] > 0.5                # (H, W) bool
    missing = ~pixel_obs

    if not missing.any():
        return filled

    for c in range(C):
        filled[:, :, c] = _local_spatial_band_fill(
            hsi_hwc[:, :, c], pixel_obs, missing,
        )

    return filled


# ─────────────────────────────────────────────────────────────────────────
# Masked NMF initialisation  (NO test leakage)
# ─────────────────────────────────────────────────────────────────────────

def masked_nmf_initialization(hsi_hwc: np.ndarray,
                              mask: np.ndarray,
                              rank: int,
                              mask_type: str = "auto",
                              max_iter: int = 12000,
                              random_state: int = 42):
    """NMF on masked HSI with mask-type-aware missing-value imputation.

    Imputation strategy depends on ``mask_type``:

      - **elementwise**: per-pixel spectral mean fill (observed bands of the
        same pixel -> missing bands).  Fully-missing pixels fall back to local
        spatial same-band fill.
      - **random** / **block** / pixel-wise: local spatial same-band fill
        (expanding-window mean of observed neighbours in each band
        independently).  Global per-band mean is used only as a last resort.
      - **auto** (default): auto-detect by checking whether the mask varies
        across channels.

    Args:
        hsi_hwc:  (H, W, C) float numpy array, the **full** GT HSI.
                  Only the observed entries (mask==1) will be used.
        mask:     Binary mask broadcastable to (H, W, C).
                  Accepted shapes: (H, W), (H, W, 1), (1, 1, H, W),
                  (1, C, H, W), (H, W, C).  Will be reshaped internally.
        rank:     Number of endmembers.
        mask_type: ``"elementwise"`` | ``"random"`` | ``"block"`` | ``"auto"``.
        max_iter: Maximum NMF iterations.
        random_state: Random seed for NMF.

    Returns:
        endmember: (rank, C) numpy array  --  E0.
        abundance: (H*W, rank) numpy array -- A0.
    """
    H, W, C = hsi_hwc.shape

    # ── normalise mask to (H, W, C) boolean ─────────────────────────────
    m = np.asarray(mask, dtype=np.float64)
    if m.ndim == 4:
        # (1, 1, H, W) or (1, C, H, W) → squeeze batch dim
        m = m.squeeze(0)
        if m.shape[0] == 1:
            m = np.broadcast_to(m.transpose(1, 2, 0), (H, W, C))
        elif m.shape[0] == C:
            m = m.transpose(1, 2, 0)
        else:
            raise ValueError(f"Unexpected mask shape after squeeze: {m.shape}")
    elif m.ndim == 3 and m.shape == (H, W, 1):
        m = np.broadcast_to(m, (H, W, C))
    elif m.ndim == 2:
        m = np.broadcast_to(m[:, :, None], (H, W, C))
    m = (m > 0.5).astype(np.float64)  # binarise
    assert m.shape == (H, W, C), f"Mask shape {m.shape} != HSI shape {(H, W, C)}"

    # ── mask-type-aware imputation ──────────────────────────────────────
    data = hsi_hwc.astype(np.float64)
    if mask_type == "auto":
        # Auto-detect: if mask differs across channels -> elementwise
        is_elementwise = not np.all(m[:, :, 0:1] == m)
        effective_type = "elementwise" if is_elementwise else "random"
    else:
        effective_type = mask_type

    if effective_type == "elementwise":
        filled = _fill_elementwise_spectral(data, m)
        print(f"  [fill] elementwise -> per-pixel spectral mean + spatial fallback")
    else:
        filled = _fill_pixelwise_local_spatial(data, m)
        print(f"  [fill] {effective_type} -> local spatial same-band mean")

    filled = np.clip(filled, 0, None)  # ensure non-negative for NMF

    # ── reshape to (C, H*W) — same convention as legacy code ────────────
    I = filled.transpose(2, 0, 1).reshape(C, H * W)  # (C, N)

    # ── NMF ─────────────────────────────────────────────────────────────
    print(f"Running masked NMF (rank={rank}, max_iter={max_iter}) ...")
    nmf = NMF(rank, init='random', random_state=random_state, max_iter=max_iter)
    W = nmf.fit_transform(I)       # (C, rank)
    endmember = W.T                # (rank, C)
    abundance = nmf.components_.T  # (H*W, rank)


    return endmember.astype(np.float32), abundance.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────
# Legacy full-GT NMF  (DEPRECATED for inpainting)
# ─────────────────────────────────────────────────────────────────────────

def load_dataset(name):
    """Load and normalize hyperspectral dataset.  Returns (C, H*W)."""
    name = name.lower()
    if name == "salinas":
        data = scipy.io.loadmat("HSI/data/Salinas_crop.mat")['I'].astype(float)
        data = np.clip(data, 0, None)
        for i in range(204):
            data[:, :, i] /= np.max(data[:, :, i])
    elif name == "urban":
        data = scipy.io.loadmat("HSI/data/Urban_R162.mat")['Y'].astype(float)
        for i in range(162):
            data[i, :] /= np.max(data[i, :])
        data = data.reshape(162, 307, 307).transpose(2, 1, 0)
    elif name == "jasperridge":
        data = scipy.io.loadmat("HSI/data/jasperRidge2_R198.mat")['Y'].astype(float)
        for i in range(198):
            data[i, :] /= np.max(data[i, :])
        data = data.reshape(198, 100, 100).transpose(2, 1, 0)
    elif name == "paviau":
        data = scipy.io.loadmat("HSI/data/PaviaU.mat")['paviaU'].astype(float)
        for i in range(103):
            data[:, :, i] /= np.max(data[:, :, i])
        data = data[-340:, :, :]
    else:
        raise ValueError(f"Unknown dataset: {name}")
    data = np.transpose(data, (2, 0, 1)).reshape(data.shape[2], -1)
    return data