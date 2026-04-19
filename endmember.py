"""
Endmember initialization via NMF.

Provides two entry points:
  1. masked_nmf_initialization — uses only masked (observed) HSI pixels.
     Missing positions are filled with per-band observed mean before NMF.
     This is the ONLY path that should be used for HSI inpainting.
  2. (legacy) nmf_initialization — uses full GT HSI.  DEPRECATED for
     inpainting because it leaks test information.
"""

import argparse
import numpy as np
import scipy.io
from sklearn.decomposition import NMF
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
# Masked NMF initialisation  (NO test leakage)
# ─────────────────────────────────────────────────────────────────────────

def masked_nmf_initialization(hsi_hwc: np.ndarray,
                              mask: np.ndarray,
                              rank: int,
                              dataset_name: str = "",
                              save: bool = False,
                              max_iter: int = 12000,
                              random_state: int = 42):
    """NMF on masked HSI — missing pixels filled with per-band observed mean.

    Args:
        hsi_hwc:  (H, W, C) float numpy array, the **full** GT HSI.
                  Only the observed entries (mask==1) will be used.
        mask:     Binary mask broadcastable to (H, W, C).
                  Accepted shapes: (H, W), (H, W, 1), (1, 1, H, W),
                  (1, C, H, W), (H, W, C).  Will be reshaped internally.
        rank:     Number of endmembers.
        dataset_name: For file-naming when *save* is True.
        save:     Whether to save .npy files under ``HSI/init/``.
        max_iter: Maximum NMF iterations.
        random_state: Random seed for NMF.

    Returns:
        endmember: (rank, C) numpy array  —  E0.
        abundance: (H*W, rank) numpy array — A0.
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

    # ── per-band observed-mean fill ─────────────────────────────────────
    filled = hsi_hwc.copy().astype(np.float64)
    for c in range(C):
        band_mask = m[:, :, c]  # (H, W)
        observed = filled[:, :, c][band_mask > 0.5]
        if observed.size == 0:
            band_mean = 0.0
        else:
            band_mean = observed.mean()
        filled[:, :, c] = np.where(band_mask > 0.5, filled[:, :, c], band_mean)
    filled = np.clip(filled, 0, None)  # ensure non-negative for NMF

    # ── reshape to (C, H*W) — same convention as legacy code ────────────
    I = filled.transpose(2, 0, 1).reshape(C, H * W)  # (C, N)

    # ── NMF ─────────────────────────────────────────────────────────────
    print(f"Running masked NMF (rank={rank}, max_iter={max_iter}) ...")
    nmf = NMF(rank, init='random', random_state=random_state, max_iter=max_iter)
    W = nmf.fit_transform(I)       # (C, rank)
    endmember = W.T                # (rank, C)
    abundance = nmf.components_.T  # (H*W, rank)

    # ── optionally save with masked-NMF naming ──────────────────────────
    if save and dataset_name:
        canonical = _canonical(dataset_name)
        os.makedirs("HSI/init", exist_ok=True)
        np.save(f"HSI/init/{canonical}_endmember_rank_{rank}_maskedNMF.npy", endmember)
        np.save(f"HSI/init/{canonical}_abundance_rank_{rank}_maskedNMF.npy", abundance)
        print(f"Saved masked-NMF results for {dataset_name}.")

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


def nmf_initialization(I, rank, dataset_name):
    """DEPRECATED — uses full GT. Use masked_nmf_initialization instead."""
    import warnings
    warnings.warn(
        "nmf_initialization uses the full GT HSI and leaks test information. "
        "Use masked_nmf_initialization for inpainting.",
        DeprecationWarning, stacklevel=2,
    )
    canonical = _canonical(dataset_name)
    print(f"[DEPRECATED] Running full-GT NMF on {dataset_name} with rank={rank}")
    nmf = NMF(rank, init='random', random_state=42, max_iter=12000)
    endmember = nmf.fit_transform(I).T
    abundance = nmf.components_.T
    os.makedirs("HSI/init", exist_ok=True)
    np.save(f"HSI/init/{canonical}_endmember_rank_{rank}_NMF.npy", endmember)
    np.save(f"HSI/init/{canonical}_abundance_rank_{rank}_NMF.npy", abundance)
    print(f"Saved (DEPRECATED) full-GT NMF results for {dataset_name}.")


def main():
    parser = argparse.ArgumentParser(
        description="NMF initialization for hyperspectral endmembers."
    )
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name: Salinas | Urban | JasperRidge | PaviaU")
    parser.add_argument("--rank", type=int, default=12,
                        help="Number of endmembers (rank)")
    parser.add_argument("--mask_ratio", type=float, default=0.0,
                        help="If >0, simulate a random mask and run masked NMF "
                             "(for offline pre-computation).")
    parser.add_argument("--mask_type", type=str, default="random",
                        choices=["random", "elementwise"],
                        help="Mask type for masked NMF.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    start_time = time.time()

    if args.mask_ratio > 0:
        # ── masked NMF path ─────────────────────────────────────────────
        # Load as (H, W, C) for masked_nmf_initialization
        name = args.dataset.lower()
        if name == "salinas":
            data = scipy.io.loadmat("HSI/data/Salinas_crop.mat")['I'].astype(float)
            data = np.clip(data, 0, None)
            for i in range(data.shape[2]):
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

        H, W, C = data.shape
        rng = np.random.RandomState(args.seed)
        if args.mask_type == "elementwise":
            mask = (rng.rand(H, W, C) >= args.mask_ratio).astype(float)
        else:
            mask = (rng.rand(H, W, 1) >= args.mask_ratio).astype(float)
            mask = np.broadcast_to(mask, (H, W, C)).copy()

        masked_nmf_initialization(data, mask, args.rank, args.dataset, save=True)
    else:
        # ── legacy full-GT NMF ──────────────────────────────────────────
        I = load_dataset(args.dataset)
        nmf_initialization(I, args.rank, args.dataset)

    print(f"Finished in {time.time() - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()
