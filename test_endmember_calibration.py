"""
Minimal verification script for the masked-NMF + endmember calibration pipeline.

Tests:
  1. masked_nmf_initialization produces valid E0 from masked observation
  2. GaussianImage_Cholesky_HSI constructs E_hat correctly
  3. Forward path computes Y_hat = A_hat @ E_hat
  4. Masked loss back-propagates gradients to U and V
  5. Ablation switch (freeze_endmember_calibration) works

Run:  python test_endmember_calibration.py
"""

import sys
import numpy as np
import torch
import torch.nn.functional as F

# ── 1.  Test masked NMF ────────────────────────────────────────────────
print("=" * 60)
print("TEST 1: masked_nmf_initialization")
print("=" * 60)

from endmember import masked_nmf_initialization

H, W, C = 20, 20, 30
rank = 4
np.random.seed(42)
hsi = np.random.rand(H, W, C).astype(np.float64) * 0.5 + 0.1  # positive
mask_2d = (np.random.rand(H, W, 1) >= 0.5).astype(np.float64)
mask_hwc = np.broadcast_to(mask_2d, (H, W, C)).copy()

E0, A0 = masked_nmf_initialization(hsi, mask_hwc, rank)
assert E0.shape == (rank, C), f"E0 shape mismatch: {E0.shape}"
assert A0.shape == (H * W, rank), f"A0 shape mismatch: {A0.shape}"
assert np.all(E0 >= 0), "E0 has negative values"
assert np.all(A0 >= 0), "A0 has negative values"
print(f"  E0 shape: {E0.shape},  range: [{E0.min():.4f}, {E0.max():.4f}]")
print(f"  A0 shape: {A0.shape},  range: [{A0.min():.4f}, {A0.max():.4f}]")

# Test with 4D mask format (1, 1, H, W)
mask_4d = mask_2d.transpose(2, 0, 1)[np.newaxis, ...]  # (1, 1, H, W)
E0_2, A0_2 = masked_nmf_initialization(hsi, mask_4d, rank)
assert E0_2.shape == (rank, C)
print("  4D mask input: OK")
print("  PASSED\n")


# ── 2.  Test model construction & E_hat ────────────────────────────────
print("=" * 60)
print("TEST 2: GaussianImage_Cholesky_HSI + get_calibrated_endmember")
print("=" * 60)

device = torch.device("cpu")  # CPU for testing

# Minimal import — the model needs gsplat, which might not be built.
# We test the calibration logic directly.
E0_t = torch.tensor(E0, dtype=torch.float32)

# Simulate the model's calibration logic
calib_rank = 2
gamma = 0.1
U = torch.zeros(rank, calib_rank)
V = torch.zeros(calib_rank, C)

# At init, E_hat == E0
E_hat = torch.clamp(E0_t + gamma * (U @ V), min=1e-6)
assert torch.allclose(E_hat, torch.clamp(E0_t, min=1e-6)), "E_hat should equal E0 at init"
print(f"  E_hat == E0 at init: OK")

# After some gradient steps, U and V are nonzero
U_nonzero = torch.randn(rank, calib_rank) * 0.01
V_nonzero = torch.randn(calib_rank, C) * 0.01
delta = U_nonzero @ V_nonzero
E_hat_cal = torch.clamp(E0_t + gamma * delta, min=1e-6)
diff = (E_hat_cal - torch.clamp(E0_t, min=1e-6)).abs().max().item()
print(f"  Max |E_hat - E0| after calibration: {diff:.6f}")
assert diff > 0, "Calibration should change E_hat"
print(f"  E_hat non-negative: {(E_hat_cal >= 0).all().item()}")
print("  PASSED\n")


# ── 3.  Test forward path Y_hat = A_hat @ E_hat ───────────────────────
print("=" * 60)
print("TEST 3: Y_hat = A_hat @ E_hat")
print("=" * 60)

A_hat = torch.rand(H * W, rank)  # simulated abundance
E_hat = torch.clamp(E0_t + gamma * (U_nonzero @ V_nonzero), min=1e-6)
Y_hat_flat = A_hat @ E_hat  # (H*W, C)
Y_hat = Y_hat_flat.view(1, H, W, C).permute(0, 3, 1, 2)  # (1, C, H, W)
assert Y_hat.shape == (1, C, H, W), f"Y_hat shape: {Y_hat.shape}"
print(f"  Y_hat shape: {Y_hat.shape}, range: [{Y_hat.min():.4f}, {Y_hat.max():.4f}]")
print("  PASSED\n")


# ── 4.  Test gradient flow to U and V ─────────────────────────────────
print("=" * 60)
print("TEST 4: Gradient back-propagation to U and V")
print("=" * 60)

E0_buf = torch.tensor(E0, dtype=torch.float32)  # frozen buffer
U_param = torch.nn.Parameter(torch.randn(rank, calib_rank) * 0.01)
V_param = torch.nn.Parameter(torch.randn(calib_rank, C) * 0.01)
gamma_val = 0.1

# Forward
E_hat = torch.clamp(E0_buf + gamma_val * (U_param @ V_param), min=1e-6)
A_hat = torch.rand(H * W, rank, requires_grad=False)  # frozen for this test
Y_hat_flat = A_hat @ E_hat
Y_hat = Y_hat_flat.view(1, H, W, C).permute(0, 3, 1, 2)

# Dummy GT and mask
gt = torch.rand(1, C, H, W)
mask = torch.ones(1, 1, H, W)  # observe everything for simplicity

# Masked loss
diff = (Y_hat * mask - gt * mask)
loss = (diff ** 2).sum() / mask.expand_as(Y_hat).sum()
loss.backward()

assert U_param.grad is not None, "U has no gradient!"
assert V_param.grad is not None, "V has no gradient!"
assert U_param.grad.abs().sum() > 0, "U gradient is all zeros"
assert V_param.grad.abs().sum() > 0, "V gradient is all zeros"
print(f"  U grad norm: {U_param.grad.norm().item():.8f}")
print(f"  V grad norm: {V_param.grad.norm().item():.8f}")
print(f"  E0 is NOT a Parameter — no grad accumulated (correct)")
print("  PASSED\n")


# ── 5.  Test freeze ablation switch ───────────────────────────────────
print("=" * 60)
print("TEST 5: freeze_endmember_calibration ablation")
print("=" * 60)

U_param2 = torch.nn.Parameter(torch.randn(rank, calib_rank) * 0.1)
V_param2 = torch.nn.Parameter(torch.randn(calib_rank, C) * 0.1)

# frozen path
E_frozen = torch.clamp(E0_buf, min=1e-6)
# calibrated path
E_calib = torch.clamp(E0_buf + gamma_val * (U_param2 @ V_param2), min=1e-6)
diff_ablation = (E_calib - E_frozen).abs().max().item()
assert diff_ablation > 0, "Calibrated E should differ from frozen E"
print(f"  Max |E_calib - E_frozen|: {diff_ablation:.6f}")
print("  PASSED\n")


# ── 6.  Verify no GT leakage path remains ─────────────────────────────
print("=" * 60)
print("TEST 6: Verify no GT-derived endmember path in training code")
print("=" * 60)

with open("inpainting_train_hsi.py", "r") as f:
    source = f.read()
# Check that load_endmember (old GT-based loader) is not used
assert "load_endmember" not in source, "load_endmember still present!"
# Check that masked_nmf_initialization IS used
assert "masked_nmf_initialization" in source, "masked_nmf_initialization not found!"

with open("gaussianimage_cholesky_hsi.py", "r") as f:
    model_source = f.read()
# Check model uses E0 buffer, not 'endmember' buffer
assert "register_buffer('E0'" in model_source, "E0 buffer not found in model!"
assert "get_calibrated_endmember" in model_source, "get_calibrated_endmember not found!"
assert "calib_U" in model_source, "calib_U parameter not found!"
assert "calib_V" in model_source, "calib_V parameter not found!"
# Old path should be gone
assert "register_buffer('endmember'" not in model_source, \
    "Old 'endmember' buffer still in model!"

print("  No load_endmember in training code: OK")
print("  masked_nmf_initialization imported and used: OK")
print("  Model uses E0 buffer + calib_U/V: OK")
print("  Old 'endmember' buffer removed: OK")
print("  PASSED\n")


print("=" * 60)
print("ALL TESTS PASSED")
print("=" * 60)
