"""
Minimal tests for the inpainting pipeline.

Test 1: masked loss only affects observed region.
Test 2: small-scale training reduces observed-region error.
"""

import math
import torch
import torch.nn.functional as F
import pytest
import numpy as np

from inpainting_utils import (
    generate_mask,
    generate_random_mask,
    generate_block_mask,
    masked_mse_loss,
    masked_l1_loss,
    masked_loss_fn,
    gabor_weights_l1_reg,
    position_l2_reg,
    cholesky_l2_reg,
    compute_psnr,
    compute_error_map,
    mask_to_image_tensor,
    save_tensor_as_image,
)


# -----------------------------------------------------------------------
# Test 1: masked loss only applies to observed (mask==1) pixels
# -----------------------------------------------------------------------

class TestMaskedLoss:
    """Verify masked losses ignore missing pixels."""

    def test_masked_mse_ignores_missing(self):
        """If pred matches target on observed pixels but differs on missing
        pixels, the masked MSE should be 0."""
        H, W, C = 8, 8, 3
        target = torch.rand(1, C, H, W)
        # Create a mask: top-half observed, bottom-half missing
        mask = torch.zeros(1, 1, H, W)
        mask[:, :, :H // 2, :] = 1.0

        # pred == target on observed, different on missing
        pred = torch.rand(1, C, H, W)
        pred[:, :, :H // 2, :] = target[:, :, :H // 2, :]

        loss = masked_mse_loss(pred, target, mask)
        assert loss.item() == pytest.approx(0.0, abs=1e-7)

    def test_masked_l1_ignores_missing(self):
        """Same logic for L1."""
        H, W, C = 8, 8, 3
        target = torch.rand(1, C, H, W)
        mask = torch.zeros(1, 1, H, W)
        mask[:, :, :H // 2, :] = 1.0

        pred = torch.rand(1, C, H, W)
        pred[:, :, :H // 2, :] = target[:, :, :H // 2, :]

        loss = masked_l1_loss(pred, target, mask)
        assert loss.item() == pytest.approx(0.0, abs=1e-7)

    def test_masked_loss_nonzero_on_observed(self):
        """If pred differs from target on observed region, loss > 0."""
        H, W, C = 8, 8, 3
        target = torch.zeros(1, C, H, W)
        pred = torch.ones(1, C, H, W)
        mask = torch.ones(1, 1, H, W)  # all observed

        loss = masked_mse_loss(pred, target, mask)
        assert loss.item() > 0

    def test_masked_loss_fn_dispatch(self):
        """masked_loss_fn dispatches correctly to L1 and L2."""
        H, W, C = 4, 4, 3
        pred = torch.ones(1, C, H, W)
        target = torch.zeros(1, C, H, W)
        mask = torch.ones(1, 1, H, W)

        loss_l2 = masked_loss_fn(pred, target, mask, loss_type="L2")
        loss_l1 = masked_loss_fn(pred, target, mask, loss_type="L1")
        assert loss_l2.item() == pytest.approx(1.0, abs=1e-5)  # MSE of 1
        assert loss_l1.item() == pytest.approx(1.0, abs=1e-5)  # L1 of 1

    def test_all_missing_no_nan(self):
        """When mask is all zeros, loss should still be finite (not NaN)."""
        H, W, C = 4, 4, 3
        pred = torch.rand(1, C, H, W)
        target = torch.rand(1, C, H, W)
        mask = torch.zeros(1, 1, H, W)

        loss = masked_mse_loss(pred, target, mask)
        assert torch.isfinite(loss)


# -----------------------------------------------------------------------
# Test: mask generation
# -----------------------------------------------------------------------

class TestMaskGeneration:
    def test_random_mask_shape(self):
        mask = generate_random_mask(32, 32, mask_ratio=0.5)
        assert mask.shape == (1, 1, 32, 32)
        assert mask.min() >= 0 and mask.max() <= 1

    def test_block_mask_shape(self):
        mask = generate_block_mask(64, 64, block_size=16, num_blocks=2)
        assert mask.shape == (1, 1, 64, 64)
        # At least some pixels should be blocked
        assert mask.sum() < 64 * 64

    def test_random_mask_ratio_approx(self):
        """Random mask ratio should be approximately correct."""
        torch.manual_seed(42)
        mask = generate_random_mask(256, 256, mask_ratio=0.3)
        observed_ratio = mask.mean().item()
        assert abs(observed_ratio - 0.7) < 0.05  # within 5% tolerance

    def test_generate_mask_dispatch(self):
        mask_r = generate_mask(16, 16, mask_type="random")
        mask_b = generate_mask(16, 16, mask_type="block", block_size=4, num_blocks=1)
        assert mask_r.shape == (1, 1, 16, 16)
        assert mask_b.shape == (1, 1, 16, 16)


# -----------------------------------------------------------------------
# Test: utility functions
# -----------------------------------------------------------------------

class TestUtils:
    def test_compute_psnr_perfect(self):
        t = torch.rand(1, 3, 16, 16)
        psnr = compute_psnr(t, t)
        assert psnr == float('inf')

    def test_compute_psnr_finite(self):
        a = torch.zeros(1, 3, 16, 16)
        b = torch.ones(1, 3, 16, 16)
        psnr = compute_psnr(a, b)
        assert psnr == pytest.approx(0.0, abs=1e-3)

    def test_error_map_shape(self):
        a = torch.rand(1, 3, 8, 8)
        b = torch.rand(1, 3, 8, 8)
        err = compute_error_map(a, b)
        assert err.shape == a.shape

    def test_mask_to_image_tensor(self):
        mask = torch.ones(1, 1, 8, 8)
        img = mask_to_image_tensor(mask)
        assert img.shape == (1, 3, 8, 8)


# -----------------------------------------------------------------------
# Test 2: small-scale training reduces observed-region error (requires CUDA)
# -----------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_inpainting_training_reduces_error():
    """Train for a few steps on a tiny synthetic image, verify observed
    region MSE decreases."""
    from gaussianimage_cholesky import GaussianImage_Cholesky
    from inpainting_utils import masked_mse_loss, generate_random_mask

    device = torch.device("cuda:0")
    torch.manual_seed(42)
    np.random.seed(42)

    H, W = 32, 32
    BLOCK_H, BLOCK_W = 16, 16

    # Synthetic GT: solid-ish color with gradient
    gt = torch.zeros(1, 3, H, W, device=device)
    gt[:, 0, :, :] = torch.linspace(0, 1, W, device=device).unsqueeze(0)  # red gradient
    gt[:, 1, :, :] = 0.5
    gt[:, 2, :, :] = 0.3

    mask = generate_random_mask(H, W, mask_ratio=0.5, device=device)

    model = GaussianImage_Cholesky(
        loss_type="L2", opt_type="adan",
        num_points=200, H=H, W=W,
        BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
        device=device, lr=1e-2,
        num_gabor=2, quantize=False,
    ).to(device)

    # Measure initial error
    model.train()
    with torch.no_grad():
        out0 = model()["render"]
        loss0 = masked_mse_loss(out0, gt, mask).item()

    # Train for a few steps
    num_steps = 50
    for _ in range(num_steps):
        render_pkg = model.forward()
        image = render_pkg["render"]
        loss = masked_mse_loss(image, gt, mask)
        loss.backward()
        model.optimizer.step()
        model.optimizer.zero_grad(set_to_none=True)
        model.scheduler.step()

    # Measure final error
    with torch.no_grad():
        out1 = model()["render"]
        loss1 = masked_mse_loss(out1, gt, mask).item()

    assert loss1 < loss0, (
        f"Observed-region MSE should decrease: initial={loss0:.6f}, final={loss1:.6f}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
