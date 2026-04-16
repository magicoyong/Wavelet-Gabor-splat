"""Quick integration test for init_strategies + inpainting_train."""
import torch
from gaussianimage_cholesky import GaussianImage_Cholesky
from init_strategies import apply_obs_first_init
from inpainting_train import parse_args

H, W = 64, 64
N = 500
num_gabor = 2

# --- Test 1: pixel-wise mask ---
model = GaussianImage_Cholesky(
    loss_type="L2", opt_type="adam",
    num_points=N, H=H, W=W,
    BLOCK_H=16, BLOCK_W=16,
    device="cpu", lr=1e-3,
    num_gabor=num_gabor, quantize=False,
)
mask = (torch.rand(1, 1, H, W) >= 0.5).float()
gt_image = torch.rand(1, 3, H, W)

orig_xyz = model._xyz.data.clone()
apply_obs_first_init(model, mask, gt_image, obs_init_ratio=0.8)

assert model._xyz.shape == (N, 2)
assert model._features_dc.shape == (N, 3)
assert model.gabor_freqs.shape == (N * num_gabor, 2)
assert not torch.allclose(model._xyz.data, orig_xyz)
assert torch.isfinite(model._xyz.data).all()
assert torch.isfinite(model._features_dc.data).all()
xy = torch.tanh(model._xyz.data)
assert xy.min() >= -1.0 and xy.max() <= 1.0
print("PASS: pixel-wise mask integration")

# --- Test 2: elementwise mask ---
model2 = GaussianImage_Cholesky(
    loss_type="L2", opt_type="adam",
    num_points=N, H=H, W=W,
    BLOCK_H=16, BLOCK_W=16,
    device="cpu", lr=1e-3,
    num_gabor=num_gabor, quantize=False,
)
mask_elem = (torch.rand(1, 3, H, W) >= 0.7).float()
apply_obs_first_init(model2, mask_elem, gt_image, obs_init_ratio=0.9)
assert model2._xyz.shape == (N, 2)
assert torch.isfinite(model2._xyz.data).all()
print("PASS: elementwise mask integration")

# --- Test 3: CLI arg parsing ---
args = parse_args(["--image_path", "dummy.png"])
assert args.init_strategy == "random"
assert args.obs_init_ratio == 0.8

args2 = parse_args(["--image_path", "dummy.png", "--init_strategy", "obs_first", "--obs_init_ratio", "0.7"])
assert args2.init_strategy == "obs_first"
assert args2.obs_init_ratio == 0.7
print("PASS: CLI parsing")

print("\nAll integration tests passed.")
