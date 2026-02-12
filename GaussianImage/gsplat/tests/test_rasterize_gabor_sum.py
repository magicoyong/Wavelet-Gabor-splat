import pytest
import torch


device = torch.device("cuda:0")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_rasterize_empty_scene():
    from gsplat import rasterize_gabor_sum

    # minimal empty scene: no intersects
    num_points = 4
    xys = torch.zeros((num_points, 2), device=device)
    depths = torch.zeros((num_points, 1), device=device)
    radii = torch.zeros((num_points, 1), dtype=torch.int32, device=device)
    conics = torch.zeros((num_points, 3), device=device)
    num_tiles_hit = torch.zeros((num_points, 1), dtype=torch.int32, device=device)
    colors = torch.ones((num_points, 3), device=device)
    opacity = torch.zeros((num_points, 1), device=device)

    H, W = 16, 16
    bg = torch.tensor([0.2, 0.3, 0.4], device=device)

    out = rasterize_gabor_sum(
        xys,
        depths,
        radii,
        conics,
        num_tiles_hit,
        colors,
        opacity,
        gabor_freqs_x=torch.zeros((num_points * 1,), device=device),
        gabor_freqs_y=torch.zeros((num_points * 1,), device=device),
        gabor_weights=torch.zeros((num_points * 1,), device=device),
        num_freqs=1,
        img_height=H,
        img_width=W,
        BLOCK_H=8,
        BLOCK_W=8,
        background=bg,
        return_alpha=False,
    )

    assert out.shape == (H, W, 3)
    # empty scene should be filled with background
    torch.testing.assert_close(out, torch.ones_like(out) * bg)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_rasterize_gabor_forward_backward():
    from gsplat import rasterize_gabor_sum
    from gsplat.project_gaussians_2d import project_gaussians_2d

    torch.manual_seed(0)


    # Use the 2D projector to generate 2D gaussians and tile data.
    num_points = 20
    # means2d in [0,1] normalized image coordinates, L_elements packs 2D L (l11,l21,l22)
    means2d = torch.rand((num_points, 2), device=device)
    L_elements = torch.rand((num_points, 3), device=device) * 0.5 + 0.1
    H, W = 64, 64
    BLOCK_X, BLOCK_Y = 16, 16
    tile_bounds = ((W + BLOCK_X - 1) // BLOCK_X, (H + BLOCK_Y - 1) // BLOCK_Y, 1)

    xys, depths, radii, conics, num_tiles_hit = project_gaussians_2d(
        means2d, L_elements, H, W, tile_bounds, 0.01
    )

    # keep only valid gaussians (radii > 0)
    mask = radii > 0
    xys = xys[mask.squeeze()]
    depths = depths[mask.squeeze()]
    radii = radii[mask.squeeze()]
    conics = conics[mask.squeeze()]
    num_tiles_hit = num_tiles_hit[mask.squeeze()]

    num_points_valid = xys.shape[0]
    if num_points_valid == 0:
        pytest.skip("no valid gaussians from project_gaussians_2d")

    colors = torch.rand((num_points_valid, 3), device=device)
    opacity = torch.rand((num_points_valid, 1), device=device)


    # simple forward + backward gradient check for Gabor params
    num_freqs = 2
    gabor_freqs_x = torch.rand((num_points_valid * num_freqs,), device=device, requires_grad=True)
    gabor_freqs_y = torch.rand((num_points_valid * num_freqs,), device=device, requires_grad=True)
    gabor_weights = torch.rand((num_points_valid * num_freqs,), device=device, requires_grad=True)

    out = rasterize_gabor_sum(
        xys,
        depths,
        radii,
        conics,
        num_tiles_hit,
        colors,
        opacity,
        gabor_freqs_x,
        gabor_freqs_y,
        gabor_weights,
        num_freqs,
        H,
        W,
        BLOCK_H=BLOCK_Y,
        BLOCK_W=BLOCK_X,
        background=torch.ones(3, device=device),
    )

    loss = out.sum()
    loss.backward()

    # gradients should exist for gabor params
    assert gabor_freqs_x.grad is not None
    assert gabor_freqs_y.grad is not None
    assert gabor_weights.grad is not None
    # grads should have correct shapes
    assert gabor_freqs_x.grad.shape == gabor_freqs_x.shape
    assert gabor_freqs_y.grad.shape == gabor_freqs_y.shape
    assert gabor_weights.grad.shape == gabor_weights.shape


if __name__ == "__main__":
    test_rasterize_empty_scene()
    test_rasterize_gabor_forward_backward()
