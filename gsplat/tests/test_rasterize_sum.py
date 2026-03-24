import pytest
import torch


device = torch.device("cuda:0")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_rasterize_empty_scene():
    from gsplat import rasterize_gaussians_sum

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

    out = rasterize_gaussians_sum(
        xys,
        depths,
        radii,
        conics,
        num_tiles_hit,
        colors,
        opacity,
        H,
        W,
        BLOCK_H=8,
        BLOCK_W=8,
        background=bg,
        return_alpha=False,
    )

    assert out.shape == (H, W, 3)
    # empty scene should be filled with background
    torch.testing.assert_close(out, torch.ones_like(out) * bg)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_rasterize_compare_to_reference():
    from gsplat import _torch_impl3d
    from gsplat import rasterize_gaussians_sum
    from gsplat.utils import bin_and_sort_gaussians

    torch.manual_seed(0)

    num_points = 40
    means3d = torch.randn((num_points, 3), device=device)
    scales = torch.rand((num_points, 3), device=device) * 0.5 + 0.1
    glob_scale = 0.3
    quats = torch.randn((num_points, 4), device=device)
    quats = quats / torch.linalg.norm(quats, dim=-1, keepdim=True)
    viewmat = torch.eye(4, device=device)
    projmat = torch.eye(4, device=device)
    fx, fy = 3.0, 3.0
    H, W = 64, 64
    BLOCK_X, BLOCK_Y = 16, 16
    tile_bounds = ((W + BLOCK_X - 1) // BLOCK_X, (H + BLOCK_Y - 1) // BLOCK_Y, 1)

    (
        _cov3d,
        xys,
        depths,
        radii,
        conics,
        num_tiles_hit,
        mask,
    ) = _torch_impl3d.project_gaussians_forward(
        means3d,
        scales,
        glob_scale,
        quats,
        viewmat,
        projmat,
        fx,
        fy,
        (W, H),
        tile_bounds,
        0.01,
    )

    # keep only valid gaussians
    xys = xys[mask]
    depths = depths[mask]
    radii = radii[mask]
    conics = conics[mask]
    num_tiles_hit = num_tiles_hit[mask]

    num_points_valid = xys.shape[0]
    assert num_points_valid >= 0

    colors = torch.rand((num_points_valid, 3), device=device)
    opacity = torch.rand((num_points_valid, 1), device=device)

    # compute sorting/bins using utils
    num_intersects = torch.cumsum(num_tiles_hit, dim=0)[-1].item() if num_points_valid > 0 else 0
    if num_intersects == 0:
        # fall back to empty-scene behavior
        out = rasterize_gaussians_sum(
            xys,
            depths,
            radii,
            conics,
            num_tiles_hit,
            colors,
            opacity,
            H,
            W,
            BLOCK_H=BLOCK_Y,
            BLOCK_W=BLOCK_X,
        )
        assert out.shape == (H, W, 3)
        return

    cum_tiles_hit = torch.cumsum(num_tiles_hit, dim=0).to(torch.int32)

    (
        isect_ids_unsorted,
        gaussian_ids_unsorted,
        isect_ids_sorted,
        gaussian_ids_sorted,
        tile_bins,
    ) = bin_and_sort_gaussians(
        num_points_valid,
        num_intersects,
        xys,
        depths,
        radii,
        cum_tiles_hit,
        tile_bounds,
    )

    # reference implementation (pure PyTorch)
    ref_out, ref_final_Ts, ref_final_idx = _torch_impl3d.rasterize_forward(
        tile_bounds,
        (BLOCK_X, BLOCK_Y, 1),
        (W, H, 1),
        gaussian_ids_sorted,
        tile_bins,
        xys,
        conics,
        colors,
        opacity,
        torch.ones(3, device=device),
    )

    # our GPU implementation via wrapper
    out = rasterize_gaussians_sum(
        xys,
        depths,
        radii,
        conics,
        num_tiles_hit,
        colors,
        opacity,
        H,
        W,
        BLOCK_H=BLOCK_Y,
        BLOCK_W=BLOCK_X,
        background=torch.ones(3, device=device),
    )

    torch.testing.assert_close(ref_out, out, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    test_rasterize_empty_scene()
    test_rasterize_compare_to_reference()
