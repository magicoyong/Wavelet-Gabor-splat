"""Python bindings for custom Cuda functions"""

from typing import Optional

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.autograd import Function

import gsplat.cuda as _C
from .utils import bin_and_sort_gaussians, compute_cumulative_intersects


def rasterize_gaussians_sum(
    xys: Float[Tensor, "*batch 2"],
    depths: Float[Tensor, "*batch 1"],
    radii: Float[Tensor, "*batch 1"],
    conics: Float[Tensor, "*batch 3"],
    num_tiles_hit: Int[Tensor, "*batch 1"],
    colors: Float[Tensor, "*batch channels"],
    opacity: Float[Tensor, "*batch 1"],
    img_height: int,
    img_width: int,
    BLOCK_H: int=16,
    BLOCK_W: int=16, 
    background: Optional[Float[Tensor, "channels"]] = None,
    return_alpha: Optional[bool] = False,
) -> Tensor:
    """Rasterizes 2D gaussians by sorting and binning gaussian intersections for each tile and returns an N-dimensional output using alpha-compositing.

    Note:
        This function is differentiable w.r.t the xys, conics, colors, and opacity inputs.

    Args:
        xys (Tensor): xy coords of 2D gaussians.
        depths (Tensor): depths of 2D gaussians.
        radii (Tensor): radii of 2D gaussians
        conics (Tensor): conics (inverse of covariance) of 2D gaussians in upper triangular format
        num_tiles_hit (Tensor): number of tiles hit per gaussian
        colors (Tensor): N-dimensional features associated with the gaussians.
        opacity (Tensor): opacity associated with the gaussians.
        img_height (int): height of the rendered image.
        img_width (int): width of the rendered image.
        background (Tensor): background color
        return_alpha (bool): whether to return alpha channel

    Returns:
        A Tensor:

        - **out_img** (Tensor): N-dimensional rendered output image.
        - **out_alpha** (Optional[Tensor]): Alpha channel of the rendered output image.
    """
    if colors.dtype == torch.uint8:
        # make sure colors are float [0,1]
        colors = colors.float() / 255

    if background is not None:
        assert (
            background.shape[0] == colors.shape[-1]
        ), f"incorrect shape of background color tensor, expected shape {colors.shape[-1]}"
    else:
        background = torch.ones(
            colors.shape[-1], dtype=torch.float32, device=colors.device
        )

    if xys.ndimension() != 2 or xys.size(1) != 2:
        raise ValueError("xys must have dimensions (N, 2)")

    if colors.ndimension() != 2:
        raise ValueError("colors must have dimensions (N, D)")

    return _RasterizeGaussiansSum.apply(
        xys.contiguous(),
        depths.contiguous(),
        radii.contiguous(),
        conics.contiguous(),
        num_tiles_hit.contiguous(),
        colors.contiguous(),
        opacity.contiguous(),
        img_height,
        img_width,
        BLOCK_H, 
        BLOCK_W,
        background.contiguous(),
        return_alpha,
    )


class _RasterizeGaussiansSum(Function):
    """Rasterizes 2D gaussians"""

    @staticmethod
    def forward(
        ctx,
        xys: Float[Tensor, "*batch 2"],
        depths: Float[Tensor, "*batch 1"],
        radii: Float[Tensor, "*batch 1"],
        conics: Float[Tensor, "*batch 3"],
        num_tiles_hit: Int[Tensor, "*batch 1"],
        colors: Float[Tensor, "*batch channels"],
        opacity: Float[Tensor, "*batch 1"],
        img_height: int,
        img_width: int,
        BLOCK_H: int=16,
        BLOCK_W: int=16, 
        background: Optional[Float[Tensor, "channels"]] = None,
        return_alpha: Optional[bool] = False,
    ) -> Tensor:
        num_points = xys.size(0)
        BLOCK_X, BLOCK_Y = BLOCK_W, BLOCK_H
        tile_bounds = (
            (img_width + BLOCK_X - 1) // BLOCK_X,
            (img_height + BLOCK_Y - 1) // BLOCK_Y,
            1,
        )
        block = (BLOCK_X, BLOCK_Y, 1)
        img_size = (img_width, img_height, 1)

        num_intersects, cum_tiles_hit = compute_cumulative_intersects(num_tiles_hit)

        if num_intersects < 1:
            out_img = (
                torch.ones(img_height, img_width, colors.shape[-1], device=xys.device)
                * background
            )
            gaussian_ids_sorted = torch.zeros(0, 1, device=xys.device)
            tile_bins = torch.zeros(0, 2, device=xys.device)
            final_Ts = torch.zeros(img_height, img_width, device=xys.device)
            final_idx = torch.zeros(img_height, img_width, device=xys.device)
        else:
            (
                isect_ids_unsorted,
                gaussian_ids_unsorted,
                isect_ids_sorted,
                gaussian_ids_sorted,
                tile_bins,
            ) = bin_and_sort_gaussians(
                num_points,
                num_intersects,
                xys,
                depths,
                radii,
                cum_tiles_hit,
                tile_bounds,
            )
            if colors.shape[-1] == 3:
                rasterize_fn = _C.rasterize_sum_forward
            else:
                rasterize_fn = _C.nd_rasterize_sum_forward

            out_img, final_Ts, final_idx = rasterize_fn(
                tile_bounds,
                block,
                img_size,
                gaussian_ids_sorted,
                tile_bins,
                xys,
                conics,
                colors,
                opacity,
                background,
            )

        ctx.img_width = img_width
        ctx.img_height = img_height
        ctx.BLOCK_H = BLOCK_H
        ctx.BLOCK_W = BLOCK_W
        ctx.num_intersects = num_intersects
        ctx.save_for_backward(
            gaussian_ids_sorted,
            tile_bins,
            xys,
            conics,
            colors,
            opacity,
            background,
            final_Ts,
            final_idx,
        )

        if return_alpha:
            out_alpha = 1 - final_Ts
            return out_img, out_alpha
        else:
            return out_img

    @staticmethod
    def backward(ctx, v_out_img, v_out_alpha=None):
        img_height = ctx.img_height
        img_width = ctx.img_width
        BLOCK_H = ctx.BLOCK_H
        BLOCK_W = ctx.BLOCK_W
        num_intersects = ctx.num_intersects

        if v_out_alpha is None:
            v_out_alpha = torch.zeros_like(v_out_img[..., 0])

        (
            gaussian_ids_sorted,
            tile_bins,
            xys,
            conics,
            colors,
            opacity,
            background,
            final_Ts,
            final_idx,
        ) = ctx.saved_tensors

        if num_intersects < 1:
            v_xy = torch.zeros_like(xys)
            v_conic = torch.zeros_like(conics)
            v_colors = torch.zeros_like(colors)
            v_opacity = torch.zeros_like(opacity)

        else:
            if colors.shape[-1] == 3:
                rasterize_fn = _C.rasterize_sum_backward
            else:
                rasterize_fn = _C.nd_rasterize_sum_backward
            v_xy, v_conic, v_colors, v_opacity = rasterize_fn(
                img_height,
                img_width,
                BLOCK_H,
                BLOCK_W,
                gaussian_ids_sorted,
                tile_bins,
                xys,
                conics,
                colors,
                opacity,
                background,
                final_Ts,
                final_idx,
                v_out_img,
                v_out_alpha,
            )

        return (
            v_xy,  # xys
            None,  # depths
            None,  # radii
            v_conic,  # conics
            None,  # num_tiles_hit
            v_colors,  # colors
            v_opacity,  # opacity
            None,  # img_height
            None,  # img_width
            None,  # block_w
            None,  # block_h
            None,  # background
            None,  # return_alpha
        )


def rasterize_gaussians_sum_nd(
    xys: Float[Tensor, "*batch 2"],
    depths: Float[Tensor, "*batch 1"],
    radii: Float[Tensor, "*batch 1"],
    conics: Float[Tensor, "*batch 3"],
    num_tiles_hit: Int[Tensor, "*batch 1"],
    colors: Float[Tensor, "*batch channels"],
    opacity: Float[Tensor, "*batch 1"],
    img_height: int,
    img_width: int,
    BLOCK_H: int = 16,
    BLOCK_W: int = 16,
    background: Optional[Float[Tensor, "channels"]] = None,
    return_alpha: Optional[bool] = False,
) -> Tensor:
    """Multi-channel Gaussian sum-mode rasterization via 3-ch + 4-ch split.

    The CUDA kernels only support exactly 3 channels (float3) or 4 channels
    (float4).  This wrapper decomposes ``colors`` (N, D) into groups of 3 and
    4, renders each group through rasterize_gaussians_sum, and concatenates.
    For any D >= 3, we find non-negative integers (a, b) such that
    3a + 4b = D (except D=1,2,5 which use padding).
    """
    C = colors.shape[-1]

    # Direct path for small channel counts
    if C <= 4:
        if background is not None:
            bg = background[:C]
        else:
            bg = None
        return rasterize_gaussians_sum(
            xys, depths, radii, conics, num_tiles_hit,
            colors, opacity,
            img_height, img_width, BLOCK_H, BLOCK_W,
            background=bg, return_alpha=return_alpha,
        )

    # Decompose C into groups of 3 and 4: find (n3, n4) s.t. 3*n3 + 4*n4 = C
    remainder = C % 3
    if remainder == 0:
        n4, n3 = 0, C // 3
    elif remainder == 1 and C >= 4:
        n4, n3 = 1, (C - 4) // 3
    elif remainder == 2 and C >= 8:
        n4, n3 = 2, (C - 8) // 3
    else:
        if C <= 3:
            n4, n3 = 0, 1
        elif C == 5:
            n4, n3 = 0, 2
        else:
            n4, n3 = 0, (C + 2) // 3

    split_sizes = [3] * n3 + [4] * n4
    total = sum(split_sizes)

    # Pad colors if decomposition overshoots
    if total > C:
        pad_n = total - C
        colors = torch.cat([
            colors,
            torch.zeros(colors.shape[0], pad_n, device=colors.device, dtype=colors.dtype),
        ], dim=1)
        if background is not None:
            background = torch.cat([
                background,
                torch.zeros(pad_n, device=background.device, dtype=background.dtype),
            ], dim=0)

    # Split and render each group
    color_groups = torch.split(colors, split_sizes, dim=1)
    if background is not None:
        bg_groups = torch.split(background, split_sizes, dim=0)
    else:
        bg_groups = [None] * len(color_groups)

    out_imgs = []
    for cg_colors, cg_bg in zip(color_groups, bg_groups):
        out = rasterize_gaussians_sum(
            xys, depths, radii, conics, num_tiles_hit,
            cg_colors, opacity,
            img_height, img_width, BLOCK_H, BLOCK_W,
            background=cg_bg, return_alpha=False,
        )
        out_imgs.append(out)

    result = torch.cat(out_imgs, dim=-1)  # (H, W, total)

    # Trim padding channels
    if total > C:
        result = result[..., :C]

    return result
