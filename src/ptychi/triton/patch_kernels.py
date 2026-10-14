# Copyright © 2025 UChicago Argonne, LLC All right reserved
# Full license accessible at https://github.com//AdvancedPhotonSource/pty-chi/blob/main/LICENSE

"""Rectangle kernels, imported lazily by the optional Triton backend."""

import torch
import triton
import triton.language as tl


@triton.jit
def _accumulate_deterministic(
    image, patches, sy, sx, output,
    width: tl.constexpr, n_patches: tl.constexpr,
    patch_height: tl.constexpr, patch_width: tl.constexpr, components: tl.constexpr,
    patch_stride_0: tl.constexpr, patch_stride_1: tl.constexpr, patch_stride_2: tl.constexpr,
    sy_stride: tl.constexpr, sx_stride: tl.constexpr, block_size: tl.constexpr,
):
    # One owner per output component, summing patches in a fixed order.
    row = tl.program_id(1)
    start = tl.program_id(0) * block_size
    offsets = start + tl.arange(0, block_size)
    col = offsets // components
    component = offsets % components
    total = tl.full((block_size,), 0, tl.float32)
    for n in range(n_patches):
        y = tl.load(sy + n * sy_stride)
        x = tl.load(sx + n * sx_stride)
        if ((row >= y) & (row < y + patch_height)
                & (start // components < x + patch_width)
                & ((start + block_size - 1) // components >= x)):
            mask = (col >= x) & (col < x + patch_width) & (col < width)
            values = tl.load(
                patches + n * patch_stride_0 + (row - y) * patch_stride_1
                + (col - x) * patch_stride_2 + component,
                mask, 0,
            )
            total = total + values
    image_offsets = (row * width + col) * components + component
    previous = tl.load(image + image_offsets, col < width, 0)
    tl.store(output + image_offsets, previous + total, col < width)


@triton.jit
def _accumulate_atomic(
    patches, sy, sx, output,
    height: tl.constexpr, width: tl.constexpr,
    patch_height: tl.constexpr, patch_width: tl.constexpr, components: tl.constexpr,
    patch_stride_0: tl.constexpr, patch_stride_1: tl.constexpr, patch_stride_2: tl.constexpr,
    sy_stride: tl.constexpr, sx_stride: tl.constexpr, block_size: tl.constexpr,
):
    n = tl.program_id(1)
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    pixel = offsets // components
    component = offsets % components
    patch_y = pixel // patch_width
    patch_x = pixel % patch_width
    y = tl.load(sy + n * sy_stride) + patch_y
    x = tl.load(sx + n * sx_stride) + patch_x
    mask = offsets < patch_height * patch_width * components
    values = tl.load(
        patches + n * patch_stride_0 + patch_y * patch_stride_1
        + patch_x * patch_stride_2 + component,
        mask, 0,
    )
    mask = mask & (y >= 0) & (y < height) & (x >= 0) & (x < width)
    tl.atomic_add(
        output + (y * width + x) * components + component, values, mask, sem="relaxed"
    )


def accumulate_patches(image, patches, sy, sx, deterministic):
    """Accumulate into private storage so failed launches can be retried natively."""
    components = 2 if image.is_complex() else 1
    image_view = torch.view_as_real(image) if image.is_complex() else image
    patches_view = torch.view_as_real(patches) if patches.is_complex() else patches
    output = torch.empty_like(image) if deterministic else image.clone()
    output_view = torch.view_as_real(output) if output.is_complex() else output
    height, width = image.shape
    patch_height, patch_width = patches.shape[-2:]
    strides = tuple(stride * components for stride in patches.stride())
    block_size = 256
    if deterministic:
        grid = (triton.cdiv(width * components, block_size), height)
        _accumulate_deterministic[grid](
            image_view, patches_view, sy, sx, output_view, width, len(sy),
            patch_height, patch_width, components, *strides,
            sy.stride(0), sx.stride(0), block_size,
        )
    else:
        grid = (triton.cdiv(patch_height * patch_width * components, block_size), len(sy))
        _accumulate_atomic[grid](
            patches_view, sy, sx, output_view, height, width,
            patch_height, patch_width, components, *strides,
            sy.stride(0), sx.stride(0), block_size,
        )
    return output
