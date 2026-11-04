# Copyright © 2025 UChicago Argonne, LLC All right reserved
# Full license accessible at https://github.com//AdvancedPhotonSource/pty-chi/blob/main/LICENSE

from typing import Tuple, Literal, Optional, NamedTuple
import math
import logging

import torch
from torch import Tensor
import torch.signal

import ptychi.maths as pmath
from ptychi.triton.patches import try_accumulate_patches
from ptychi.api.enums import PatchInterpolationMethods
import ptychi.interpolation as interpolation
from ptychi.device import AcceleratorModuleWrapper
from ptychi.timing.timer_utils import timer, InlineTimer

logger = logging.getLogger(__name__)

_PATCH_OFFSET_CACHE: dict[tuple[torch.device, int, int, int], Tensor] = {}
_PATCH_CENTER_CACHE: dict[tuple[torch.device, torch.dtype, int, int], Tensor] = {}


def _patch_linear_indices(
    sy: Tensor, sx: Tensor, patch_size: Tuple[int, int], image_width: int
) -> Tensor:
    """Build linear indices for extracting or placing a batch of 2D patches."""
    patch_h, patch_w = patch_size
    sy = sy.to(torch.long)
    sx = sx.to(torch.long)
    cache_key = (sy.device, patch_h, patch_w, image_width)
    patch_offsets = _PATCH_OFFSET_CACHE.get(cache_key)
    if patch_offsets is None:
        y_offsets = torch.arange(patch_h, device=sy.device, dtype=torch.long) * image_width
        x_offsets = torch.arange(patch_w, device=sx.device, dtype=torch.long)
        patch_offsets = (y_offsets[:, None] + x_offsets[None, :]).reshape(-1)
        _PATCH_OFFSET_CACHE[cache_key] = patch_offsets
    start_offsets = sy * image_width + sx
    return start_offsets[:, None] + patch_offsets[None, :]


@timer()
def batch_slice(
    image: Tensor, sy: Tensor, sx: Tensor, patch_size: Tuple[int, int], check_bounds: bool = True
) -> Tensor:
    """
    Slice patches from an image at given window positions. The patch size is determined
    from the starting and ending coordinates in each direction, and is assumed to be
    the same for all patches.
    
    Parameters
    ----------
    image : Tensor
        A (H, W) tensor of the image.
    sy : Tensor
        A (N,) tensor of integers giving the starting y-coordinates of the patches.
    sx : Tensor
        A (N,) tensor of integers giving the starting x-coordinates of the patches.
    patch_size : tuple of int
        A tuple giving the patch shape in pixels.

    Returns
    -------
    Tensor
        A tensor of shape (N, h, w) containing the extracted patches.
    """
    h, w = image.shape[-2:]
    if check_bounds and (
        sy.min() < 0 
        or sy.max() + patch_size[0] > image.shape[-2] 
        or sx.min() < 0 
        or sx.max() + patch_size[1] > image.shape[-1]
    ):
        raise ValueError("Patch indices are out of bounds.")
    
    if torch.is_grad_enabled() and image.requires_grad:
        # Indexing an unfolded view would allocate its full shape during backward.
        inds = _patch_linear_indices(sy, sx, patch_size, w)
        patches = image.reshape(-1)[inds.reshape(-1)]
        patches = patches.reshape(len(sy), patch_size[0], patch_size[1])
    else:
        windows = image.unfold(0, patch_size[0], 1).unfold(1, patch_size[1], 1)
        patches = windows[sy.long(), sx.long()]
    if not check_bounds and patches.is_cuda:
        AcceleratorModuleWrapper.get_module().synchronize()
    return patches
    

@timer()
def batch_put(
    image: Tensor, 
    patches: Tensor, 
    sy: Tensor, 
    sx: Tensor, 
    op: Literal["add", "set"] = "add",
    check_bounds: bool = True,
) -> Tensor:
    """
    Slice patches from an image at given window positions. The patch size is assumed 
    to be the same for all patches.
    
    Parameters
    ----------
    image : Tensor
        A (H, W) tensor of the buffer to place the patches into.
    patches : Tensor
        A (N, h, w) tensor of the patches.
    sy : Tensor
        A (N,) tensor of integers giving the starting y-coordinates of the patches.
    sx : Tensor
        A (N,) tensor of integers giving the starting x-coordinates of the patches.
    op : Literal["add", "set"]
        The operation to perform. "add" adds the patches to the image, 
        "set" sets the patches to the image replacing the existing values.
    
    Returns
    -------
    Tensor
        A tensor of shape (H, W) containing the image with patches added or set.
    """
    h, w = image.shape[-2:]
    if check_bounds and (
        sy.min() < 0 
        or sy.max() + patches.shape[-2] > image.shape[-2] 
        or sx.min() < 0 
        or sx.max() + patches.shape[-1] > image.shape[-1]
    ):
        raise ValueError("Patch indices are out of bounds.")
    
    patch_size = patches.shape[-2:]
    if op == "add":
        result = try_accumulate_patches(
            image, patches, sy, sx, deterministic=True,
        )
        if result is not None:
            image.copy_(result)
            if not check_bounds:
                AcceleratorModuleWrapper.get_module().synchronize()
            return image
    inds = _patch_linear_indices(sy, sx, patch_size, w)
    image = image.reshape(-1)
    
    try:
        patches_flattened = patches.view(-1)
    except RuntimeError:
        patches_flattened = patches.reshape(-1)
        
    if pmath.get_allow_nondeterministic_algorithms():
        # scatter_add_ and scatter_ are non-deterministic but faster.
        if op == "add":
            image.scatter_add_(0, inds.reshape(-1), patches_flattened)
        else:
            image.scatter_(0, inds.reshape(-1), patches_flattened)
    else:
        image.index_put_((inds.reshape(-1),), patches_flattened, accumulate=(op == "add"))
    if not check_bounds and image.is_cuda:
        AcceleratorModuleWrapper.get_module().synchronize()
    return image.reshape(h, w)


@timer()
def extract_patches_integer(
    image: Tensor, positions: Tensor, shape: Tuple[int, int]
) -> Tensor:
    """
    Extract patches from 2D object, assuming the positions are at the
    exact center between pixels, so that no interpolation is needed. 
    If a patch's footprint goes outside the image,
    the image is padded with zeros to account for the missing pixels.

    Parameters
    ----------
    image : Tensor
        A (H, W) tensor giving the whole image.
    positions : Tensor
        A tensor of shape (N, 2) giving the center positions of the patches in pixels.
        The origin of the given positions are assumed to be the TOP LEFT corner of the image.
        For even patch shapes, the positions are assumed be at half pixels (i.e., positions % 1 == 0.5)
        so that the starting and ending coordinates of the patches are integers.
    shape : tuple of int
        A tuple giving the patch shape in pixels.

    Returns
    -------
    Tensor
        A tensor of shape (N, H, W) containing the extracted patches.
    """
    sys = (positions[:, 0] - (shape[0] - 1.0) / 2.0).round().int()
    sxs = (positions[:, 1] - (shape[1] - 1.0) / 2.0).round().int()
    
    pad_lengths = [
        max(-sxs.min(), 0),
        max(sxs.max() + shape[1] - image.shape[1], 0),
        max(-sys.min(), 0),
        max(sys.max() + shape[0] - image.shape[0], 0),
    ]
    if any(pad_lengths):
        logging.warning(
            f"Patch extractor has to pad the image by {[int(x) for x in pad_lengths]} "
            "(left, right, top, bottom) to avoid out-of-bounds errors. This should be "
            "avoided whenever possible. Consider using a larger object size."
        )
        image = torch.nn.functional.pad(image[None], pad_lengths, mode="replicate")[0]
        sys = sys + pad_lengths[2]
        sxs = sxs + pad_lengths[0]
    
    patches = batch_slice(image, sys, sxs, patch_size=shape, check_bounds=False)
    return patches


@timer()
def place_patches_integer(
    image: Tensor, 
    positions: Tensor, 
    patches: Tensor, 
    op: Literal["add", "set"] = "add"
) -> Tensor:
    """
    Place patches into a 2D object, assuming the positions are at the
    exact center between pixels, so that no interpolation is needed. 
    If a patch's footprint goes outside the image,
    the image is padded with zeros to account for the missing pixels.

    Parameters
    ----------
    image : Tensor
        The whole image to place the patches into.
    positions : Tensor
        A tensor of shape (N, 2) giving the center positions of the patches in pixels.
        The origin of the given positions are assumed to be the TOP LEFT corner of the image.
        For even patch shapes, the positions are assumed be at half pixels (i.e., positions % 1 == 0.5)
        so that the starting and ending coordinates of the patches are integers.
    patches : Tensor
        A (N, H, W) or (H, W) tensor of image patches.
    op : Literal["add", "set"]
        The operation to perform. "add" adds the patches to the image, 
        "set" sets the patches to the image replacing the existing values.

    Returns
    -------
    Tensor
        A tensor with the same shape as the object with patches added onto it.
    """
    shape = patches.shape[-2:]
    sys = (positions[:, 0] - (shape[0] - 1.0) / 2.0).round().int()
    sxs = (positions[:, 1] - (shape[1] - 1.0) / 2.0).round().int()
    
    pad_lengths = [
        max(-sxs.min(), 0),
        max(sxs.max() + shape[1] - image.shape[1], 0),
        max(-sys.min(), 0),
        max(sys.max() + shape[0] - image.shape[0], 0),
    ]
    if any(pad_lengths):
        logging.warning(
            f"Patch placer has to pad the image by {[int(x) for x in pad_lengths]} "
            "(left, right, top, bottom) to avoid out-of-bounds errors. This should be "
            "avoided whenever possible. Consider using a larger object size."
        )
        image = torch.nn.functional.pad(image, pad_lengths)
        sys = sys + pad_lengths[2]
        sxs = sxs + pad_lengths[0]
    
    image = batch_put(image, patches, sys, sxs, op=op, check_bounds=False)
    if any(pad_lengths):
        image = image[
            pad_lengths[2] : image.shape[0] - pad_lengths[3],
            pad_lengths[0] : image.shape[1] - pad_lengths[1],
        ]
    return image


def resolve_shift_operator(requested, distance, object_shape, probe_shape):
    """Resolve sampling using dimensions represented on the same pixel grid."""
    if requested == "auto":
        return (
            "full_object"
            if math.isfinite(distance)
            and all(5 * p > 4 * o for p, o in zip(probe_shape, object_shape))
            else "patch"
        )
    if requested not in ("patch", "full_object"):
        raise ValueError(f"Invalid shift operator: {requested}")
    return requested


def _fixed_patch_geometry(object_shape, patch_shape):
    overlap = tuple(min(o, p) for o, p in zip(object_shape, patch_shape))
    source_start = tuple(o // 2 - n // 2 for o, n in zip(object_shape, overlap))
    target_start = tuple(p // 2 - n // 2 for p, n in zip(patch_shape, overlap))
    source = tuple(slice(a, a + n) for a, n in zip(source_start, overlap))
    target = tuple(slice(b, b + n) for b, n in zip(target_start, overlap))
    center = tuple(a - b + (p - 1) / 2 for a, b, p in zip(source_start, target_start, patch_shape))
    return source, target, center


def _fixed_patch_resize(image, shape):
    source, target, _ = _fixed_patch_geometry(image.shape[-2:], shape)
    cropped = image[..., source[0], source[1]]
    if tuple(cropped.shape[-2:]) == tuple(shape):
        return cropped
    return torch.nn.functional.pad(cropped, (
        target[1].start, shape[1] - target[1].stop,
        target[0].start, shape[0] - target[0].stop,
    ))


@timer()
def extract_patches_full_object(
    image: Tensor, positions: Tensor, shape: Tuple[int, int],
    method: str = "fourier", *, derivative: Tuple[int, int] = (0, 0),
) -> Tensor:
    r"""Periodically shift a complete 2D image, then center crop/pad to patch shape.

    Positions are absolute patch centers in (y, x) pixels. Derivative orders
    refer to these centers, and are applied before the fixed crop. Pixels
    outside the image-sized shifted support are zero when the patch is larger.

    Let :math:`\widehat O_{kl}` be the image FFT, :math:`N_y,N_x` its shape,
    and :math:`M_y,M_x` the factors from ``interpolation.translation_multiplier``.
    For ``derivative=(m, n)``, compute the complete periodic field

    .. math::

        Z_q(r_y,r_x)=\frac{1}{N_yN_x}\sum_{k,l}\widehat O_{kl}
            (\partial_{q_y}^m M_y)(\partial_{q_x}^n M_x)
            \exp\left[2\pi i(f_{y,k}r_y+f_{x,l}r_x)\right].

    The returned patch is :math:`Z_q(u+a_y-b_y,v+a_x-b_x)` on the
    fixed overlap, where :math:`a,b` are source/target crop starts.
    Real inputs return the real part. The image FFT is shared by all positions.
    """
    positions = positions.to(device=image.device, dtype=image.real.dtype)
    spectrum = pmath.fft2_precise(image)
    _, _, center = _fixed_patch_geometry(image.shape, shape)
    shifts = positions.new_tensor(center) - positions
    multiplier = interpolation.translation_multiplier(shifts, image.shape, method, derivative)
    if sum(derivative) % 2:
        multiplier = -multiplier
    shifted = pmath.ifft2_precise(spectrum[None] * multiplier)
    if not image.is_complex():
        shifted = shifted.real
    return _fixed_patch_resize(shifted, shape)


@timer()
def place_patches_full_object(
    image: Tensor, positions: Tensor, patches: Tensor,
    op: Literal["add", "set"] = "add", method: str = "fourier",
) -> Tensor:
    r"""Add the exact adjoint of full-object patch extraction to an image.

    Undo the fixed crop/pad for each patch :math:`Y_i`: copy its retained
    overlap into a zero object-sized array :math:`E_i`, discarding padded
    margins. With :math:`M_i` the two-axis interpolation multiplier, add

    .. math::

        \Delta O(r_y,r_x)=\frac{1}{N_yN_x}\sum_{k,l}
            \left[\sum_i\overline{M_i(k,l)}\,\widehat E_i(k,l)\right]
            \exp\left[2\pi i(f_{y,k}r_y+f_{x,l}r_x)\right].

    Contributions are summed in Fourier space before the inverse FFT.
    Conjugating the multiplier gives the adjoint periodic translation;
    the fixed crop/pad is reversed separately. Real image buffers receive
    the real part. A 2D patch is shared across all positions;
    batched patches have shape ``(batch_size, patch_height, patch_width)``.
    Only additive placement is supported.
    """
    if op != "add":
        raise ValueError("Full-object placement supports only additive placement.")
    if patches.ndim == 2:
        patches = patches[None].expand(len(positions), -1, -1)
    positions = positions.to(device=patches.device, dtype=patches.real.dtype)
    expanded = _fixed_patch_resize(patches, image.shape)
    _, _, center = _fixed_patch_geometry(image.shape, patches.shape[-2:])
    shifts = positions.new_tensor(center) - positions
    multiplier = interpolation.translation_multiplier(shifts, image.shape, method)
    spectrum = (pmath.fft2_precise(expanded) * multiplier.conj()).sum(0)
    update = pmath.ifft2_precise(spectrum)
    if not image.is_complex():
        update = update.real
    return image + update


class ObjectPatchSampler:
    """Own support selection, translation and accumulation on one pixel grid.

    Positions are absolute patch centers in (y, x) pixels. Local Fourier uses
    a symmetric halo and periodic translation within that support. Bilinear
    uses one extra row/column and requires its support to lie inside the image.
    Local nearest retains center rounding followed by bilinear evaluation.
    Full-object sampling translates the complete periodic object before cropping.
    """

    def __init__(self, operator="patch", method: PatchInterpolationMethods | str = "fourier", pad=0):
        self.operator = operator
        self.method = method
        self.pad = pad or 0

    def integer_mode(self, requested):
        return requested and self.operator == "patch"

    def _local_geometry(self, positions, shape, adjoint_mode=True):
        if self.method == "nearest":
            positions = positions.round().to(int)
        elif self.method not in ("fourier", "bilinear"):
            raise ValueError(f"Invalid interpolation method: {self.method}")
        if self.method != "fourier":
            dtype = positions.dtype if positions.is_floating_point() else torch.get_default_dtype()
            key = (positions.device, dtype, *shape)
            center = _PATCH_CENTER_CACHE.get(key)
            if center is None:
                center = positions.new_tensor([(shape[0] - 1) / 2, (shape[1] - 1) / 2], dtype=dtype)
                _PATCH_CENTER_CACHE[key] = center
            starts = positions - center
            integer_starts = starts.int()
            shifts = integer_starts - starts
            sy, sx = integer_starts.unbind(-1)
            return sy, sx, shifts, (shape[0] + 1, shape[1] + 1)
        y = positions[:, 0] - (shape[0] - 1.0) / 2
        x = positions[:, 1] - (shape[1] - 1.0) / 2
        margin = self.pad if adjoint_mode else -self.pad
        sy, sx = y.floor().int() - margin, x.floor().int() - margin
        fractions = torch.stack((y - sy - margin, x - sx - margin), -1)
        extra = 2 * margin
        return sy, sx, -fractions, (shape[0] + extra, shape[1] + extra)

    @staticmethod
    def _pad_image(image, sy, sx, shape):
        padding = [
            max(-sx.min(), 0), max(sx.max() + shape[1] - image.shape[1], 0),
            max(-sy.min(), 0), max(sy.max() + shape[0] - image.shape[0], 0),
        ]
        return (torch.nn.functional.pad(image, padding),
                sy + padding[2], sx + padding[0], padding)

    def extract(self, image, positions, shape, integer_mode=False):
        """Extract (N, h, w) patches from an (H, W) image.

        ``positions`` gives absolute (y, x) centers in pixels, and ``shape``
        is (h, w). The first sample is at ``center - (size - 1) / 2`` on each
        axis. Local Fourier sampling gathers a halo of ``self.pad`` pixels,
        translates it, and crops the halo; missing object pixels are zero.
        Local bilinear support must fit inside the image, including its extra
        bottom/right pixel. Local nearest rounds centers before interpolation.

        For local sampling, ``integer_mode=True`` selects rounded integer
        windows without interpolation, replicating image edges when needed.
        Full-object sampling always translates the periodic object and then
        applies the fixed center crop/pad.
        """
        if self.operator == "full_object":
            return extract_patches_full_object(image, positions, shape, self.method)
        if integer_mode:
            return extract_patches_integer(image, positions, shape)
        sy, sx, shifts, support_shape = self._local_geometry(positions, shape)
        if self.method == "fourier":
            image, sy, sx, _ = self._pad_image(image, sy, sx, support_shape)
        if self.method != "fourier" and len(positions) == 1:
            y, x = sy[0], sx[0]
            supports = image[y:y + support_shape[0], x:x + support_shape[1]][None]
        else:
            supports = batch_slice(image, sy, sx, support_shape, check_bounds=False)
        if self.method == "fourier":
            if not torch.allclose(shifts, torch.zeros_like(shifts), atol=1e-7):
                supports = interpolation.translate(supports, shifts, self.method)
            return supports[:, self.pad:supports.shape[-2] - self.pad,
                            self.pad:supports.shape[-1] - self.pad]
        return interpolation.translate(supports, shifts, "bilinear", valid_shape=shape)

    def place(self, image, positions, patches, integer_mode=False, *,
              op="add", adjoint_mode=True):
        """Place patches at absolute (y, x) centers and return the updated image.

        ``image`` is (H, W), ``positions`` is (N, 2), and ``patches`` is
        (N, h, w) or a shared (h, w) patch. Default interpolated placement
        accumulates the extraction adjoint with the same forward centers.
        Local integer mode uses rounded windows and discards contributions
        outside the image; full-object placement uses periodic translation.

        Local Fourier placement zero-pads patches by ``self.pad`` before
        adjoint translation. With ``adjoint_mode=False``, it instead translates
        unpadded patches and crops their borders to discard wrap-around values.
        This mode serves ordinary placement rather than gradient accumulation.
        ``op="set"`` replaces pixels for local Fourier or integer placement.
        Bilinear, nearest and full-object placement support addition.
        ``adjoint_mode`` affects local Fourier sampling.
        """
        if self.operator == "full_object":
            return place_patches_full_object(image, positions, patches, op=op, method=self.method)
        if integer_mode:
            return place_patches_integer(image, positions, patches, op=op)
        if self.method != "fourier" and op == "set":
            raise NotImplementedError('"set" operation is not supported.')
        if patches.ndim == 2:
            patches = patches[None].expand(len(positions), -1, -1)
        shape = patches.shape[-2:]
        sy, sx, shifts, support_shape = self._local_geometry(positions, shape, adjoint_mode)
        if self.method == "fourier":
            if adjoint_mode:
                patches = torch.nn.functional.pad(patches, [self.pad] * 4)
            image, sy, sx, padding = self._pad_image(image, sy, sx, support_shape)
            if not torch.allclose(shifts, torch.zeros_like(shifts), atol=1e-7):
                patches = interpolation.translate(patches, shifts, self.method, adjoint=True)
            if not adjoint_mode:
                patches = patches[:, self.pad:patches.shape[-2] - self.pad,
                                  self.pad:patches.shape[-1] - self.pad]
        else:
            if len(positions) == 1:
                y, x = sy[0], sx[0]
                support = image[y:y + support_shape[0], x:x + support_shape[1]]
                interpolation.translate(
                    patches, shifts, "bilinear", adjoint=True,
                    valid_shape=shape, out=support,
                )
                return image
            # Accumulate in the destination precision, retaining patch precision
            # for multiplication by interpolation weights.
            support = (image.new_zeros((len(positions), *support_shape))
                       if patches.dtype != image.dtype else None)
            patches = interpolation.translate(
                patches, shifts, "bilinear", adjoint=True, valid_shape=shape, out=support
            )
        inline_timer = InlineTimer("add or set patches on image")
        inline_timer.start()
        result = batch_put(image, patches, sy, sx, op=op, check_bounds=False)
        if self.method != "fourier" and not image.is_contiguous():
            image.copy_(result)
        else:
            image = result
        inline_timer.end()
        if self.method == "fourier":
            image = image[padding[2]:image.shape[0] - padding[3],
                          padding[0]:image.shape[1] - padding[1]]
        return image


class ObjectSamplingParameters(NamedTuple):
    """Sampler and its geometry on one pixel grid.

    ``positions`` contains continuous absolute centers in (y, x) pixels;
    ``shape`` is the complete object shape on that same sampling grid.
    """

    sampler: ObjectPatchSampler
    positions: Tensor
    shape: Tuple[int, int]


@timer()
def shift_images(
    images: Tensor, 
    shifts: Tensor, 
    method: Literal["bilinear", "fourier", "nearest"] = "bilinear",
    adjoint: bool = False,
    pad: Optional[int] = 0,
) -> Tensor:
    """Shift a batch of images by a given amount.
    
    Parameters
    ----------
    images : Tensor
        A [N, H, W] tensor of images.
    shifts : Tensor
        A [N, 2] tensor of shifts in pixels. Use the same shift values as the
        forward shift operation even if `adjoint` is True; do not flip the sign.
    method : Literal["bilinear", "fourier", "nearest"]
        The method to use for shifting.
    adjoint : bool
        If True, the adjoint of the shift operation is performed. For Fourier
        shift, it shifts the image by `-shifts`. For bilinear shift, it computes
        the adjoint of the bilinear shift operation.
    pad : Optional[int]
        Positive values replicate-pad the image before shifting and crop the
        result. The adjoint zero-pads before shifting and accumulates the halo
        onto the boundary pixels afterward.
    
    Returns
    -------
    Tensor
        Shifted images.
    """
    if pad is not None and pad > 0:
        mode = "constant" if adjoint else "replicate"
        images = torch.nn.functional.pad(images, (pad, pad, pad, pad), mode=mode)
    images = interpolation.translate(images, shifts, method, adjoint=adjoint)
    if pad is not None and pad > 0:
        if adjoint:
            # Fold columns first, then rows, so corner contributions are included.
            columns = images[..., pad:-pad].clone()
            columns[..., 0] += images[..., :pad].sum(dim=-1)
            columns[..., -1] += images[..., -pad:].sum(dim=-1)
            images = columns[..., pad:-pad, :].clone()
            images[..., 0, :] += columns[..., :pad, :].sum(dim=-2)
            images[..., -1, :] += columns[..., -pad:, :].sum(dim=-2)
        else:
            images = images[..., pad:-pad, pad:-pad]
    return images


@timer()
def fourier_shift(images: Tensor, shifts: Tensor, strictly_preserve_zeros: bool = False) -> Tensor:
    """
    Apply Fourier shift to a batch of images.

    Parameters
    ----------
    images : Tensor
        A [N, H, W] tensor of images.
    shifts : Tensor
        A [N, 2] tensor of shifts in pixels.
    strictly_preserve_zeros : bool
        If True, mask of strictly zero pixels will be generated and shifted
        by the same amount. Pixels that have a non-zero value in the shifted
        mask will be set to zero in the shifted image. This preserves the zero
        pixels in the original image, preventing FFT from introducing small
        non-zero values due to machine precision.

    Returns
    -------
    Tensor
        Shifted images.
    """
    if strictly_preserve_zeros:
        zero_mask = images == 0
        zero_mask = zero_mask.float()
        zero_mask_shifted = fourier_shift(zero_mask, shifts, strictly_preserve_zeros=False)
    shifted_images = interpolation.translate(images, shifts, "fourier")
    if strictly_preserve_zeros:
        shifted_images[zero_mask_shifted > 0] = 0
    return shifted_images


@timer()
def nearest_neighbor_gradient(
    image: Tensor, direction: Literal["forward", "backward"], dim: Tuple[int, ...] = (0, 1)
) -> Tuple[Tensor, Tensor]:
    """
    Calculate the nearest neighbor gradient of a 2D image.

    Parameters
    ----------
    image : Tensor
        a (... H, W) tensor of images.
    direction : str
        'forward' or 'backward'.
    dim : tuple of int, optional
        Dimensions to calculate gradient. Default is (0, 1).

    Returns
    -------
    Tuple[Tensor, Tensor]
        A tuple of 2 images with the gradient in y and x directions.
    """
    if not hasattr(dim, "__len__"):
        dim = (dim,)
    grad_x = None
    grad_y = None
    if direction == "forward":
        if 1 in dim:
            grad_x = torch.concat([image[:, 1:], image[:, -1:]], dim=1) - image
        if 0 in dim:
            grad_y = torch.concat([image[1:, :], image[-1:, :]], dim=0) - image
    elif direction == "backward":
        if 1 in dim:
            grad_x = image - torch.concat([image[:, :1], image[:, :-1]], dim=1)
        if 0 in dim:
            grad_y = image - torch.concat([image[:1, :], image[:-1, :]], dim=0)
    else:
        raise ValueError("direction must be 'forward' or 'backward'")
    return grad_y, grad_x


@timer()
def gaussian_gradient(image: Tensor, sigma: float = 1.0, kernel_size=5) -> Tuple[Tensor, Tensor]:
    """
    Calculate the gradient of a 2D image with a Gaussian-derivative kernel.

    Parameters
    ----------
    image : tensor
        A (... H, W) tensor of images.
    sigma : float
        Sigma of the Gaussian.

    Returns
    -------
    Tuple[Tensor, Tensor]
        A tuple of 2 images with the gradient in y and x directions.
    """
    r = torch.arange(kernel_size) - (kernel_size - 1) / 2.0
    kernel = -r / (math.sqrt(2 * math.pi) * sigma**3) * torch.exp(-(r**2) / (2 * sigma**2))
    grad_y = convolve2d(image, kernel.view(-1, 1), padding="same", padding_mode="replicate")
    grad_x = convolve2d(image, kernel.view(1, -1), padding="same", padding_mode="replicate")

    # Gate the gradients
    grads = [grad_y, grad_x]
    for i, g in enumerate(grads):
        m = torch.logical_and(g.abs() < 1e-6, g.abs() != 0)
        if torch.count_nonzero(m) > 0:
            logger.debug("Gradient magnitudes between 0 and 1e-6 are set to 0.")
            g = g * torch.logical_not(m)
            grads[i] = g
    grad_y, grad_x = grads
    return grad_y, grad_x


@timer()
def fourier_gradient(image: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Calculate the gradient of an image using Fourier differentiation
    theorem: `fft(df/dx) = 2 * pi * i * u * fft(f)`.

    Parameters
    ----------
    image : Tensor
        A (... H, W) tensor of images.

    Returns
    -------
    Tuple[Tensor, Tensor]
        The y and x gradients.
    """
    u, v = torch.fft.fftfreq(image.shape[-2]), torch.fft.fftfreq(image.shape[-1])
    u = u.to(image.device)
    v = v.to(image.device)
    u, v = torch.meshgrid(u, v, indexing="ij")
    grad_y = torch.fft.ifft(torch.fft.fft(image, dim=-2) * (2j * torch.pi) * u, dim=-2)
    grad_x = torch.fft.ifft(torch.fft.fft(image, dim=-1) * (2j * torch.pi) * v, dim=-1)
    return grad_y, grad_x


@timer()
def fourier_shift_gradient(image: Tensor, step: float = 0.5) -> Tuple[Tensor, Tensor]:
    """
    Calculate the gradient of an image using Fourier shift. 

    Parameters
    ----------
    image : Tensor
        A (... H, W) tensor of images.

    Returns
    -------
    Tuple[Tensor, Tensor]
        A tuple of (... H, W) tensor of gradients in y and x directions.
    """
    orig_shape = image.shape
    if image.ndim == 2:
        image = image.unsqueeze(0)
    
    pad = math.ceil(step) + 1
    image = torch.nn.functional.pad(image, (pad, pad, pad, pad), mode="replicate")
    
    sx1 = torch.tensor([[0, -step]], device=image.device).repeat(image.shape[0], 1)
    sx2 = torch.tensor([[0, step]], device=image.device).repeat(image.shape[0], 1)
    gx = fourier_shift(image, sx1) - fourier_shift(image, sx2)
    
    sy1 = torch.tensor([[-step, 0]], device=image.device).repeat(image.shape[0], 1)
    sy2 = torch.tensor([[step, 0]], device=image.device).repeat(image.shape[0], 1)
    gy = fourier_shift(image, sy1) - fourier_shift(image, sy2)
    
    if pad > 0:
        gy = gy[..., pad:-pad, pad:-pad]
        gx = gx[..., pad:-pad, pad:-pad]

    if len(orig_shape) == 2:
        gy = gy[0]
        gx = gx[0]
    return gy, gx


@timer()
def get_phase_gradient(
    img: Tensor,
    fourier_shift_step: float = 0,
    image_grad_method: Literal[
        "fourier_shift", "fourier_differentiation", "nearest"
    ] = "fourier_shift",
    eps: float = 1e-6,
) -> Tensor:
    """
    Get the gradient of the phase of a complex 2D image by first calculating
    the spatial gradient of the complex image, then taking the phase of the
    complex gradient -- i.e., it takes the phase of the gradient rather than
    the gradient of the phase. This avoids the sharp gradients due to phase
    wrapping when directly taking the gradient of the phase.

    Parameters
    ----------
    img : Tensor
        A [N, H, W] or [H, W] tensor giving a batch of images or a single image.
    step : float
        The finite-difference step size used to calculate the gradient, if
        the Fourier shift method is used.
    finite_diff_method : enums.ImageGradientMethods
        The method used to calculate the phase gradient.
            - "fourier_shift": Use Fourier shift to perform shift.
            - "nearest": Use nearest neighbor to perform shift.
            - "fourier_differentiation": Use Fourier differentiation.
    eps : float
        A stablizing constant.

    Returns
    -------
    Tuple[Tensor, Tensor]
        A tuple of 2 tensors with the gradient in y and x directions.
    """
    if fourier_shift_step <= 0 and image_grad_method == "fourier_shift":
        raise ValueError("Step must be positive.")

    if image_grad_method == "fourier_differentiation":
        gy, gx = fourier_gradient(img)
        gy = torch.imag(img.conj() * gy)
        gx = torch.imag(img.conj() * gx)
    else:
        # Use finite difference.
        if img.ndim == 2:
            img = img.unsqueeze(0)
        pad = int(math.ceil(fourier_shift_step)) + 1
        img = torch.nn.functional.pad(img, (pad, pad, pad, pad))

        sy1 = torch.tensor([[-fourier_shift_step, 0]], device=img.device).repeat(img.shape[0], 1)
        sy2 = torch.tensor([[fourier_shift_step, 0]], device=img.device).repeat(img.shape[0], 1)
        if image_grad_method == "fourier_shift":
            # If the image contains zero-valued pixels, Fourier shift can result in small
            # non-zero values that dangles around 0. This can cause the phase
            # of the shifted image to dangle between pi and -pi. In that case, use
            # `finite_diff_method="nearest" instead`, or use `step=1`.
            complex_prod = fourier_shift(img, sy1) * fourier_shift(img, sy2).conj()
        elif image_grad_method == "nearest":
            complex_prod = img * torch.concat([img[:, :1, :], img[:, :-1, :]], dim=1).conj()
        else:
            raise ValueError(f"Unknown finite-difference method: {image_grad_method}")
        complex_prod = torch.where(
            complex_prod.abs() < complex_prod.abs().max() * 1e-6, 0, complex_prod
        )
        gy = pmath.angle(complex_prod, eps=eps) / (2 * fourier_shift_step)
        gy = gy[0, pad:-pad, pad:-pad]

        sx1 = torch.tensor([[0, -fourier_shift_step]], device=img.device).repeat(img.shape[0], 1)
        sx2 = torch.tensor([[0, fourier_shift_step]], device=img.device).repeat(img.shape[0], 1)
        if image_grad_method == "fourier_shift":
            complex_prod = fourier_shift(img, sx1) * fourier_shift(img, sx2).conj()
        elif image_grad_method == "nearest":
            complex_prod = img * torch.concat([img[:, :, :1], img[:, :, :-1]], dim=2).conj()
        complex_prod = torch.where(
            complex_prod.abs() < complex_prod.abs().max() * 1e-6, 0, complex_prod
        )
        gx = pmath.angle(complex_prod, eps=eps) / (2 * fourier_shift_step)
        gx = gx[0, pad:-pad, pad:-pad]
    return gy, gx


@timer()
def integrate_image_2d_fourier(grad_y: Tensor, grad_x: Tensor) -> Tensor:
    """
    Integrate an image with the gradient in y and x directions using Fourier
    differentiation.

    Parameters
    ----------
    grad_y, grad_x: Tensor
        A (H, W) tensor of gradients in y or x directions.

    Returns
    -------
    Tensor
        The integrated image.
    """
    shape = grad_y.shape
    f = pmath.fft2_precise(grad_x + 1j * grad_y)
    y, x = torch.fft.fftfreq(shape[0]), torch.fft.fftfreq(shape[1])
    y = y.to(grad_y.device)
    x = x.to(grad_y.device)

    # In PtychoShelves' get_img_int_2D.m, they set the numerator of r to be
    # exp(2j * pi * (x + y[:, None])) to shift it by 1 pixel. We should NOT
    # do this in order to get the same result as PtychoShelves.
    r = 1.0
    r = r / (2j * torch.pi * (x + 1j * y[:, None]))
    r[0, 0] = 0
    integrated_image = f * r
    integrated_image = pmath.ifft2_precise(integrated_image)
    if not torch.is_complex(grad_x):
        integrated_image = integrated_image.real
    return integrated_image


@timer()
def integrate_image_2d_deconvolution(
    grad_y: Tensor,
    grad_x: Tensor,
    tf_y: Optional[Tensor] = None,
    tf_x: Optional[Tensor] = None,
    bc_center: float = 0,
) -> Tensor:
    """
    Integrate an image with the gradient in y and x directions by deconvolving
    the differentiation kernel, whose transfer function is assumed to be a
    ramp function.
    
    Adapted from Tripathi, A., McNulty, I., Munson, T., & Wild, S. M. (2016). 
    Single-view phase retrieval of an extended sample by exploiting edge detection 
    and sparsity. Optics Express, 24(21), 24719–24738. doi:10.1364/OE.24.024719

    Parameters
    ----------
    grad_y, grad_x: Tensor
        A (H, W) tensor of gradients in y or x directions.
    tf_y, tf_x: Tensor
        A (H, W) tensor of transfer functions in y or x directions. If not
        provided, they are assumed to be 2i * pi * u (or v), which are the
        effective transfer functions in Fourier differentiation.
    bc_center: float
        The value of the boundary condition at the center of the image.

    Returns
    -------
    Tensor
        The integrated image.
    """
    u, v = torch.fft.fftfreq(grad_x.shape[0]), torch.fft.fftfreq(grad_x.shape[1])
    u, v = torch.meshgrid(u, v, indexing="ij")
    if tf_y is None or tf_x is None:
        tf_y = 2j * torch.pi * u
        tf_x = 2j * torch.pi * v
    f_grad_y = pmath.fft2_precise(grad_y)
    f_grad_x = pmath.fft2_precise(grad_x)
    img = (f_grad_y * tf_y + f_grad_x * tf_x) / (tf_y.abs() ** 2 + tf_x.abs() ** 2 + 1e-5)
    img = -pmath.ifft2_precise(img)
    img = img + bc_center - img[img.shape[0] // 2, img.shape[1] // 2]
    return img


@timer()
def integrate_image_2d(grad_y: Tensor, grad_x: Tensor, bc_center: float = 0) -> Tensor:
    """
    Integrate an image with the gradient in y and x directions.

    Parameters
    ----------
    grad_y : Tensor
        The gradient in y direction.
    grad_x : Tensor
        The gradient in x direction.
    bc_center : float
        The boundary condition at the center of the image, by default 0

    Returns
    -------
    Tensor
        The integrated image.
    """
    left_boundary = torch.cumsum(grad_y[:, 0], dim=0)
    int_img = torch.cumsum(grad_x, dim=1) + left_boundary[:, None]
    int_img = int_img + bc_center - int_img[int_img.shape[0] // 2, int_img.shape[1] // 2]
    return int_img


@timer()
def convolve2d(
    image: Tensor,
    kernel: Tensor,
    padding: Literal["same", "valid"] = "same",
    padding_mode: Literal["replicate", "constant"] = "replicate",
) -> Tensor:
    """
    2D convolution with an explicitly given kernel using torch.nn.functional.conv2d.

    This routine flips the kernel to adhere with the textbook definition of convolution.
    torch.nn.functional.conv2d does not flip the kernel in itself.

    Parameters
    ----------
    image : Tensor
        A (... H, W) tensor of images. If the number of dimensions is greater than 2,
        the last two dimensions are interpreted as height and width, respectively.
    kernel : Tensor
        A (H, W) tensor of kernel.

    Returns
    -------
    Tensor
        A (... H, W) tensor of convolved images.
    """
    if not image.ndim >= 2:
        raise ValueError("Image must have at least 2 dimensions.")
    if not kernel.ndim == 2:
        raise ValueError("Kernel must have exactly 2 dimensions.")
    if not (kernel.shape[-2] % 2 == 1 and kernel.shape[-1] % 2 == 1):
        raise ValueError("Kernel dimensions must be odd.")

    if image.dtype.is_complex:
        kernel = kernel.type(image.dtype)

    # Reshape image to (N, 1, H, W).
    orig_shape = image.shape
    image = image.reshape(-1, 1, image.shape[-2], image.shape[-1])

    # Reshape kernel to (1, 1, H, W).
    kernel = kernel.flip((0, 1))
    kernel = kernel.reshape(1, 1, kernel.shape[-2], kernel.shape[-1])

    if padding == "same":
        pad_lengths = [
            kernel.shape[-1] // 2,
            kernel.shape[-1] // 2,
            kernel.shape[-2] // 2,
            kernel.shape[-2] // 2,
        ]
        image = torch.nn.functional.pad(image, pad_lengths, mode=padding_mode)

    result = torch.nn.functional.conv2d(image, kernel, padding="valid")
    result = result.reshape(*orig_shape[:-2], result.shape[-2], result.shape[-1])
    return result


@timer()
def convolve1d(
    input: Tensor,
    kernel: Tensor,
    padding: Literal["same", "valid"] = "same",
    padding_mode: Literal["replicate", "constant"] = "replicate",
    dim: int = -1,
) -> Tensor:
    """
    1D convolution with an explicitly given kernel using torch.nn.functional.conv1d.

    This routine flips the kernel to adhere with the textbook definition of convolution.
    torch.nn.functional.conv1d does not flip the kernel in itself.

    Parameters
    ----------
    image : Tensor
        A (... d) tensor of signals.
    kernel : Tensor
        A (d,) tensor of kernel.

    Returns
    -------
    Tensor
        A (... d) tensor of convolved signals.
    """
    if not input.ndim >= 1:
        raise ValueError("Image must have at least 1 dimensions.")
    if not kernel.ndim == 1:
        raise ValueError("Kernel must have exactly 1 dimensions.")

    if input.dtype.is_complex:
        kernel = kernel.type(input.dtype)

    dim = dim % input.ndim
    orig_shape = input.shape
    # Move dim to the end.
    if dim != input.ndim - 1:
        input = torch.moveaxis(input, dim, input.ndim - 1)
    bcast_shape = input.shape[:-1]
    # Reshape image to (N, 1, d).
    if input.ndim == 1:
        input = input.reshape(1, 1, input.shape[-1])
    else:
        input = input.reshape(-1, 1, input.shape[-1])

    # Reshape kernel to (1, 1, d).
    kernel = kernel.flip((0,))
    kernel = kernel.reshape(1, 1, kernel.shape[-1])

    if padding == "same":
        pad_lengths = [
            kernel.shape[-1] // 2,
            kernel.shape[-1] // 2,
        ]
        if kernel.shape[-1] % 2 == 0:
            pad_lengths[-1] -= 1
        input = torch.nn.functional.pad(input, pad_lengths, mode=padding_mode)

    result = torch.nn.functional.conv1d(input, kernel, padding="valid")

    # Restore shape.
    if len(orig_shape) == 1:
        result = result.reshape(orig_shape[0])
    else:
        result = result.reshape([*bcast_shape, result.shape[-1]])
        if dim != input.ndim - 1:
            result = torch.moveaxis(result, result.ndim - 1, dim)
    return result


@timer()
def gaussian_filter(image, sigma=1, size=3):
    x = torch.arange(-((size - 1) / 2), -((size - 1) / 2) + size, 1, device=image.device)
    gauss_1d = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel = gauss_1d[:, None] * gauss_1d[None, :]
    kernel = kernel / torch.sum(kernel)
    return convolve2d(image, kernel, padding="same")


@timer()
def find_cross_corr_peak(
    f: Tensor,
    g: Tensor,
    scale: int,
    initial_guess=[0, 0],
    real_space_width: float = 3,
    dtype=torch.complex128,
):
    """
    Find the cross-correlation peak of two 2D arrays with arbitarily high precision.

    This implementation is based on:
    - Efficient subpixel image registration algorithms (2008) - Manuel Guizar-Sicairos

    The matrix multiplication inverse FT calculation was inferred from formula (12) in the paper:
    - Fast computation of Lyot-style coronagraph propagation (2007) - R. Soummer

    Note: position correction does not work properly when dtype is complex64 instead of complex128.
    The position correction will "walk off" if complex64 is used.

    Parameters
    ----------
    f : Tensor
        a (h, w) tensor.
    g : Tensor
        a (h, w) tensor.
    scale : int
        an integer specifying how much to upsample the cross-correlation.
    initial_guess : list[int]
        an initial guess of the location of the cross-correlation peak.
    real_space_width : float
        a number specifying the width of the cross-correlation (in units of non-upsampled pixels).

    Returns
    -------
    Tensor
        a tensor of the location of the cross-correlation peak.
    """
    f = torch.fft.ifftshift(f)
    g = torch.fft.ifftshift(g)

    dtype_real = torch.tensor(1 + 1j, dtype=dtype).real.dtype

    M, N = f.shape
    u = torch.fft.fftfreq(M, 1, dtype=dtype_real)[:, None]
    v = torch.fft.fftfreq(N, 1, dtype=dtype_real)[:, None]

    F = torch.fft.fft2(f.to(dtype))
    G = torch.fft.fft2(g.to(dtype))

    FG = F * G.conj()

    num_pts = int(real_space_width * scale / 2) * 2 + 1
    span = torch.linspace(-real_space_width / 2, real_space_width / 2, num_pts, dtype=dtype_real)

    x = initial_guess[0] + span[:, None]
    y = initial_guess[1] + span[:, None]

    # Do inverse FFT using matrix multiplication
    y_exp = torch.exp(2j * torch.pi * torch.matmul(v, y.transpose(1, 0))).to(dtype)
    tmp = torch.matmul(FG, y_exp)
    x_exp = torch.exp(2j * torch.pi * torch.matmul(x, u.transpose(1, 0))).to(dtype)
    corr = torch.matmul(x_exp, tmp)

    max_idx = torch.tensor(torch.unravel_index(torch.argmax(corr.abs()), corr.shape))
    est_shift = torch.tensor([x[max_idx[0]], y[max_idx[1]]])

    return est_shift


@timer()
def total_variation_2d_chambolle(
    image: Tensor, lmbda: float = 0.01, niter: int = 2, tau: float = 0.125
) -> Tensor:
    """
    Apply total variation constraint to an image using Chambolle's total variation projection method
    (https://doi.org/10.5201/ipol.2013.61).

    Parameters
    ----------
    image : Tensor
        A (H, W) tensor.
    lmbda : float
        The trade-off parameter (1 / lambda in the paper).
        Larger lmbda means stronger smoothing.
    niter : int
        An integer specifying the number of iterations.
    tau : float
        The time-step parameter.

    Returns
    -------
    Tensor
        A (H, W) tensor.
    """

    def div(p):
        py = p[..., 0]
        px = p[..., 1]
        fy = nearest_neighbor_gradient(py, "backward", dim=0)[0]
        fx = nearest_neighbor_gradient(px, "backward", dim=1)[1]
        fd = fx + fy
        return fd

    if image.ndim != 2:
        raise ValueError("Input tensor must be two dimensional")

    x0 = image
    xi = torch.zeros(list(image.shape) + [2], dtype=image.dtype, device=image.device)

    for _ in range(niter):
        # Chambolle step
        grad_y, grad_x = nearest_neighbor_gradient(div(xi) - image / lmbda, "forward")
        gdv = torch.stack([grad_y, grad_x], dim=-1)

        # Isotropic
        d = torch.sqrt(torch.sum(gdv**2, dim=-1, keepdim=True))

        xi = (xi + tau * gdv) / (1 + tau * d)

        # Reconstruct
        image = image - lmbda * div(xi)

    # Prevent pushing values to 0
    image = torch.sum(x0 * image) / torch.sum(image**2) * image
    return image


@timer()
def remove_grid_artifacts(
    img: Tensor,
    pixel_size_m: float,
    period_x_m: float,
    period_y_m: float,
    window_size: int,
    direction: Literal["x", "y", "xy"] = "xy",
):
    """
    Remove grid artifacts by setting the region near each harmonic peak
    corresponding to the set periodicity of the artifacts to 0 in the input
    image's Fourier domain.

    Adapted from fold_slice (remove_grid_artifact.m).

    Parameters
    ----------
    img : Tensor
        The input image.
    pixel_size_m : float
        The pixel size in meter.
    step_size_x : float
        The period of the artifacts in the x direction in meter.
    step_size_y : float
        The period of the artifacts in the y direction in meter.
    window_size : int
        The radius of the region around each harmonic peak in which the values
        are to be set to 0, given in pixels.
    direction : {'x', 'y', 'xy'}
        The direction of the artifacts to be removed.

    Returns
    -------
    Tensor
        The output image.
    """
    if img.ndim != 2:
        raise ValueError("Input tensor must be two dimensional.")

    ny, nx = img.shape
    dk_y, dk_x = 1 / pixel_size_m / ny, 1 / pixel_size_m / nx
    center_y, center_x = math.floor(ny / 2) + 1, math.floor(nx / 2) + 1

    k_max = 0.5 / pixel_size_m
    f_img = torch.fft.fftshift(pmath.fft2_precise(img))
    # Frequencies of the artifacts.
    dk_s_y, dk_s_x = 1 / period_y_m, 1 / period_x_m

    x_range, y_range = torch.zeros([1], dtype=torch.int32), torch.zeros([1], dtype=torch.int32)
    # Get the frequencies of all harmonic peaks.
    if "x" in direction:
        x_range = torch.arange(math.ceil(-k_max / dk_s_x), math.floor(k_max / dk_s_x))
    if "y" in direction:
        y_range = torch.arange(math.ceil(-k_max / dk_s_y), math.floor(k_max / dk_s_y))

    for i in range(len(y_range)):
        for j in range(len(x_range)):
            # Avoid DC.
            if not (x_range[j] == 0 and y_range[i] == 0):
                window_y_lb = int(
                    max(torch.round(y_range[i] * dk_s_y / dk_y) + center_y - window_size, 0)
                )
                window_y_ub = int(
                    min(torch.round(y_range[i] * dk_s_y / dk_y) + center_y + window_size, ny)
                )

                window_x_lb = int(
                    max(torch.round(x_range[j] * dk_s_x / dk_x) + center_x - window_size, 0)
                )
                window_x_ub = int(
                    min(torch.round(x_range[j] * dk_s_x / dk_x) + center_x + window_size, nx)
                )

                f_img[window_y_lb:window_y_ub, window_x_lb:window_x_ub] = 0

    img_new = torch.fft.ifft2(torch.fft.ifftshift(f_img))
    if not img.is_complex():
        img_new = torch.real(img_new)
    return img_new


@timer()
def median_filter_1d(x: Tensor, window_size: int = 5):
    """
    Apply a median filter to a 1D array.

    Parameters
    ----------
    x : Tensor
        A (..., N) tensor.
    window_size : int
        The size of the window.

    Returns
    -------
    Tensor
        The filtered array.
    """
    y = x.detach().clone()
    rad = window_size // 2
    for i in range(rad, x.shape[-1] - rad):
        y[..., i] = torch.median(x[..., i - rad : i - rad + window_size], dim=-1).values
    return y


@timer()
def generate_vignette_mask(
    shape: tuple[int, int], 
    margin: int = 20, 
    sigma: float = 1.0, 
    method: Literal["gaussian", "linear"] = "gaussian",
    device: Optional[torch.device] = None,
):
    """
    Generate a vignette mask for an image of shape `shape`.
    """
    device = device or torch.get_default_device()
    mask = torch.ones(shape, device=device)
    mask = vignette(mask, margin, sigma, method=method)
    return mask


@timer()
def vignette(
    img: Tensor, 
    margin: int = 20, 
    sigma: float = 1.0, 
    method: Literal["gaussian", "linear", "window"] = "gaussian",
    dim=(-2, -1)):
    """
    Vignett an image so that it gradually decays near the boundary.
    For each dimension of the image, a mask with a width of `2 * margin`
    and with half of it filled with 0s and half with 1s is
    generated and convolved with a Gaussian kernel of size
    `margin` and standard deviation `sigma`. The blurred mask is cropped and
    multiplied to the near-edge regions of the image.

    This function is not differentiable because of the clone and
    slice-assignment operation.

    Parameters
    ----------
    img : Tensor
        The input image.
    margin : int
        The margin of image where the decay takes place. 
        Only used if `method` is "gaussian" or "linear".
    sigma : float
        The standard deviation of the Gaussian kernel. 
        Only used if `method` is "gaussian".
    method : Literal["gaussian", "linear", "window"]
        The method to use to generate the vignette mask. 
        "window" is a Hann window.
    """
    dims = [d % img.ndim for d in dim]
    img = img.clone()
    for i_dim in dims:
        if img.shape[i_dim] <= 2 * margin:
            continue
        mask_shape = (
            [img.shape[i] for i in range(i_dim)]
            + [2 * margin]
            + [img.shape[i] for i in range(i_dim + 1, img.ndim)]
        )
        if method == "gaussian":
            mask = torch.zeros(mask_shape, device=img.device)
            mask_slicer = [slice(None)] * i_dim + [slice(margin, None)]
            mask[*mask_slicer] = 1.0
            gauss_win = torch.signal.windows.gaussian(margin // 2, std=sigma, device=img.device)
            gauss_win = gauss_win / torch.sum(gauss_win)
            mask = convolve1d(mask, gauss_win, dim=i_dim, padding="same")
            mask_final_slicer = [slice(None)] * i_dim + [slice(len(gauss_win), len(gauss_win) + margin)]
            mask = mask[*mask_final_slicer]
            mask = torch.where(mask < 1e-3, 0, mask)
        elif method == "linear":
            ramp = torch.linspace(0, 1, margin)
            new_shape = [1] * img.ndim
            new_shape[i_dim] = margin
            ramp = ramp.reshape(new_shape)
            rep = list(img.shape)
            rep[i_dim] = 1
            mask = ramp.repeat(rep)
        elif method == "window":
            window_func = torch.hamming_window(
                img.shape[i_dim], periodic=True, alpha=0.5, beta=0.5
            )
            img = img * window_func.reshape([-1] + [1] * (img.ndim - i_dim - 1))
        else:
            raise ValueError(f"Unknown method: {method}")

        if method in ["gaussian", "linear"]:
            slicer = [slice(None)] * i_dim + [slice(0, margin)]
            img[slicer] = img[slicer] * mask

            slicer = [slice(None)] * i_dim + [slice(-margin, None)]
            img[slicer] = img[slicer] * mask.flip(i_dim)
    return img


@timer()
def remove_polynomial_background(
    img: Tensor,
    flat_region_mask: Tensor,
    polyfit_order: int = 1,
) -> Tensor:
    """
    Fit a 2D polynomial to the region that is supposed to be flat in an
    image, and subtract the fitted function from the image.

    Parameters
    ----------
    img : Tensor
        The input image.
    flat_region_mask : Tensor
        A boolean mask with the same shape as `img` that specifies the region of
        the image that should be flat.
    polyfit_order : int, optional
        The order of the polynomial to fit. Should be an integer >= 0. If 0,
        just subtract the average. The default is 1.

    Returns
    -------
    Tensor
        The image with the polynomial background subtracted.
    """
    if polyfit_order == 0:
        return img - img[flat_region_mask].mean()

    if polyfit_order < 0:
        raise ValueError("polyfit_order must be >= 0.")

    ys, xs = torch.where(flat_region_mask)
    if ys.numel() == 0:
        raise ValueError("flat_region_mask must contain at least one True value.")

    y_full, x_full = torch.meshgrid(
        torch.arange(img.shape[0]), torch.arange(img.shape[1]), indexing="ij"
    )
    y_full = y_full.reshape(-1)
    x_full = x_full.reshape(-1)

    fit_dtype = torch.float64
    ys = ys.to(device=img.device, dtype=fit_dtype)
    xs = xs.to(device=img.device, dtype=fit_dtype)
    y_full = y_full.to(device=img.device, dtype=fit_dtype)
    x_full = x_full.to(device=img.device, dtype=fit_dtype)

    y_scale = max(img.shape[0] - 1, 1)
    x_scale = max(img.shape[1] - 1, 1)
    ys = ys / y_scale
    xs = xs / x_scale
    y_full = y_full / y_scale
    x_full = x_full / x_scale

    basis = []
    basis_full = []
    for total_order in range(polyfit_order + 1):
        for y_order in range(total_order + 1):
            x_order = total_order - y_order
            basis.append((ys**y_order) * (xs**x_order))
            basis_full.append((y_full**y_order) * (x_full**x_order))

    a_mat = torch.stack(basis, dim=1)
    b_vec = img[flat_region_mask].reshape(-1, 1).to(fit_dtype)
    b_mean = b_vec.mean()
    b_std = b_vec.std()
    if b_std == 0:
        b_std = torch.tensor(1.0, dtype=b_vec.dtype, device=b_vec.device)
    b_vec_std = (b_vec - b_mean) / b_std
    x_vec = torch.linalg.lstsq(a_mat, b_vec_std).solution

    a_mat_full = torch.stack(basis_full, dim=1)
    bg = a_mat_full @ x_vec

    bg = bg * b_std + b_mean
    bg = bg.reshape(img.shape).to(img.dtype)

    return img - bg


@timer()
def unwrap_phase_2d(
    img: Tensor,
    fourier_shift_step: float = 0.5,
    image_grad_method: Literal[
        "fourier_shift", "fourier_differentiation", "nearest"
    ] = "fourier_differentiation",
    image_integration_method: Literal[
        "fourier", "discrete", "deconvolution"
    ] = "fourier",
    weight_map: Optional[Tensor] = None,
    flat_region_mask: Optional[Tensor] = None,
    deramp_polyfit_order: int = 1,
    return_phase_grads: bool = False,
    eps: float = 1e-9,
):
    """
    Unwrap phase of 2D image.

    Parameters
    ---------- 
    img : Tensor
        A complex 2D tensor giving the image.
    fourier_shift_step : float
        The finite-difference step size used to calculate the gradient,
        if the Fourier shift method is used.
    image_grad_method : str
        The method used to calculate the phase gradient.
            - "fourier_shift": Use Fourier shift to perform shift.
            - "nearest": Use nearest neighbor to perform shift.
            - "fourier_differentiation": Use Fourier differentiation.
    image_integration_method : str
        The method used to integrate the image back from gradients.
            - "fourier": Use Fourier integration as implemented in PtychoShelves.
            - "deconvolution": Deconvolve ramp filter.
            - "discrete": Use cumulative sum.
    weight_map : Optional[Tensor]
        A weight map multiplied to the input image.
    flat_region_mask : Optional[Tensor]
        A boolean mask with the same shape as `img` that specifies the region of
        the image that should be flat. This is used to remove unrealistic phase
        ramps in the image. If None, de-ramping will not be done.
    deramp_polyfit_order : int
        The order of the polynomal fit used to de-ramp the phase.
    return_phase_grads : bool
        Whether to return the phase gradient.
    eps : float
        A small number to avoid division by zero.

    Returns
    -------
    Tensor
        The phase of the original image after unwrapping.
    """
    if not img.is_complex():
        raise ValueError("Input tensor must be complex.")

    if isinstance(weight_map, Tensor):
        weight_map = torch.clip(weight_map, 0.0, 1.0)
    else:
        weight_map = 1

    img = weight_map * img / (img.abs() + eps)
    bc_center = torch.angle(img[img.shape[0] // 2, img.shape[1] // 2])

    # Pad image to avoid FFT boundary artifacts.
    padding = [64, 64]
    if torch.any(torch.tensor(padding) > 0):
        if img.shape[-2] > padding[0] and img.shape[-1] > padding[1]:
            padding_mode = "reflect"
        else:
            padding_mode = "replicate"
        img = torch.nn.functional.pad(
            img[None, None, :, :], (padding[1], padding[1], padding[0], padding[0]), mode=padding_mode
        )[0, 0]
        img = vignette(img, margin=10, sigma=2.5)

    gy, gx = get_phase_gradient(
        img, fourier_shift_step=fourier_shift_step, image_grad_method=image_grad_method
    )

    if image_integration_method == "discrete" and torch.any(torch.tensor(padding) > 0):
        gy = gy[padding[0] : -padding[0], padding[1] : -padding[1]]
        gx = gx[padding[0] : -padding[0], padding[1] : -padding[1]]
    if image_integration_method == "discrete":
        phase = torch.real(integrate_image_2d(gy, gx, bc_center=bc_center))
    elif image_integration_method == "fourier":
        phase = torch.real(integrate_image_2d_fourier(gy, gx))
    elif image_integration_method == "deconvolution":
        phase = torch.real(integrate_image_2d_deconvolution(gy, gx, bc_center=bc_center))
    else:
        raise ValueError(f"Unknown integration method: {image_integration_method}")
    
    if image_integration_method != "discrete" and torch.any(torch.tensor(padding) > 0):
        gy = gy[padding[0] : -padding[0], padding[1] : -padding[1]]
        gx = gx[padding[0] : -padding[0], padding[1] : -padding[1]]
        phase = phase[padding[0] : -padding[0], padding[1] : -padding[1]]

    if flat_region_mask is not None:
        phase = remove_polynomial_background(
            phase, flat_region_mask, polyfit_order=deramp_polyfit_order
        )
    if return_phase_grads:
        return phase, (gy, gx)
    return phase


@timer()
def find_center_of_mass(img: Tensor):
    """
    Find the center of mass of one or a stack of 2D images.
    
    Parameters
    ----------
    img : Tensor
        A (N, H, W) or (H, W) tensor giving a batch of images or a single image.
    
    Returns
    -------
    Tensor
        A (N, 2) or (2,) tensor giving the centers of mass.
    """
    orig_ndim = img.ndim
    if img.ndim == 2:
        img = img[None]
    
    if img.is_complex():
        img = torch.abs(img)
    
    # Find the centers of mass of the last 2 dimensions.
    y, x = torch.meshgrid(torch.arange(img.shape[-2]), torch.arange(img.shape[-1]), indexing="ij")
    y = y.to(img.device)
    x = x.to(img.device)
    r = torch.stack((y, x), dim=-1)
    com = (r[None] * img[..., None]).sum(dim=(-3, -2)) / img.sum(dim=(-2, -1))[..., None]
    if orig_ndim == 2:
        return com[0]
    return com


@timer()
def central_crop(img: Tensor, crop_size: tuple[int, int]) -> Tensor:
    """
    Crop the center of an image.
    
    Parameters
    ----------
    img : Tensor
        A (..., H, W) tensor giving the input image(s).
    crop_size : tuple[int, int]
        crop size.
    
    Returns
    -------
    Tensor
        The image cropped to the target size.
    """
    return img[
        ..., 
        img.shape[-2] // 2 - crop_size[0] // 2 : img.shape[-2] // 2 - crop_size[0] // 2 + crop_size[0], 
        img.shape[-1] // 2 - crop_size[1] // 2 : img.shape[-1] // 2 - crop_size[1] // 2 + crop_size[1]
    ]


@timer()
def central_pad(
    img: Tensor, 
    target_size: tuple[int, int], 
    mode: Literal["constant", "reflect", "replicate", "circular"] = "constant", 
    value: float = 0.0
) -> Tensor:
    """
    Pad the center of an image.
    
    Parameters
    ----------
    img : Tensor
        A (..., H, W) tensor giving the input image(s).
    target_size : tuple[int, int]
        target size.
    
    Returns
    -------
    Tensor
        The image padded to the target size.
    """
    pad_size = [
        (target_size[0] - img.shape[-2]) // 2, target_size[-2] - (target_size[0] - img.shape[-2]) // 2 - img.shape[-2],
        (target_size[1] - img.shape[-1]) // 2, target_size[-1] - (target_size[1] - img.shape[-1]) // 2 - img.shape[-1]
    ]
    return torch.nn.functional.pad(img, (pad_size[2], pad_size[3], pad_size[0], pad_size[1]), mode=mode, value=value)


@timer()
def central_crop_or_pad(img: Tensor, target_size: tuple[int, int]) -> Tensor:
    """
    Crop or pad the center of an image to the target size.
    
    Parameters
    ----------
    img : Tensor
        A (..., H, W) tensor giving the input image(s).
    target_size : tuple[int, int]
        target size.
    
    Returns
    -------
    Tensor
        The image cropped or padded to the target size.
    """
    for i in range(2):
        target_size_current_dim = [img.shape[-2]] * (i == 1) + [target_size[i]] + [img.shape[-1]] * (i == 0)
        if img.shape[-2 + i] > target_size[-2 + i]:
            img = central_crop(img, target_size_current_dim)
        elif img.shape[-2 + i] < target_size[-2 + i]:
            img = central_pad(img, target_size_current_dim)
    return img


def fourier_resize(
    image: Tensor,
    target_shape: tuple[int, int],
    *,
    adjoint: bool = False,
    normalized_adjoint: bool = False,
) -> Tensor:
    """Resize the last two dimensions by centered Fourier cropping/padding.

    The forward operation preserves the amplitude of a constant input.  With
    ``adjoint=True`` this returns its exact Hermitian adjoint.  Setting
    ``normalized_adjoint=True`` additionally applies the normalization that
    makes the reverse operation a pseudoinverse on retained Fourier modes.

    Parameters
    ----------
    image : Tensor
        A real or complex tensor with at least two dimensions.
    target_shape : tuple[int, int]
        Requested height and width.
    adjoint : bool
        Apply the Hermitian adjoint of a forward resize from ``target_shape``
        to ``image.shape[-2:]``.
    normalized_adjoint : bool
        Return the constant-amplitude reverse resize rather than the raw
        Hermitian adjoint. This is only valid when ``adjoint`` is true.
    """
    target_shape = tuple(int(v) for v in target_shape)
    if len(target_shape) != 2 or any(v < 1 for v in target_shape):
        raise ValueError("`target_shape` must contain two positive integers.")
    if normalized_adjoint and not adjoint:
        raise ValueError("`normalized_adjoint` requires `adjoint=True`.")
    source_shape = tuple(image.shape[-2:])
    if source_shape == target_shape:
        return image

    spectrum = torch.fft.fftshift(torch.fft.fft2(image, dim=(-2, -1)), dim=(-2, -1))
    # Align the zero-frequency samples explicitly.  Symmetric ``F.pad`` puts
    # the extra sample on the wrong side for some odd/even transitions.
    source_slices = []
    target_slices = []
    for source_size, target_size in zip(source_shape, target_shape):
        overlap = min(source_size, target_size)
        source_start = source_size // 2 - overlap // 2
        target_start = target_size // 2 - overlap // 2
        source_slices.append(slice(source_start, source_start + overlap))
        target_slices.append(slice(target_start, target_start + overlap))
    resized_spectrum = torch.zeros(
        (*spectrum.shape[:-2], *target_shape), dtype=spectrum.dtype, device=spectrum.device
    )
    resized_spectrum[..., target_slices[0], target_slices[1]] = spectrum[
        ..., source_slices[0], source_slices[1]
    ]
    resized = torch.fft.ifft2(
        torch.fft.ifftshift(resized_spectrum, dim=(-2, -1)), dim=(-2, -1)
    )
    if not adjoint:
        resized = resized * (
            (target_shape[0] * target_shape[1]) / (source_shape[0] * source_shape[1])
        )
    elif normalized_adjoint:
        resized = resized * (
            (target_shape[0] * target_shape[1]) / (source_shape[0] * source_shape[1])
        )
    if not image.is_complex():
        resized = resized.real
    return resized
