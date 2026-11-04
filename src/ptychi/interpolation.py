# Copyright © 2025 UChicago Argonne, LLC All right reserved
# Full license accessible at https://github.com//AdvancedPhotonSource/pty-chi/blob/main/LICENSE

"""Batched translations shared by image shifting and object sampling.

Shifts are (y, x) displacements in pixels: the forward operation samples at
``output_coordinate - shift``. Adjoint operations take the forward shifts.
Spectral translations are periodic; spatial translations replicate boundaries.
Support selection, padding and accumulation belong to the caller.
"""

import torch
from torch import Tensor

import ptychi.maths as pmath


_FREQUENCY_CACHE = {}
_SPATIAL_GRID_CACHE = {}


def _frequencies(device, dtype, shape):
    """Return cached one-dimensional FFT frequencies for a spatial grid.

    ``shape`` is (height, width). The returned (fy, fx) tensors have shapes
    (height,) and (width,), with signed frequencies in cycles per pixel and
    the unshifted ordering used by ``torch.fft.fft2``. They can be broadcast
    along separate axes when constructing a two-dimensional multiplier.

    Cache entries are keyed by device, floating-point dtype and spatial shape.
    The returned tensors are shared cache values and should be treated as
    read-only. They contain no position-dependent quantities or autograd state.
    """
    key = (device, dtype, *shape)
    grid = _FREQUENCY_CACHE.get(key)
    if grid is None:
        grid = tuple(torch.fft.fftfreq(n, device=device, dtype=dtype) for n in shape)
        _FREQUENCY_CACHE[key] = grid
    return grid


def _linear_stencil(coordinates, *, fractional=False):
    """Integer lower neighbors and linear weights, with piecewise derivatives.

    For each coordinate c, return (j, w0, w1), where j = floor(c),
    w1 = c - j and w0 = 1 - w1. Interpolation therefore evaluates
    ``w0 * values[j] + w1 * values[j + 1]``. Coordinates may have any tensor
    shape; the weights retain that shape and the coordinate precision.

    ``fractional=True`` accepts coordinates already in [0, 1), with lower
    neighbor returned as the Python integer zero. Otherwise, the lower
    neighbors are a tensor with the coordinate dtype. Patch geometry uses
    fractional coordinates to avoid rebuilding its indices.

    The weights retain their autograd connection to the coordinates. Their
    derivatives describe the linear segment selected by floor(c); classical
    derivatives need not exist at integer knots. Boundary handling belongs
    to the caller.
    """
    lower = 0 if fractional else coordinates.floor()
    fraction = coordinates if fractional else coordinates - lower
    return lower, 1 - fraction, fraction


def translation_multiplier(shifts, shape, method="fourier", derivative=(0, 0), *,
                           frequency_dtype=None, separable=True):
    r"""Periodic translation multiplier, differentiated with respect to shifts.

    For frequency f in cycles/pixel and displacement s, Fourier uses
    exp(-2 pi i f s). Linear interpolation uses the weighted sum of the phase
    factors for floor(s) and floor(s)+1. Its pure second derivatives vanish
    between knots; mixed derivatives need not vanish. Nearest rounds s and
    returns zero derivatives, including at ties. Derivative units are inverse
    pixels to the total requested order.

    Separable phase factors avoid a full-grid exponential for full-object
    sampling. Image shifts use a dense phase to preserve their established
    floating-point evaluation order, including mixed-precision inputs.

    Parameters
    ----------
    shifts : Tensor
        Floating-point tensor of shape (N, 2), containing (y, x) displacements
        in pixels. Positive displacements move image features toward larger
        indices, so the translated image samples at ``coordinate - shift``.
    shape : tuple of int
        (height, width) of the periodic image grid and its unshifted FFT.
    method : {"fourier", "bilinear", "nearest"}
        Interpolation kernel represented by the multiplier. Nearest rounds
        displacements using ``torch.round``, including its ties-to-even rule.
    derivative : tuple of int
        Nonnegative derivative orders (dy, dx) with respect to the supplied
        displacements; (0, 0) returns the ordinary translation multiplier.
        If a caller defines displacement as ``fixed_center - patch_center``,
        patch-center derivatives acquire a factor of (-1)**(dy + dx).
    frequency_dtype : torch.dtype, optional
        Floating-point precision for the frequency vectors. Defaults to the
        shift dtype; subsequent tensor operations follow PyTorch promotion.
    separable : bool
        For Fourier interpolation, form the multiplier as a product of
        one-dimensional factors when True, or exponentiate the summed
        two-dimensional phase when False. These are mathematically equivalent
        but can differ in floating-point rounding. Bilinear and nearest use
        separable factors.

    Returns
    -------
    Tensor
        Complex multiplier of shape (N, height, width), on the shift device.
        Multiply an image FFT by this tensor before the inverse FFT. For the
        adjoint translation, use its complex conjugate with the same forward
        shifts. The caller schedules transforms, cropping and batch reduction.
    """
    frequencies = _frequencies(
        shifts.device, frequency_dtype or shifts.dtype, shape
    )
    if method == "fourier" and not separable:
        fy, fx = frequencies
        phase = (fx[None, None, :] * shifts[:, 1, None, None]
                 + fy[None, :, None] * shifts[:, 0, None, None])
        multiplier = torch.exp(-2j * torch.pi * phase)
        for axis, order in enumerate(derivative):
            if order:
                frequency = fy[:, None] if axis == 0 else fx[None, :]
                multiplier = multiplier * (-2j * torch.pi * frequency) ** order
        return multiplier
    factors = []
    for axis, (frequency, order) in enumerate(zip(frequencies, derivative)):
        omega = -2j * torch.pi * frequency[None]
        shift = shifts[:, axis, None]
        if method == "fourier":
            factor = torch.exp(omega * shift)
            if order:
                factor = factor * omega ** order
        elif method == "bilinear":
            lower, left_weight, right_weight = _linear_stencil(shift)
            left = torch.exp(omega * lower)
            right = torch.exp(omega * (lower + 1))
            if order == 0:
                factor = left_weight * left + right_weight * right
            elif order == 1:
                factor = right - left
            else:
                factor = (left + right) * 0
        elif method == "nearest":
            factor = torch.exp(omega * shift.round()) + shift * 0
            if order:
                factor = factor * 0
        else:
            raise ValueError(f"Invalid interpolation method: {method}")
        factors.append(factor)
    return factors[0][:, :, None] * factors[1][:, None, :]


def _bilinear_support(images, shifts, shape, adjoint, out=None):
    """Translate valid rectangles with a one-pixel bottom/right support halo.

    Each shift lies in (-1, 0] on both axes. Forward input is (N, h+1, w+1)
    and output is (N, h, w); the adjoint reverses these shapes. Scalar weights
    per position avoid dense coordinate/index grids for patch translations.

    ``shape=(h, w)`` always describes the forward output, and ``shifts`` has
    shape (N, 2) in (y, x) pixels. The caller has already gathered each support
    at its integer origin. A shift (-fy, -fx) samples between that origin and
    its bottom/right neighbors with fractional offsets (fy, fx). All required
    neighbors lie inside the supplied support; no boundary extension is used.

    With ``adjoint=True``, scatter each input value to its four support
    neighbors using the same real weights. Use the forward shifts unchanged.
    By default, contributions are accumulated into a new zero buffer. An
    optional ``out`` buffer instead receives additive updates in place and is
    returned directly; its existing values are retained. This buffer has shape
    (N, h+1, w+1), or (h+1, w+1) for a single patch, and is used only for the
    adjoint. It must be compatible with the input dtype and device.

    Inputs may be real or complex. Weight application follows the image's
    precision, matching scalar-weight multiplication for an individual patch.
    A single-patch output view allows accumulation directly into the object.
    """
    # The sampler has already separated the integer support origin.
    single = len(shifts) == 1
    output_buffer = out
    if single:
        images = images[0]
        if out is not None and out.ndim == 3:
            out = out[0]
    _, left, right = _linear_stencil(-shifts[0] if single else -shifts, fractional=True)
    wy0, wx0 = left.unbind(-1)
    wy1, wx1 = right.unbind(-1)
    weights = (
        wy0 * wx0, wy0 * wx1, wy1 * wx0, wy1 * wx1,
    )
    if not single:
        # Legacy per-patch scalar weights were converted to the image precision.
        dtype = images.real.dtype
        weights = tuple(
            (weight.to(dtype) if weight.dtype != dtype else weight).view(-1, 1, 1)
            for weight in weights
        )
    h, w = shape
    if adjoint:
        result = out if out is not None else images.new_zeros((*images.shape[:-2], h + 1, w + 1))
        for (y, x), weight in zip(((0, 0), (0, 1), (1, 0), (1, 1)), weights):
            result[..., y:y + h, x:x + w].add_(weight * images)
    else:
        result = weights[0] * images[..., :h, :w]
        for (y, x), weight in zip(((0, 1), (1, 0), (1, 1)), weights[1:]):
            result += weight * images[..., y:y + h, x:x + w]
    if output_buffer is not None:
        return output_buffer
    return result[None] if single else result


def _spatial_shift(images, shifts, method, adjoint):
    """Translate full image batches with replicated boundary values.

    ``images`` has shape (N, H, W), and ``shifts`` has shape (N, 2) with
    (y, x) displacements in pixels on the same device. The result retains
    (N, H, W). ``method`` is "bilinear" or "nearest". Forward evaluation
    gathers at ``output_coordinate - shift``; neighbor indices are clamped
    to the image bounds. Nearest rounds the displacement before indexing.

    ``adjoint=True`` uses the same forward shifts and scatter-adds through
    the forward indices and weights. Multiple clamped neighbors contribute
    to the same boundary pixel, so this implements the transpose of the
    boundary extension as well as interpolation. Real weights also make it
    the Hermitian adjoint for complex image data.

    Image-coordinate grids are cached by device and shape. All images are
    processed together, with a fixed loop over interpolation neighbors.
    Shifts close to zero under ``torch.allclose`` return the input tensor
    directly, preserving the existing identity shortcut. Bilinear upper
    neighbors use floor(coordinate + 1) to retain image shifting's numerical
    behavior near floating-point grid knots.
    """
    if torch.allclose(shifts, torch.zeros_like(shifts)):
        return images
    shape = images.shape
    key = (images.device, *shape[-2:])
    grid = _SPATIAL_GRID_CACHE.get(key)
    if grid is None:
        grid = torch.meshgrid(
            torch.arange(shape[-2], device=images.device),
            torch.arange(shape[-1], device=images.device), indexing="ij",
        )
        _SPATIAL_GRID_CACHE[key] = grid
    if method == "nearest":
        # Round the displacement, as in the periodic spectral implementation.
        y = grid[0][None] - shifts[:, 0, None, None].round()
        x = grid[1][None] - shifts[:, 1, None, None].round()
        indices = ((y.int().clamp(0, shape[-2] - 1).reshape(-1),
                    x.int().clamp(0, shape[-1] - 1).reshape(-1)),)
        weights = (torch.ones_like(y),)
    else:
        y = grid[0][None] - shifts[:, 0, None, None]
        x = grid[1][None] - shifts[:, 1, None, None]
        # Preserve image shifting's floating-point evaluation at grid knots.
        # floor(x + 1) and floor(x) + 1 can differ after rounding near zero.
        y0, y1 = y.floor().int(), (y + 1).floor().int()
        x0, x1 = x.floor().int(), (x + 1).floor().int()
        wy0, wy1 = y1 - y, y - y0
        wx0, wx1 = x1 - x, x - x0
        weights = (wy0 * wx0, wy0 * wx1, wy1 * wx0, wy1 * wx1)
        del wy0, wy1, wx0, wx1
        y0 = y0.clamp(0, shape[-2] - 1).reshape(-1)
        y1 = y1.clamp(0, shape[-2] - 1).reshape(-1)
        x0 = x0.clamp(0, shape[-1] - 1).reshape(-1)
        x1 = x1.clamp(0, shape[-1] - 1).reshape(-1)
        indices = ((y0, x0), (y0, x1), (y1, x0), (y1, x1))
    batch = torch.arange(shape[0], device=images.device).repeat_interleave(shape[-2] * shape[-1])
    result = torch.zeros_like(images) if adjoint else None
    for (y, x), weight in zip(indices, weights):
        inds = (batch, y, x)
        if adjoint:
            result.index_put_(inds, (weight * images).reshape(-1), accumulate=True)
        else:
            contribution = weight.reshape(-1) * images[inds]
            result = contribution if result is None else result + contribution
            del contribution
    return result.reshape(shape)


def translate(images: Tensor, shifts: Tensor, method="bilinear", *, adjoint=False,
              valid_shape=None, out=None) -> Tensor:
    """Translate a batch without adding padding.

    ``valid_shape`` selects the compact bilinear support contract described by
    ``_bilinear_support``. Otherwise the spatial shape is preserved. Fourier
    frequency precision follows the default dtype, preserving image shifting's
    existing numerical contract. Real Fourier inputs return the real part.
    For a valid bilinear adjoint, ``out`` accumulates directly into an existing
    support buffer (also accepting a 2D view for a single patch), avoiding a
    temporary and a scatter for a single patch.

    Parameters
    ----------
    images : Tensor
        Real or complex tensor of shape (N, H, W). For compact bilinear
        sampling, the forward input includes the bottom/right support halo.
    shifts : Tensor
        Tensor of shape (N, 2) containing forward (y, x) displacements in
        pixels. It is moved to the image device if necessary. Positive shifts
        move image features toward larger indices.
    method : {"bilinear", "fourier", "nearest"}
        Fourier uses periodic boundaries. Ordinary bilinear/nearest shifts
        replicate boundary values. The sampler implements local nearest by
        rounding patch centers before requesting compact bilinear sampling.
    adjoint : bool
        Apply the adjoint using the same shifts as the forward operation.
        For spatial interpolation this includes accumulation at clamped
        boundaries; Fourier uses the conjugate translation multiplier.
    valid_shape : tuple of int, optional
        Select compact bilinear sampling with forward output shape (h, w).
        This requires shifts in (-1, 0] and forward support (N, h+1, w+1).
        In adjoint mode, the input is (N, h, w) and the output includes the
        halo. Callers use this option with the bilinear method.
    out : Tensor, optional
        Existing support buffer for additive compact bilinear adjoint updates.
        See ``_bilinear_support`` for its shape and mutation contract.

    Returns
    -------
    Tensor
        Translated images or adjoint contributions. Ordinary translations
        preserve spatial shape; compact bilinear sampling follows the shapes
        above. When supplied for the compact adjoint, ``out`` is returned.
    """
    if shifts.device != images.device:
        shifts = shifts.to(images.device)
    if method == "fourier":
        spectrum = pmath.fft2_precise(images)
        multiplier = translation_multiplier(
            shifts, images.shape[-2:], method,
            frequency_dtype=torch.get_default_dtype(), separable=False,
        )
        spectrum = spectrum * (multiplier.conj() if adjoint else multiplier)
        result = pmath.ifft2_precise(spectrum)
        return result if images.is_complex() else result.real
    if method not in ("bilinear", "nearest"):
        raise ValueError(f"Invalid interpolation method: {method}")
    if valid_shape is not None:
        return _bilinear_support(images, shifts, valid_shape, adjoint, out)
    return _spatial_shift(images, shifts, method, adjoint)
