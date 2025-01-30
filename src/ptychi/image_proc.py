from typing import Tuple, Literal, Optional, Protocol
import math
import logging

import torch
from torch import Tensor
import torch.signal

import ptychi.maths as pmath
from ptychi.api.types import ComplexTensor, RealTensor
from ptychi.timing.timer_utils import timer, InlineTimer

class PlacePatchesProtocol(Protocol):
    def __call__(
        self, image: Tensor, positions: Tensor, patches: Tensor, op: Literal["add", "set"] = "add"
    ) -> Tensor: ...


class ExtractPatchesProtocol(Protocol):
    def __call__(self, image: Tensor, positions: Tensor, shape: Tuple[int, int]) -> Tensor: ...


logger = logging.getLogger(__name__)


@timer()
def extract_patches_fourier_shift(
    image: Tensor, positions: Tensor, shape: Tuple[int, int]
) -> Tensor:
    """
    Extract patches from 2D object. If a patch's footprint goes outside the image,
    the image is padded with zeros to account for the missing pixels.

    Parameters
    ----------
    image : Tensor
        The whole image.
    positions : Tensor
        A tensor of shape (N, 2) giving the center positions of the patches in pixels.
        The origin of the given positions are assumed to be the TOP LEFT corner of the image.
    shape : tuple of int
        A tuple giving the patch shape in pixels.

    Returns
    -------
    Tensor
        A tensor of shape (N, H, W) containing the extracted patches.
    """
    # Floating point ranges over which interpolations should be done
    sys_float = positions[:, 0] - (shape[0] - 1.0) / 2.0
    sxs_float = positions[:, 1] - (shape[1] - 1.0) / 2.0

    patch_padding = 1

    # Crop one more pixel each side for Fourier shift
    sys = sys_float.floor().int() - patch_padding
    eys = sys + shape[0] + 2 * patch_padding
    sxs = sxs_float.floor().int() - patch_padding
    exs = sxs + shape[1] + 2 * patch_padding

    fractional_shifts = torch.stack([sys_float - sys - patch_padding, sxs_float - sxs - patch_padding], -1)

    pad_lengths = [
        max(-sxs.min(), 0),
        max(exs.max() - image.shape[1], 0),
        max(-sys.min(), 0),
        max(eys.max() - image.shape[0], 0),
    ]
    image = torch.nn.functional.pad(image, pad_lengths)
    sys = sys + pad_lengths[2]
    eys = eys + pad_lengths[2]
    sxs = sxs + pad_lengths[0]
    exs = exs + pad_lengths[0]

    patches = []
    for sy, ey, sx, ex in zip(sys, eys, sxs, exs):
        p = image[sy:ey, sx:ex]
        patches.append(p)
    patches = torch.stack(patches)

    # Apply Fourier shift to account for fractional shifts
    if not torch.allclose(fractional_shifts, torch.zeros_like(fractional_shifts), atol=1e-7):
        patches = fourier_shift(patches, -fractional_shifts)
    patches = patches[:, patch_padding:-patch_padding, patch_padding:-patch_padding]
    return patches


@timer()
def place_patches_fourier_shift(
    image: Tensor, 
    positions: Tensor, 
    patches: Tensor, 
    op: Literal["add", "set"] = "add",
    adjoint_mode: bool = True,
) -> Tensor:
    """
    Place patches into a 2D object. If a patch's footprint goes outside the image,
    the image is padded with zeros to account for the missing pixels.

    Parameters
    ----------
    image : Tensor
        The whole image.
    positions : Tensor
        A tensor of shape (N, 2) giving the center positions of the patches in pixels.
        The origin of the given positions are assumed to be the TOP LEFT corner of the image.
    patches : Tensor
        A (N, H, W) or (H, W) tensor of image patches.
    op : Literal["add", "set"]
        The operation to perform. "add" adds the patches to the image, 
        "set" sets the patches to the image replacing the existing values.
    adjoint_mode : bool
        If True, this function performs the exact adjoint operation of `extract_patches_fourier_shift`.
        This means it will run the adjoint operation of every step of the extraction 
        function in reverse order: it first zero-pads the patches, shifts them back, 
        and puts them back into the image. Turn it on if this function is used in
        backpropagating the gradient. Note that due to the zero-padding, ripple
        artifacts may appear around the borders of each patch, so it is not suitable
        for placing patches that are not gradients. In that case, set this to False,
        and it will skip the zero-padding and crop the patches before placing them
        to remove Fourier shift wrap-arounds.

    Returns
    -------
    Tensor
        A tensor with the same shape as the object with patches added onto it.
    """
    # If the input is a single patch, add the third dimension
    # and expand it to the correct number of patches
    if len(patches.shape) == 2:
        patches = patches[None].expand(len(positions), -1, -1)

    shape = patches.shape[-2:]
    
    if adjoint_mode:
        patch_padding = 1
        patches = torch.nn.functional.pad(patches, [patch_padding] * 4)
    else:
        patch_padding = -1

    sys_float = positions[:, 0] - (shape[0] - 1.0) / 2.0
    sxs_float = positions[:, 1] - (shape[1] - 1.0) / 2.0

    # Crop one more pixel each side for Fourier shift
    sys = sys_float.floor().int() - patch_padding
    eys = sys + shape[0] + 2 * patch_padding
    sxs = sxs_float.floor().int() - patch_padding
    exs = sxs + shape[1] + 2 * patch_padding

    fractional_shifts = torch.stack([sys_float - sys - patch_padding, sxs_float - sxs - patch_padding], -1)

    pad_lengths = [
        max(-sxs.min(), 0),
        max(exs.max() - image.shape[1], 0),
        max(-sys.min(), 0),
        max(eys.max() - image.shape[0], 0),
    ]
    image = torch.nn.functional.pad(image, pad_lengths)
    sys = sys + pad_lengths[2]
    eys = eys + pad_lengths[2]
    sxs = sxs + pad_lengths[0]
    exs = exs + pad_lengths[0]

    if not torch.allclose(fractional_shifts, torch.zeros_like(fractional_shifts), atol=1e-7):
        patches = fourier_shift(patches, fractional_shifts)
    if not adjoint_mode:
        patches = patches[:, abs(patch_padding):-abs(patch_padding), abs(patch_padding):-abs(patch_padding)]

    inline_timer = InlineTimer("add or set patches on image")
    inline_timer.start()
    for i in range(patches.shape[0]):
        if op == "add":
            image[sys[i] : eys[i], sxs[i] : exs[i]] += patches[i]
        elif op == "set":
            image[sys[i] : eys[i], sxs[i] : exs[i]] = patches[i]
    inline_timer.end()

    # Undo padding
    image = image[
        pad_lengths[2] : image.shape[0] - pad_lengths[3],
        pad_lengths[0] : image.shape[1] - pad_lengths[1],
    ]
    return image


class ObjectPatchInterpolator:
    def __init__(self, object_: ComplexTensor, position_px: RealTensor, size: torch.Size) -> None:
        # top left corner of object support
        xmin = position_px[-1] - size[-1] / 2
        ymin = position_px[-2] - size[-2] / 2

        # whole components (pixel indexes)
        xmin_wh = xmin.int()
        ymin_wh = ymin.int()

        # fractional (subpixel) components
        xmin_fr = xmin - xmin_wh
        ymin_fr = ymin - ymin_wh

        # bottom right corner of object patch support
        xmax_wh = xmin_wh + size[-1] + 1
        ymax_wh = ymin_wh + size[-2] + 1

        # reused quantities
        xmin_fr_c = 1.0 - xmin_fr
        ymin_fr_c = 1.0 - ymin_fr

        # barycentric interpolant weights
        self._weight00 = ymin_fr_c * xmin_fr_c
        self._weight01 = ymin_fr_c * xmin_fr
        self._weight10 = ymin_fr * xmin_fr_c
        self._weight11 = ymin_fr * xmin_fr

        # extract patch support region from full object
        self._object_support = object_[ymin_wh:ymax_wh, xmin_wh:xmax_wh]

    @timer()
    def get_patch(self) -> ComplexTensor:
        """interpolate object support to extract patch"""
        object_patch = self._weight00 * self._object_support[:-1, :-1]
        object_patch += self._weight01 * self._object_support[:-1, 1:]
        object_patch += self._weight10 * self._object_support[1:, :-1]
        object_patch += self._weight11 * self._object_support[1:, 1:]
        return object_patch

    @timer()
    def update_patch(self, object_update: ComplexTensor) -> None:
        """add patch update to object support"""
        self._object_support[:-1, :-1] += self._weight00 * object_update
        self._object_support[:-1, 1:] += self._weight01 * object_update
        self._object_support[1:, :-1] += self._weight10 * object_update
        self._object_support[1:, 1:] += self._weight11 * object_update


@timer()
def extract_patches_bilinear_shift(
    image: Tensor,
    positions: Tensor,
    shape: Tuple[int, int],
    round_positions: bool = False,
) -> Tensor:
    if round_positions:
        positions = torch.round(positions).to(int)

    obj_patches = torch.zeros((len(positions), *shape), dtype=image.dtype)
    for i in range(len(positions)):
        interpolator = ObjectPatchInterpolator(
            image,
            positions[i],
            shape,
        )
        obj_patches[i] = interpolator.get_patch()
    return obj_patches


@timer()
def place_patches_bilinear_shift(
    image: Tensor,
    positions: Tensor,
    patches: Tensor,
    op: Literal["add", "set"] = "add",
    round_positions: bool = False,
) -> Tensor:
    if op == "set":
        raise NotImplementedError("\"set\" operation is not supported.")

    if round_positions:
        positions = torch.round(positions).to(int)

    for i in range(len(positions)):
        if patches.shape[0] == len(positions):
            patch_input = patches[i]
        elif len(patches.shape) == 2:
            patch_input = patches
        else:
            raise ValueError("Incorrect patch size.")

        interpolator = ObjectPatchInterpolator(
            image,
            positions[i],
            patch_input.shape,
        )
        interpolator.update_patch(patch_input)
    return image


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
    ft_images = torch.fft.fft2(images)
    freq_y, freq_x = torch.meshgrid(
        torch.fft.fftfreq(images.shape[-2]), torch.fft.fftfreq(images.shape[-1]), indexing="ij"
    )
    freq_x = freq_x.to(ft_images.device)
    freq_y = freq_y.to(ft_images.device)
    freq_x = freq_x.repeat(images.shape[0], 1, 1)
    freq_y = freq_y.repeat(images.shape[0], 1, 1)
    mult = torch.exp(
        1j
        * -2
        * torch.pi
        * (freq_x * shifts[:, 1].view(-1, 1, 1) + freq_y * shifts[:, 0].view(-1, 1, 1))
    )
    ft_images = ft_images * mult
    shifted_images = torch.fft.ifft2(ft_images)
    if not images.dtype.is_complex:
        shifted_images = shifted_images.real
    if strictly_preserve_zeros:
        shifted_images[zero_mask_shifted > 0] = 0
    return shifted_images


@timer()
def nearest_neighbor_gradient(
    image: Tensor, direction: Literal["forward", "backward"], dim: Tuple[int, ...] = (0, 1)
) -> Tensor:
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
    tuple of Tensor
        a tuple of 2 images with the gradient in y and x directions.
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
def gaussian_gradient(image: Tensor, sigma: float = 1.0, kernel_size=5) -> Tensor:
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
    tuple of tensor
        A tuple of 2 images with the gradient in y and x directions.
    """
    r = torch.arange(kernel_size) - (kernel_size - 1) / 2.0
    kernel = -r / (math.sqrt(2 * math.pi) * sigma**3) * torch.exp(-(r**2) / (2 * sigma**2))
    grad_y = convolve2d(image, kernel.view(-1, 1), padding="same", padding_mode="replicate")
    grad_x = convolve2d(image, kernel.view(1, -1), padding="same", padding_mode="replicate")

    # Gate the gradients
    grads = [grad_y, grad_x]
    for i, g in enumerate(grads):
        m = torch.logical_and(grad_y.abs() < 1e-6, grad_y.abs() != 0)
        if torch.count_nonzero(m) > 0:
            logger.debug("Gradient magnitudes between 0 and 1e-6 are set to 0.")
            g = g * torch.logical_not(m)
            grads[i] = g
    grad_y, grad_x = grads
    return grad_y, grad_x


@timer()
def fourier_gradient(image: Tensor) -> Tensor:
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
    u, v = torch.meshgrid(u, v, indexing="ij")
    grad_y = torch.fft.ifft(torch.fft.fft(image, dim=-2) * (2j * torch.pi) * u, dim=-2)
    grad_x = torch.fft.ifft(torch.fft.fft(image, dim=-1) * (2j * torch.pi) * v, dim=-1)
    return grad_y, grad_x


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
    f = torch.fft.fft2(grad_x + 1j * grad_y)
    y, x = torch.fft.fftfreq(shape[0]), torch.fft.fftfreq(shape[1])

    r = torch.exp(2j * torch.pi * (x + y[:, None]))
    r = r / (2j * torch.pi * (x + 1j * y[:, None]))
    r[0, 0] = 0
    integrated_image = f * r
    integrated_image = torch.fft.ifft2(integrated_image)
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
    f_grad_y = torch.fft.fft2(grad_y)
    f_grad_x = torch.fft.fft2(grad_x)
    img = (f_grad_y * tf_y + f_grad_x * tf_x) / (tf_y.abs() ** 2 + tf_x.abs() ** 2 + 1e-5)
    img = -torch.fft.ifft2(img)
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
    x = torch.arange(-((size - 1) / 2), -((size - 1) / 2) + size, 1)
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
    f_img = torch.fft.fftshift(torch.fft.fft2(img))
    # Frequencies of the artifacts.
    dk_s_y, dk_s_x = 1 / period_y_m, 1 / period_x_m

    x_range, y_range = 0, 0
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

    img_new = torch.real(torch.fft.ifft2(torch.fft.ifftshift(f_img)))
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
def vignett(img: Tensor, margin: int = 20, sigma: float = 1.0):
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
    sigma : float
        The standard deviation of the Gaussian kernel.
    """
    img = img.clone()
    for i_dim in range(img.ndim):
        if img.shape[i_dim] <= 2 * margin:
            continue
        mask_shape = (
            [img.shape[i] for i in range(i_dim)]
            + [2 * margin]
            + [img.shape[i] for i in range(i_dim + 1, img.ndim)]
        )
        mask = torch.zeros(mask_shape, device=img.device)
        mask_slicer = [slice(None)] * i_dim + [slice(margin, None)]
        mask[*mask_slicer] = 1.0
        gauss_win = torch.signal.windows.gaussian(margin // 2, std=sigma)
        gauss_win = gauss_win / torch.sum(gauss_win)
        mask = convolve1d(mask, gauss_win, dim=i_dim, padding="same")
        mask_final_slicer = [slice(None)] * i_dim + [slice(len(gauss_win), len(gauss_win) + margin)]
        mask = mask[*mask_final_slicer]
        mask = torch.where(mask < 1e-3, 0, mask)

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
    ys, xs = torch.where(flat_region_mask)
    y_full, x_full = torch.meshgrid(
        torch.arange(img.shape[0]), torch.arange(img.shape[1]), indexing="ij"
    )
    y_full = y_full.reshape(-1)
    x_full = x_full.reshape(-1)

    y_all_orders = []
    x_all_orders = []
    y_full_all_orders = []
    x_full_all_orders = []
    for order in range(polyfit_order + 1):
        y_all_orders.append(ys**order)
        x_all_orders.append(xs**order)
        y_full_all_orders.append(y_full**order)
        x_full_all_orders.append(x_full**order)
    const_basis = torch.ones(len(ys), device=img.device)
    const_basis_full = torch.ones(len(y_full), device=img.device)

    a_mat = torch.stack(y_all_orders + x_all_orders + [const_basis], dim=1)
    b_vec = img[flat_region_mask].reshape(-1, 1)
    x_vec = torch.linalg.solve(a_mat, b_vec)
    a_mat_full = torch.stack(y_full_all_orders + x_full_all_orders + [const_basis_full], dim=1)
    bg = a_mat_full @ x_vec
    bg = bg.reshape(img.shape)
    return img - bg


@timer()
def unwrap_phase_2d(
    img: Tensor,
    fourier_shift_step: float = 0.5,
    image_grad_method: Literal[
        "fourier_shift", "fourier_differentiation", "nearest"
    ] = "fourier_shift",
    image_integration_method: Literal[
        "fourier", "discrete", "deconvolution"
    ] = "deconvolution",
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
        img = torch.nn.functional.pad(
            img[None, None, :, :], (padding[1], padding[1], padding[0], padding[0]), mode="reflect"
        )[0, 0]
        img = vignett(img, margin=10, sigma=2.5)

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
