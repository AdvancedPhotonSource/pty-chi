from typing import Tuple, Literal
import math
import logging

import torch
from torch import Tensor


def extract_patches_fourier_shift(
    image: Tensor, positions: Tensor, shape: Tuple[int, int]
) -> Tensor:
    """Extract patches from 2D object.

    :param image: the whole image.
    :param positions: a tensor of shape (N, 2) giving the center positions of the patches in pixels.
        The origin of the given positions are assumed to be the TOP LEFT corner of the image.
    :param shape: a tuple giving the patch shape in pixels.
    """
    # Floating point ranges over which interpolations should be done
    sys_float = positions[:, 0] - (shape[0] - 1.0) / 2.0
    sxs_float = positions[:, 1] - (shape[1] - 1.0) / 2.0

    # Crop one more pixel each side for Fourier shift
    sys = sys_float.floor().int() - 1
    eys = sys + shape[0] + 2
    sxs = sxs_float.floor().int() - 1
    exs = sxs + shape[1] + 2

    fractional_shifts = torch.stack([sys_float - sys - 1.0, sxs_float - sxs - 1.0], -1)

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
    patches = fourier_shift(patches, -fractional_shifts)
    patches = patches[:, 1:-1, 1:-1]
    return patches


def place_patches_fourier_shift(
    image: Tensor, positions: Tensor, patches: Tensor, op: Literal["add", "set"] = "add"
) -> Tensor:
    """Place patches into a 2D object.

    :param image: the whole image.
    :param positions: a tensor of shape (N, 2) giving the center positions of the patches in pixels.
        The origin of the given positions are assumed to be the TOP LEFT corner of the image.
    :param patches: (N, H, W) tensor ofimage patches.
    """
    shape = patches.shape[-2:]

    # Floating point ranges over which interpolations should be done. +1 to shrink
    # patches by 1 pixel
    sys_float = (positions[:, 0] - (shape[0] - 1.0) / 2.0) + 1
    sxs_float = (positions[:, 1] - (shape[1] - 1.0) / 2.0) + 1

    # Crop one more pixel each side for Fourier shift
    sys = sys_float.floor().int()
    eys = sys + shape[0] - 2
    sxs = sxs_float.floor().int()
    exs = sxs + shape[1] - 2

    fractional_shifts = torch.stack([sys_float - sys, sxs_float - sxs], -1)

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

    patches = fourier_shift(patches, fractional_shifts)
    patches = patches[:, 1:-1, 1:-1]

    for i in range(patches.shape[0]):
        if op == "add":
            image[sys[i] : eys[i], sxs[i] : exs[i]] += patches[i]
        elif op == "set":
            image[sys[i] : eys[i], sxs[i] : exs[i]] = patches[i]

    # Undo padding
    image = image[
        pad_lengths[2] : image.shape[0] - pad_lengths[3],
        pad_lengths[0] : image.shape[1] - pad_lengths[1],
    ]
    return image


def fourier_shift(images: Tensor, shifts: Tensor) -> Tensor:
    """Apply Fourier shift to a batch of images.

    :param images: a [N, H, W] tensor of images.
    :param shifts: a [N, 2] tensor of shifts in pixels.
    :return: shifted images.
    """
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
    return shifted_images


def nearest_neighbor_gradient(
    image: Tensor, direction: Literal["forward", "backward"], dim: Tuple[int, ...] = (0, 1)
) -> Tensor:
    """
    Calculate the nearest neighbor gradient of a 2D image.

    :param image: a (... H, W) tensor of images.
    :param sigma: sigma of the Gaaussian.
    :return: a tuple of 2 images with the gradient in y and x directions.
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


def gaussian_gradient(image: Tensor, sigma: float = 1.0, kernel_size=5) -> Tensor:
    """
    Calculate the gradient of a 2D image with a Gaussian-derivative kernel.

    :param image: a (... H, W) tensor of images.
    :param sigma: sigma of the Gaaussian.
    :return: a tuple of 2 images with the gradient in y and x directions.
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
            logging.debug("Gradient magnitudes between 0 and 1e-6 are set to 0.")
            g = g * torch.logical_not(m)
            grads[i] = g
    grad_y, grad_x = grads
    return grad_y, grad_x


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

    :param image: a (... H, W) tensor of images. If the number of dimensions is greater than 2,
        the last two dimensions are interpreted as height and width, respectively.
    :param kernel: a (H, W) tensor of kernel.
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

    :param f: a (h, w) tensor.
    :param g: a (h, w) tensor.
    :param scale: an integer specifying how much to upsample the cross-correlation.
    :param initial_guess: an initial guess of the location of the cross-correlation peak.
    :param real_space_width: a number specifying the width of the cross-correlation (in units of non-upsampled pixels).
    :return: a tensor of the location of the cross-correlation peak.

    This implementation is based on:
    - Efficient subpixel image registration algorithms (2008) - Manuel Guizar-Sicairos

    The matrix multiplication inverse FT calculation was inferred from formula (12) in the paper:
    - Fast computation of Lyot-style coronagraph propagation (2007) - R. Soummer

    Note: position correction does not work properly when dtype is complex64 instead of complex128.
    The position correction will "walk off" if complex64 is used.
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


def total_variation_2d_chambolle(
    image: Tensor, lmbda: float = 0.01, niter: int = 2, tau: float = 0.125
) -> Tensor:
    """
    Apply total variation constraint to an image using Chambolle's total variation projection method
    (https://doi.org/10.5201/ipol.2013.61).

    Adapted from fold_slice (local_TV2D_chambolle.m).

    :param image: a (H, W) tensor.
    :param lmbda: the trade-off parameter (1 / lambda in the paper).
        Larger lmbda means stronger smoothing.
    :param niter: an integer specifying the number of iterations.
    :param tau: the time-step parameter.
    :return: a (H, W) tensor.
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
