from typing import Tuple, Literal
import math
import logging

import torch
from torch import Tensor


def extract_patches_fourier_shift(
    image: Tensor, positions: Tensor, shape: Tuple[int, int]
) -> Tensor:
    """
    Extract patches from 2D object.

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
    """
    Place patches into a 2D object.

    Parameters
    ----------
    image : Tensor
        The whole image.
    positions : Tensor
        A tensor of shape (N, 2) giving the center positions of the patches in pixels.
        The origin of the given positions are assumed to be the TOP LEFT corner of the image.
    patches : Tensor
        (N, H, W) tensor ofimage patches.

    Returns
    -------
    Tensor
        A tensor with the same shape as the object with patches added onto it.
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
    """
    Apply Fourier shift to a batch of images.

    Parameters
    ----------
    images : Tensor
        A [N, H, W] tensor of images.
    shifts : Tensor
        A [N, 2] tensor of shifts in pixels.

    Returns
    -------
    Tensor
        Shifted images.
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
