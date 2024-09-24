from typing import Tuple, Literal
import math
import logging

import torch
from torch import Tensor


def extract_patches_fourier_shift(
        image: Tensor, 
        positions: Tensor, 
        shape: Tuple[int, int]
    ) -> Tensor:
    """Extract patches from 2D object.

    :param image: the whole image.
    :param positions: a tensor of shape (N, 2) giving the center positions of the patches in pixels.
        The origin of the given positions are assumed to be the TOP LEFT corner of the image.
    :param shape: a tuple giving the patch shape in pixels.
    """    
    # Floating point ranges over which interpolations should be done
    sys_float = (positions[:, 0] - (shape[0] - 1.0) / 2.0)
    sxs_float = (positions[:, 1] - (shape[1] - 1.0) / 2.0)
    
    # Crop one more pixel each side for Fourier shift
    sys = sys_float.floor().int() - 1
    eys = sys + shape[0] + 2
    sxs = sxs_float.floor().int() - 1
    exs = sxs + shape[1] + 2
    
    fractional_shifts = torch.stack([sys_float - sys - 1.0, sxs_float - sxs - 1.0], -1)
    
    pad_lengths = [max(-sxs.min(), 0), max(exs.max() - image.shape[1], 0), 
                   max(-sys.min(), 0), max(eys.max() - image.shape[0], 0)]
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
        image: Tensor, 
        positions: Tensor, 
        patches: Tensor,
        op: Literal['add', 'set'] = 'add'
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
    
    pad_lengths = [max(-sxs.min(), 0), max(exs.max() - image.shape[1], 0), 
                   max(-sys.min(), 0), max(eys.max() - image.shape[0], 0)]
    image = torch.nn.functional.pad(image, pad_lengths)
    sys = sys + pad_lengths[2]
    eys = eys + pad_lengths[2]
    sxs = sxs + pad_lengths[0]
    exs = exs + pad_lengths[0]
    
    patches = fourier_shift(patches, fractional_shifts)
    patches = patches[:, 1:-1, 1:-1]
    
    for i in range(patches.shape[0]):
        if op == 'add':
            image[sys[i]:eys[i], sxs[i]:exs[i]] += patches[i]
        elif op == 'set':
            image[sys[i]:eys[i], sxs[i]:exs[i]] = patches[i]
        
    # Undo padding
    image = image[pad_lengths[2]:image.shape[0] - pad_lengths[3], 
                  pad_lengths[0]:image.shape[1] - pad_lengths[1]]
    return image
    

def fourier_shift(images: Tensor, shifts: Tensor) -> Tensor:
    """Apply Fourier shift to a batch of images.

    :param images: a [N, H, W] tensor of images.
    :param shifts: a [N, 2] tensor of shifts in pixels.
    :return: shifted images.
    """
    ft_images = torch.fft.fft2(images)
    freq_y, freq_x = torch.meshgrid(torch.fft.fftfreq(images.shape[-2]), torch.fft.fftfreq(images.shape[-1]), indexing='ij')
    freq_x = freq_x.to(ft_images.device)
    freq_y = freq_y.to(ft_images.device)
    freq_x = freq_x.repeat(images.shape[0], 1, 1)
    freq_y = freq_y.repeat(images.shape[0], 1, 1)
    mult = torch.exp(1j * -2 * torch.pi * (freq_x * shifts[:, 1].view(-1, 1, 1) + freq_y * shifts[:, 0].view(-1, 1, 1)))
    ft_images = ft_images * mult
    shifted_images = torch.fft.ifft2(ft_images)
    if not images.dtype.is_complex:
        shifted_images = shifted_images.real
    return shifted_images


def gaussian_gradient(image: Tensor, sigma: float = 1.0, kernel_size=5) -> Tensor:
    """
    Calculate the gradient of a 2D image with a Gaussian-derivative kernel.

    :param image: a (... H, W) tensor of images.
    :param sigma: sigma of the Gaaussian.
    :return: a tuple of 2 images with the gradient in y and x directions. 
    """
    r = torch.arange(kernel_size) - (kernel_size - 1) / 2.0
    kernel = -r / (math.sqrt(2 * math.pi) * sigma ** 3) * torch.exp(-r ** 2 / (2 * sigma ** 2))
    grad_y = convolve2d(image, kernel.view(-1, 1), padding='same', padding_mode='replicate')
    grad_x = convolve2d(image, kernel.view(1, -1), padding='same', padding_mode='replicate')
    
    # Gate the gradients
    grads = [grad_y, grad_x]
    for i, g in enumerate(grads):
        m = torch.logical_and(grad_y.abs() < 1e-6, grad_y.abs() != 0)
        if torch.count_nonzero(m) > 0:
            logging.debug('Gradient magnitudes between 0 and 1e-6 are set to 0.')
            g = g * torch.logical_not(m)
            grads[i] = g
    grad_y, grad_x = grads
    return grad_y, grad_x


def convolve2d(image: Tensor, 
               kernel: Tensor, 
               padding: Literal['same', 'valid'] = 'same', 
               padding_mode: Literal['replicate', 'constant'] = 'replicate'
    ) -> Tensor:
    """
    2D convolution with an explicitly given kernel using torch.nn.functional.conv2d.
    
    This routine flips the kernel to adhere with the textbook definition of convolution.
    torch.nn.functional.conv2d does not flip the kernel in itself.
    
    :param image: a (... H, W) tensor of images. If the number of dimensions is greater than 2, 
        the last two dimensions are interpreted as height and width, respectively.
    :param kernel: a (H, W) tensor of kernel.
    """
    assert(image.ndim >= 2)
    assert(kernel.ndim == 2)
    assert(kernel.shape[-2] % 2 == 1 and kernel.shape[-1] % 2 == 1)
    
    if image.dtype.is_complex:
        kernel = kernel.type(image.dtype)
    
    # Reshape image to (N, 1, H, W).
    orig_shape = image.shape
    image = image.reshape(-1, 1, image.shape[-2], image.shape[-1])
    
    # Reshape kernel to (1, 1, H, W).
    kernel = kernel.flip((0, 1))
    kernel = kernel.reshape(1, 1, kernel.shape[-2], kernel.shape[-1])
    
    if padding == 'same':
        pad_lengths = [kernel.shape[-1] // 2, kernel.shape[-1] // 2,
                       kernel.shape[-2] // 2, kernel.shape[-2] // 2]
        image = torch.nn.functional.pad(image, pad_lengths, mode=padding_mode)
    
    result = torch.nn.functional.conv2d(image, kernel, padding='valid')
    result = result.reshape(*orig_shape[:-2], result.shape[-2], result.shape[-1])
    return result
    

def find_cross_corr_peak(f, g, scale, x0=0, y0=0, real_space_width=3, dtype=torch.complex128):
    # Find the shift applied to g that maximizes the cross-correlation
    # of f and g

    dtype_real = torch.tensor(1+1j, dtype=dtype).real.dtype

    M, N = f.shape
    u = torch.fft.fftfreq(M, 1, dtype=dtype_real)[:, None]
    v = torch.fft.fftfreq(N, 1, dtype=dtype_real)[:, None]
    
    F = torch.fft.fft2(f.to(dtype))
    G = torch.fft.fft2(g.to(dtype))

    FG = F*G.conj()

    num_pts = int(real_space_width*scale/2) * 2 + 1
    span = torch.linspace(-real_space_width/2, real_space_width/2, num_pts, dtype=dtype_real)

    x = x0 + span[:, None]
    y = y0 + span[:, None]

    # Do inverse FFT using matrix multiplication
    y_exp = torch.exp(2j * torch.pi * torch.matmul(v, y.transpose(1, 0))).to(dtype)
    tmp = torch.matmul(FG, y_exp)
    x_exp = torch.exp(2j * torch.pi * torch.matmul(x, u.transpose(1, 0))).to(dtype)
    corr = torch.matmul(x_exp, tmp)

    max_idx = torch.tensor(torch.unravel_index(torch.argmax(corr.abs()), corr.shape))
    est_shift = torch.tensor([x[max_idx[0]] , y[max_idx[1]]])

    return corr, est_shift
    
    
if __name__ == '__main__':
    img = torch.zeros(10, 10)
    img[5:, 5:] = 1
    gy, gx = gaussian_gradient(img, sigma=0.33)
    print(gx)
    
    import scipy.ndimage as ndi
    img = img.detach().numpy()
    gx = ndi.gaussian_filter1d(-img, sigma=0.3, order=1, axis=-1, mode='nearest')
    print(gx)
    