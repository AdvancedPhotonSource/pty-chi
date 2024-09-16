import torch
from torch import Tensor


def propagate_far_field(wave: Tensor):
    """
    Far-field propagation.

    :param wave: a (..., h, w) complex tensor of incident waves.
    """
    wave = torch.fft.fft2(wave, norm='ortho')
    wave = torch.fft.fftshift(wave, dim=(-2, -1))
    return wave
    

def back_propagate_far_field(wave: Tensor):
    """
    Far-field propagation.

    :param wave: a (..., h, w) complex tensor of incident waves.
    """
    wave = torch.fft.ifftshift(wave, dim=(-2, -1))
    wave = torch.fft.ifft2(wave, norm='ortho')
    return wave