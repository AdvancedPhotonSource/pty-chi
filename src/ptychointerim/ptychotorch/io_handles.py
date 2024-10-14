from typing import Optional, Union
import logging

import torch
from torch.utils.data import Dataset
from torch import Tensor
from numpy import ndarray

from .utils import to_tensor


class PtychographyDataset(Dataset):
    def __init__(
        self,
        patterns: Union[Tensor, ndarray],
        valid_pixel_mask: Optional[Union[Tensor, ndarray]] = None,
        wavelength_m: float = None,
        propagation_distance_m: float = 1.0,
        fft_shift: bool = True,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.patterns = to_tensor(patterns, device="cpu")
        if fft_shift:
            self.patterns = torch.fft.fftshift(self.patterns, dim=(-2, -1))
            logging.info("Diffraction data have been FFT-shifted.")

        if valid_pixel_mask is None:
            valid_pixel_mask = torch.ones(self.patterns.shape[-2:])
        self.valid_pixel_mask = to_tensor(valid_pixel_mask, device="cpu", dtype=torch.bool)

        self.wavelength_m = wavelength_m
        self.propagation_distance_m = propagation_distance_m

    def __getitem__(self, index):
        index = torch.tensor(index, device="cpu", dtype=torch.long)
        pattern = self.patterns[index]
        return index, pattern

    def __len__(self):
        return len(self.patterns)

    def move_attributes_to_device(self, device=None):
        if device is None:
            device = torch.get_default_device()
        self.valid_pixel_mask = self.valid_pixel_mask.to(device)
