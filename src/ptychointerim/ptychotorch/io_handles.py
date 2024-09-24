from typing import Optional, Union, Tuple, Type

import torch
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
from numpy import ndarray

from .utils import to_tensor


class PtychographyDataset(Dataset):
    
    def __init__(self, 
                 patterns: Union[Tensor, ndarray], 
                 valid_pixel_mask: Optional[Union[Tensor, ndarray]] = None,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.patterns = to_tensor(patterns, device='cpu')
        
        if valid_pixel_mask is None:
            valid_pixel_mask = torch.ones(self.patterns.shape[-2:])
        self.valid_pixel_mask = to_tensor(valid_pixel_mask, device='cpu', dtype=torch.bool)
    
    def __getitem__(self, index):
        index = torch.tensor(index, device='cpu', dtype=torch.long)
        pattern = self.patterns[index]
        return index, pattern
    
    def __len__(self):
        return len(self.patterns)
    
    def move_attributes_to_device(self, device=None):
        if device is None:
            device = torch.get_default_device()
        self.valid_pixel_mask = self.valid_pixel_mask.to(device)
    