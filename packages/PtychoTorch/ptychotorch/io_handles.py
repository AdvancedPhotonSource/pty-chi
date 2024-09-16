from typing import Optional, Union, Tuple, Type

import torch
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
from numpy import ndarray

from .utils import to_tensor


class PtychographyDataset(Dataset):
    
    def __init__(self, 
                 patterns: Union[Tensor, ndarray], 
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.patterns = to_tensor(patterns, device='cpu')
    
    def __getitem__(self, index):
        index = torch.tensor(index, device='cpu', dtype=torch.long)
        pattern = self.patterns[index]
        return index, pattern
    
    def __len__(self):
        return len(self.patterns)