from typing import Optional, Union, Literal
import dataclasses

from torch import Tensor
from numpy import ndarray

from .base import Config


@dataclasses.dataclass
class DataConfig(Config):
    
    pass
    
    
@dataclasses.dataclass
class PtychographyDataConfig(DataConfig):
    
    data: Union[ndarray, Tensor]
    """The data."""
    
    propagation_distance_m: float = 1.0
    """The propagation distance in meters."""
    
    wavelength_m: float = 1e-9
    """The wavelength in meters."""
    
    detector_pixel_size_m: float = 1e-8
    """The detector pixel size in meters."""
    
    valid_pixel_mask: Optional[Union[ndarray, Tensor]] = None
    """A 2D boolean mask where valid pixels are True."""
    