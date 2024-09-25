from typing import Optional, Union, Literal
import dataclasses

from torch import Tensor
from numpy import ndarray

from .base import ParameterConfig


@dataclasses.dataclass
class ProbePositionConfig(ParameterConfig):
    
    position_x_m: Union[ndarray, Tensor] = None
    """The x position in meters."""
    
    position_y_m: Union[ndarray, Tensor] = None
    """The y position in meters."""
    
    pixel_size_m: float = 1.0
    """The pixel size in meters."""
    
    update_magnitude_limit: Optional[float] = 0
    """Magnitude limit of the probe update. No limit is imposed if it is 0."""
