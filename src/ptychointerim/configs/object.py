from typing import Optional, Union, Literal
import dataclasses

from torch import Tensor
from numpy import ndarray

from .base import ParameterConfig


@dataclasses.dataclass
class ObjectConfig(ParameterConfig):
    
    initial_guess: Union[ndarray, Tensor] = None
    """A (h, w) complex tensor of the object initial guess."""
    
    pixel_size_m: float = 1.0
    """The pixel size in meters."""
    