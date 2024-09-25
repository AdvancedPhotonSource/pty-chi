from typing import Optional, Union, Literal
import dataclasses
import logging

from torch import Tensor
from numpy import ndarray

from .base import ParameterConfig


@dataclasses.dataclass
class OPRModeWeightsConfig(ParameterConfig):
    
    initial_eigenmode_weights: Union[list[float], float] = 0.1
    """
    The initial weight(s) of the eigenmode(s). If it is a scaler, the weights of
    all eigenmodes (i.e., the second and following OPR modes) are set to this value.
    If it is a list, the length should be the number of eigenmodes.
    """
    
    optimize_intensity_variation: bool = False
    """
    Whether to optimize intensity variation, i.e., the weight of the first OPR mode.
    
    The behavior of this parameter is currently different between LSQMLReconstructor and
    other reconstructors.
    
    LSQMLReconstructor:
        - If `optimizable == True` but `optimize_intensity_variation == False`: only
            the weights of eigenmodes (2nd and following OPR modes) are optimized.
        - If `optimizable == True` and `optimize_intensity_variation == True`: both
            the weights of eigenmodes and the weight of the first OPR mode are optimized.
        - If `optimizable == False`: nothing is optimized.
    Other reconstructors:
        - This parameter is ignored.
        - If `optimizable == True`: both the weights of eigenmodes and the weight of 
            the first OPR mode are optimized. 
        - If `optimizable == False`: nothing is optimized.
    """
    
    eigenmode_weight_update_relaxation: float = 0.1
    """
    A separate step size for eigenmode weight update. For now, this is only used by
    LSQMLReconstructor.
    """
    
    def __post_init__(self):
        if self.optimizable:
            assert self.optimize_intensity_variation or self.optimize_eigenmode_weights, \
                'At least 1 of optimize_intensity_variation and optimize_eigenmode_weights should be True.'
