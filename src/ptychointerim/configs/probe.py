from typing import Optional, Union, Literal
import dataclasses

from torch import Tensor
from numpy import ndarray

from .base import ParameterConfig


@dataclasses.dataclass
class ProbeConfig(ParameterConfig):
    """
    The probe configuration.
    
    The update behavior of eigenmodes (the second and following OPR modes) is currently
    different between LSQMLReconstructor and other reconstructors.
    
    LSQMLReconstructor:
        - The first OPR mode is always optimized as long as `optimizable == True`.
        - The eigenmodes are optimized only when
            - The probe has multiple OPR modes;
            - `optimizable == True`;
            - `OPRModeWeightsConfig` is given;
            - `OPRModeWeightsConfig` is optimizable.
    
    Other reconstructors:
        - The first OPR mode is always optimized as long as `optimizable == True`.
        - The eigenmodes are optimized when
            - The probe has multiple OPR modes;
            - `optimizable == True`;
            - `OPRModeWeightsConfig` is given.
    """
    
    initial_guess: Union[ndarray, Tensor] = None
    """A (n_opr_modes, n_modes, h, w) complex tensor of the probe initial guess."""
    
    eigenmode_update_relaxation: float = 0.1
    """
    A separate step size for eigenmode update. For now, this is only used by 
    LSQMLReconstructor.
    """
    
    def __post_init__(self):
        assert self.initial_guess is not None and self.initial_guess.ndim == 4, \
            'Probe initial_guess must be a (n_opr_modes, n_modes, h, w) tensor.'
    