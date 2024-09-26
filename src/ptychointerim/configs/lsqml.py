from typing import Optional, Literal, Union
import dataclasses
from dataclasses import field
import json

from torch import Tensor
from numpy import ndarray

from .base import *


@dataclasses.dataclass
class LSQMLReconstructorConfig(ReconstructorConfig):

    noise_model: Literal['gaussian', 'poisson'] = 'gaussian'
    
    noise_model_params: Optional[dict] = None
    """
    Noise model parameters. Depending on the choice of `noise_model`, the dictionary can contain the 
    following keys:
    
    Gaussian noise model:
        - 'gaussian_noise_std': The standard deviation of the gaussian noise.
    Poisson noise model:
        (None)
    """
    

@dataclasses.dataclass
class LSQMLObjectConfig(ObjectConfig):
    pass


@dataclasses.dataclass
class LSQMLProbeConfig(ProbeConfig):
    
    eigenmode_update_relaxation: float = 0.1
    """
    A separate step size for eigenmode update. For now, this is only used by 
    LSQMLReconstructor.
    """
    
    
@dataclasses.dataclass
class LSQMLProbePositionConfig(ProbePositionConfig):
    pass


@dataclasses.dataclass
class LSQMLOPRModeWeightsConfig(OPRModeWeightsConfig):
    
    update_relaxation: float = 0.1
    """
    A separate step size for eigenmode weight update. For now, this is only used by
    LSQMLReconstructor.
    """
    

@dataclasses.dataclass
class LSQMLConfig(PtychographyJobConfig):

    reconstructor_config: LSQMLReconstructorConfig = field(default_factory=LSQMLReconstructorConfig)
    
    object_config: LSQMLObjectConfig = field(default_factory=LSQMLObjectConfig)
    
    probe_config: LSQMLProbeConfig = field(default_factory=LSQMLProbeConfig)
    
    probe_position_config: LSQMLProbePositionConfig = field(default_factory=LSQMLProbePositionConfig)
    
    opr_mode_weight_config: LSQMLOPRModeWeightsConfig = field(default_factory=LSQMLOPRModeWeightsConfig)
        