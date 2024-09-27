from typing import Optional, Literal, Union
import dataclasses
from dataclasses import field
import json

from torch import Tensor
from numpy import ndarray

from .base import *


@dataclasses.dataclass
class LSQMLReconstructorOptions(ReconstructorOptions):

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
class LSQMLObjectOptions(ObjectOptions):
    pass


@dataclasses.dataclass
class LSQMLProbeOptions(ProbeOptions):
    
    eigenmode_update_relaxation: float = 0.1
    """
    A separate step size for eigenmode update. For now, this is only used by 
    LSQMLReconstructor.
    """
    
    
@dataclasses.dataclass
class LSQMLProbePositionOptions(ProbePositionOptions):
    pass


@dataclasses.dataclass
class LSQMLOPRModeWeightsOptions(OPRModeWeightsOptions):
    
    update_relaxation: float = 0.1
    """
    A separate step size for eigenmode weight update. For now, this is only used by
    LSQMLReconstructor.
    """
    

@dataclasses.dataclass
class LSQMLOptions(PtychographyTaskOptions):

    reconstructor_options: LSQMLReconstructorOptions = field(default_factory=LSQMLReconstructorOptions)
    
    object_options: LSQMLObjectOptions = field(default_factory=LSQMLObjectOptions)
    
    probe_options: LSQMLProbeOptions = field(default_factory=LSQMLProbeOptions)
    
    probe_position_options: LSQMLProbePositionOptions = field(default_factory=LSQMLProbePositionOptions)
    
    opr_mode_weight_options: LSQMLOPRModeWeightsOptions = field(default_factory=LSQMLOPRModeWeightsOptions)
        