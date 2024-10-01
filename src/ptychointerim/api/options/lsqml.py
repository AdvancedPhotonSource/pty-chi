from typing import Optional, Literal, Union
import dataclasses
from dataclasses import field
import json

from torch import Tensor
from numpy import ndarray

import ptychointerim.api.options.base as base
import ptychointerim.api.options.task as task_options
import ptychointerim.api.enums as enums


@dataclasses.dataclass
class LSQMLReconstructorOptions(base.ReconstructorOptions):
    
    noise_model: enums.NoiseModels = enums.NoiseModels.GAUSSIAN
    """
    The noise model to use.
    """
    
    noise_model_params: Optional[dict] = None
    """
    Noise model parameters. Depending on the choice of `noise_model`, the dictionary can contain the 
    following keys:
    
    Gaussian noise model:
        - 'gaussian_noise_std': The standard deviation of the gaussian noise.
    Poisson noise model:
        (None)
    """
    
    def get_reconstructor_type(self) -> enums.Reconstructors:
        return enums.Reconstructors.LSQML
    

@dataclasses.dataclass
class LSQMLObjectOptions(base.ObjectOptions):
    pass


@dataclasses.dataclass
class LSQMLProbeOptions(base.ProbeOptions):
    
    eigenmode_update_relaxation: float = 0.1
    """
    A separate step size for eigenmode update. For now, this is only used by 
    LSQMLReconstructor.
    """
    
    
@dataclasses.dataclass
class LSQMLProbePositionOptions(base.ProbePositionOptions):
    pass


@dataclasses.dataclass
class LSQMLOPRModeWeightsOptions(base.OPRModeWeightsOptions):
    
    update_relaxation: float = 0.1
    """
    A separate step size for eigenmode weight update. For now, this is only used by
    LSQMLReconstructor.
    """
    

@dataclasses.dataclass
class LSQMLOptions(task_options.PtychographyTaskOptions):

    reconstructor_options: LSQMLReconstructorOptions = field(default_factory=LSQMLReconstructorOptions)
    
    object_options: LSQMLObjectOptions = field(default_factory=LSQMLObjectOptions)
    
    probe_options: LSQMLProbeOptions = field(default_factory=LSQMLProbeOptions)
    
    probe_position_options: LSQMLProbePositionOptions = field(default_factory=LSQMLProbePositionOptions)
    
    opr_mode_weight_options: LSQMLOPRModeWeightsOptions = field(default_factory=LSQMLOPRModeWeightsOptions)
        