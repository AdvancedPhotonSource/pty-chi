from typing import Optional, Literal, Union
import dataclasses
from dataclasses import field
import json

from torch import Tensor
from numpy import ndarray

from .base import *


@dataclasses.dataclass
class PIEPtychographyReconstructorConfig(ReconstructorConfig):
    
    loss_function: Literal['mse_sqrt', 'poisson', 'mse'] = 'mse_sqrt'
    """
    The loss function.
    """
    

@dataclasses.dataclass
class PIEPtychographyObjectConfig(ObjectConfig):
    pass


@dataclasses.dataclass
class PIEPtychographyProbeConfig(ProbeConfig):
    pass


@dataclasses.dataclass
class PIEPtychographyProbePositionConfig(ProbePositionConfig):
    pass


@dataclasses.dataclass
class PIEPtychographyOPRModeWeightsConfig(OPRModeWeightsConfig):
    pass


@dataclasses.dataclass
class PIEPtychographyConfig(PtychographyJobConfig):
    
    reconstructor_config: PIEPtychographyReconstructorConfig = field(default_factory=PIEPtychographyReconstructorConfig)
    
    object_config: PIEPtychographyObjectConfig = field(default_factory=PIEPtychographyObjectConfig)
    
    probe_config: PIEPtychographyProbeConfig = field(default_factory=PIEPtychographyProbeConfig)
    
    probe_position_config: PIEPtychographyProbePositionConfig = field(default_factory=PIEPtychographyProbePositionConfig)
    
    opr_mode_weight_config: PIEPtychographyOPRModeWeightsConfig = field(default_factory=PIEPtychographyOPRModeWeightsConfig)
