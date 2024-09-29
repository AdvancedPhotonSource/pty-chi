from typing import Optional, Literal, Union
import dataclasses
from dataclasses import field
import json

from torch import Tensor
from numpy import ndarray

from .base import *


@dataclasses.dataclass
class PIEPtychographyReconstructorOptions(ReconstructorOptions):
    
    loss_function: Literal['mse_sqrt', 'poisson', 'mse'] = 'mse_sqrt'
    """
    The loss function.
    """
    

@dataclasses.dataclass
class PIEPtychographyObjectOptions(ObjectOptions):
    pass


@dataclasses.dataclass
class PIEPtychographyProbeOptions(ProbeOptions):
    pass


@dataclasses.dataclass
class PIEPtychographyProbePositionOptions(ProbePositionOptions):
    pass


@dataclasses.dataclass
class PIEPtychographyOPRModeWeightsOptions(OPRModeWeightsOptions):
    pass


@dataclasses.dataclass
class PIEPtychographyOptions(PtychographyTaskOptions):
    
    reconstructor_options: PIEPtychographyReconstructorOptions = field(default_factory=PIEPtychographyReconstructorOptions)
    
    object_options: PIEPtychographyObjectOptions = field(default_factory=PIEPtychographyObjectOptions)
    
    probe_options: PIEPtychographyProbeOptions = field(default_factory=PIEPtychographyProbeOptions)
    
    probe_position_options: PIEPtychographyProbePositionOptions = field(default_factory=PIEPtychographyProbePositionOptions)
    
    opr_mode_weight_options: PIEPtychographyOPRModeWeightsOptions = field(default_factory=PIEPtychographyOPRModeWeightsOptions)
