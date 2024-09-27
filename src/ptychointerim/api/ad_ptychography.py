from typing import Optional, Literal, Union
import dataclasses
from dataclasses import field
import json

from torch import Tensor
from numpy import ndarray

from .base import *


@dataclasses.dataclass
class AutodiffPtychographyReconstructorOptions(ReconstructorOptions):
    
    loss_function: Literal['mse_sqrt', 'poisson', 'mse'] = 'mse_sqrt'
    """
    The loss function.
    """
    

@dataclasses.dataclass
class AutodiffPtychographyObjectOptions(ObjectOptions):
    pass


@dataclasses.dataclass
class AutodiffPtychographyProbeOptions(ProbeOptions):
    pass


@dataclasses.dataclass
class AutodiffPtychographyProbePositionOptions(ProbePositionOptions):
    pass


@dataclasses.dataclass
class AutodiffPtychographyOPRModeWeightsOptions(OPRModeWeightsOptions):
    pass


@dataclasses.dataclass
class AutodiffPtychographyOptions(PtychographyTaskOptions):
    
    reconstructor_options: AutodiffPtychographyReconstructorOptions = field(default_factory=AutodiffPtychographyReconstructorOptions)
    
    object_options: AutodiffPtychographyObjectOptions = field(default_factory=AutodiffPtychographyObjectOptions)
    
    probe_options: AutodiffPtychographyProbeOptions = field(default_factory=AutodiffPtychographyProbeOptions)
    
    probe_position_options: AutodiffPtychographyProbePositionOptions = field(default_factory=AutodiffPtychographyProbePositionOptions)
    
    opr_mode_weight_options: AutodiffPtychographyOPRModeWeightsOptions = field(default_factory=AutodiffPtychographyOPRModeWeightsOptions)
