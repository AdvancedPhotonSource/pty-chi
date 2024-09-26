from typing import Optional, Literal, Union
import dataclasses
from dataclasses import field
import json

from torch import Tensor
from numpy import ndarray

from .base import *


@dataclasses.dataclass
class AutodiffPtychographyReconstructorConfig(ReconstructorConfig):
    
    loss_function: Literal['mse_sqrt', 'poisson', 'mse'] = 'mse_sqrt'
    """
    The loss function.
    """
    

@dataclasses.dataclass
class AutodiffPtychographyObjectConfig(ObjectConfig):
    pass


@dataclasses.dataclass
class AutodiffPtychographyProbeConfig(ProbeConfig):
    pass


@dataclasses.dataclass
class AutodiffPtychographyProbePositionConfig(ProbePositionConfig):
    pass


@dataclasses.dataclass
class AutodiffPtychographyOPRModeWeightsConfig(OPRModeWeightsConfig):
    pass


@dataclasses.dataclass
class AutodiffPtychographyConfig(PtychographyJobConfig):
    
    reconstructor_config: AutodiffPtychographyReconstructorConfig = field(default_factory=AutodiffPtychographyReconstructorConfig)
    
    object_config: AutodiffPtychographyObjectConfig = field(default_factory=AutodiffPtychographyObjectConfig)
    
    probe_config: AutodiffPtychographyProbeConfig = field(default_factory=AutodiffPtychographyProbeConfig)
    
    probe_position_config: AutodiffPtychographyProbePositionConfig = field(default_factory=AutodiffPtychographyProbePositionConfig)
    
    opr_mode_weight_config: AutodiffPtychographyOPRModeWeightsConfig = field(default_factory=AutodiffPtychographyOPRModeWeightsConfig)
