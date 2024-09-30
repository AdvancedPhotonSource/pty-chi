from typing import Optional, Literal, Union
import dataclasses
from dataclasses import field
import json

from torch import Tensor
from numpy import ndarray

from .base import *
import ptychointerim.api.enums as enums


@dataclasses.dataclass
class PIEPtychographyReconstructorOptions(ReconstructorOptions):
        
    def get_reconstructor_type(self) -> enums.Reconstructors:
        return enums.Reconstructors.PIE
    

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
