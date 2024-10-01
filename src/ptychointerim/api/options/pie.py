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
class PIEPtychographyReconstructorOptions(base.ReconstructorOptions):
        
    def get_reconstructor_type(self) -> enums.Reconstructors:
        return enums.Reconstructors.PIE
    

@dataclasses.dataclass
class PIEPtychographyObjectOptions(base.ObjectOptions):
    pass


@dataclasses.dataclass
class PIEPtychographyProbeOptions(base.ProbeOptions):
    pass


@dataclasses.dataclass
class PIEPtychographyProbePositionOptions(base.ProbePositionOptions):
    pass


@dataclasses.dataclass
class PIEPtychographyOPRModeWeightsOptions(base.OPRModeWeightsOptions):
    pass


@dataclasses.dataclass
class PIEPtychographyOptions(task_options.PtychographyTaskOptions):
    
    reconstructor_options: PIEPtychographyReconstructorOptions = field(default_factory=PIEPtychographyReconstructorOptions)
    
    object_options: PIEPtychographyObjectOptions = field(default_factory=PIEPtychographyObjectOptions)
    
    probe_options: PIEPtychographyProbeOptions = field(default_factory=PIEPtychographyProbeOptions)
    
    probe_position_options: PIEPtychographyProbePositionOptions = field(default_factory=PIEPtychographyProbePositionOptions)
    
    opr_mode_weight_options: PIEPtychographyOPRModeWeightsOptions = field(default_factory=PIEPtychographyOPRModeWeightsOptions)
