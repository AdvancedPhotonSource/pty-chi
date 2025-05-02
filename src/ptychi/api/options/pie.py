# Copyright © 2025 UChicago Argonne, LLC All right reserved
# Full license accessible at https://github.com//AdvancedPhotonSource/pty-chi/blob/main/LICENSE

import dataclasses
from dataclasses import field

import ptychi.api.options.base as base
import ptychi.api.options.task as task_options
import ptychi.api.enums as enums


@dataclasses.dataclass
class PIEReconstructorOptions(base.ReconstructorOptions):
    
    def get_reconstructor_type(self) -> enums.Reconstructors:
        return enums.Reconstructors.PIE


@dataclasses.dataclass
class PIEObjectOptions(base.ObjectOptions):
    
    alpha: float = 0.1
    """
    Multiplier for the update to the object, as defined in table 1 of Maiden (2017).
    """


@dataclasses.dataclass
class PIEProbeOptions(base.ProbeOptions):
    
    alpha: float = 0.1
    """
    Multiplier for the update to the probe, as defined in table 1 of Maiden (2017).
    """


@dataclasses.dataclass
class PIEProbePositionOptions(base.ProbePositionOptions):
    pass


@dataclasses.dataclass
class PIEOPRModeWeightsOptions(base.OPRModeWeightsOptions):
    pass


@dataclasses.dataclass
class PIEOptions(task_options.PtychographyTaskOptions):
    
    reconstructor_options: PIEReconstructorOptions = field(default_factory=PIEReconstructorOptions)
    
    object_options: PIEObjectOptions = field(default_factory=PIEObjectOptions)
    
    probe_options: PIEProbeOptions = field(default_factory=PIEProbeOptions)
    
    probe_position_options: PIEProbePositionOptions = field(default_factory=PIEProbePositionOptions)
    
    opr_mode_weight_options: PIEOPRModeWeightsOptions = field(default_factory=PIEOPRModeWeightsOptions)


@dataclasses.dataclass
class EPIEReconstructorOptions(PIEReconstructorOptions):
    
    def get_reconstructor_type(self) -> enums.Reconstructors:
        return enums.Reconstructors.EPIE


@dataclasses.dataclass
class EPIEOptions(PIEOptions):

    reconstructor_options: EPIEReconstructorOptions = field(default_factory=EPIEReconstructorOptions)


@dataclasses.dataclass
class RPIEReconstructorOptions(PIEReconstructorOptions):
    
    def get_reconstructor_type(self) -> enums.Reconstructors:
        return enums.Reconstructors.RPIE


@dataclasses.dataclass
class RPIEOptions(PIEOptions):

    reconstructor_options: RPIEReconstructorOptions = field(default_factory=RPIEReconstructorOptions)