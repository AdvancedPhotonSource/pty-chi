# Copyright © 2025 UChicago Argonne, LLC All right reserved
# Full license accessible at https://github.com//AdvancedPhotonSource/pty-chi/blob/main/LICENSE

import dataclasses
from dataclasses import field
from pydantic import Field as PydanticField
from pydantic.dataclasses import dataclass

import ptychi.api.options.base as base
import ptychi.api.options.task as task_options
import ptychi.api.enums as enums


@dataclass
class PIEReconstructorOptions(base.ReconstructorOptions):

    displayed_loss_function: enums.LossFunctions | None = None
    """Displayed metric; None uses the magnitude mean squared noise-model loss."""
    
    def get_reconstructor_type(self) -> enums.Reconstructors:
        return enums.Reconstructors.PIE


@dataclass
class PIEObjectOptions(base.ObjectOptions):
    
    alpha: float = PydanticField(default=0.1, ge=0)
    """
    Multiplier for the update to the object, as defined in table 1 of Maiden (2017).
    """


@dataclass
class PIEProbeExperimentalOptions(base.Options):
    sdl_probe_options: base.SynthesisDictLearnProbeOptions = dataclasses.field(default_factory=base.SynthesisDictLearnProbeOptions)


@dataclass
class PIEProbeOptions(base.ProbeOptions):
    
    alpha: float = PydanticField(default=0.1, ge=0)
    """
    Multiplier for the update to the probe, as defined in table 1 of Maiden (2017).
    """
    
    experimental: PIEProbeExperimentalOptions = dataclasses.field(default_factory=PIEProbeExperimentalOptions)


@dataclass
class PIEProbePositionOptions(base.ProbePositionOptions):
    pass


@dataclass
class PIEOPRModeWeightsOptions(base.OPRModeWeightsOptions):
    pass


@dataclass
class PIEOptions(task_options.PtychographyTaskOptions):
    
    reconstructor_options: PIEReconstructorOptions = field(default_factory=PIEReconstructorOptions)
    
    object_options: PIEObjectOptions = field(default_factory=PIEObjectOptions)
    
    probe_options: PIEProbeOptions = field(default_factory=PIEProbeOptions)
    
    probe_position_options: PIEProbePositionOptions = field(default_factory=PIEProbePositionOptions)
    
    opr_mode_weight_options: PIEOPRModeWeightsOptions = field(default_factory=PIEOPRModeWeightsOptions)


@dataclass
class EPIEReconstructorOptions(PIEReconstructorOptions):
    
    def get_reconstructor_type(self) -> enums.Reconstructors:
        return enums.Reconstructors.EPIE


@dataclass
class EPIEOptions(PIEOptions):

    reconstructor_options: EPIEReconstructorOptions = field(default_factory=EPIEReconstructorOptions)


@dataclass
class RPIEReconstructorOptions(PIEReconstructorOptions):
    
    def get_reconstructor_type(self) -> enums.Reconstructors:
        return enums.Reconstructors.RPIE


@dataclass
class RPIEOptions(PIEOptions):

    reconstructor_options: RPIEReconstructorOptions = field(default_factory=RPIEReconstructorOptions)
