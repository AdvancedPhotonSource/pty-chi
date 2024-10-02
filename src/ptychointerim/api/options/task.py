import dataclasses
from dataclasses import field

import ptychointerim.api.options.base as base
import ptychointerim.api.options.task as task_options
from .data import PtychographyDataOptions


@dataclasses.dataclass
class PtychographyTaskOptions(base.TaskOptions):

    data_options: PtychographyDataOptions = field(default_factory=PtychographyDataOptions)

    reconstructor_options: base.ReconstructorOptions = field(default_factory=base.ReconstructorOptions)

    object_options: base.ObjectOptions = field(default_factory=base.ObjectOptions)

    probe_options: base.ProbeOptions = field(default_factory=base.ProbeOptions)

    probe_position_options: base.ProbePositionOptions = field(default_factory=base.ProbePositionOptions)

    opr_mode_weight_options: base.OPRModeWeightsOptions = field(default_factory=base.OPRModeWeightsOptions)
