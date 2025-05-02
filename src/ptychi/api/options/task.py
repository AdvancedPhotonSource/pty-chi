# Copyright © 2025 UChicago Argonne, LLC All right reserved
# Full license accessible at https://github.com//AdvancedPhotonSource/pty-chi/blob/main/LICENSE

import dataclasses
from dataclasses import field

import ptychi.api.options.base as base
from .data import PtychographyDataOptions


@dataclasses.dataclass
class PtychographyTaskOptions(base.TaskOptions):

    data_options: PtychographyDataOptions = field(default_factory=PtychographyDataOptions)

    reconstructor_options: base.ReconstructorOptions = field(default_factory=base.ReconstructorOptions)

    object_options: base.ObjectOptions = field(default_factory=base.ObjectOptions)

    probe_options: base.ProbeOptions = field(default_factory=base.ProbeOptions)

    probe_position_options: base.ProbePositionOptions = field(default_factory=base.ProbePositionOptions)

    opr_mode_weight_options: base.OPRModeWeightsOptions = field(default_factory=base.OPRModeWeightsOptions)

    def check(self, *args, **kwargs):
        super().check(*args, **kwargs)
        for options in (
            self.data_options,
            self.reconstructor_options,
            self.object_options,
            self.probe_options,
            self.probe_position_options,
            self.opr_mode_weight_options,
        ):
            options.check(self)
