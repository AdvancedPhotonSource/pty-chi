from typing import Optional, Type, Union
import dataclasses
from dataclasses import field


import ptychi.api.options.base as base
import ptychi.api.options.task as task_options
import ptychi.api.enums as enums
import ptychi.api.options.ad_general as ad_general
import ptychi.forward_models as fm


@dataclasses.dataclass
class AutodiffPtychographyReconstructorOptions(ad_general.AutodiffReconstructorOptions):
    forward_model_class: Union["enums.ForwardModels", Type["fm.ForwardModel"]] = enums.ForwardModels.PLANAR_PTYCHOGRAPHY
    
    def get_reconstructor_type(self) -> enums.Reconstructors:
        return enums.Reconstructors.AD_PTYCHO


@dataclasses.dataclass
class AutodiffPtychographyObjectOptions(base.ObjectOptions):
    pass


@dataclasses.dataclass
class AutodiffPtychographyProbeOptions(base.ProbeOptions):
    pass


@dataclasses.dataclass
class AutodiffPtychographyProbePositionOptions(base.ProbePositionOptions):
    pass


@dataclasses.dataclass
class AutodiffPtychographyOPRModeWeightsOptions(base.OPRModeWeightsOptions):
    pass


@dataclasses.dataclass
class AutodiffPtychographyOptions(task_options.PtychographyTaskOptions):
    reconstructor_options: AutodiffPtychographyReconstructorOptions = field(
        default_factory=AutodiffPtychographyReconstructorOptions
    )

    object_options: AutodiffPtychographyObjectOptions = field(
        default_factory=AutodiffPtychographyObjectOptions
    )

    probe_options: AutodiffPtychographyProbeOptions = field(
        default_factory=AutodiffPtychographyProbeOptions
    )

    probe_position_options: AutodiffPtychographyProbePositionOptions = field(
        default_factory=AutodiffPtychographyProbePositionOptions
    )

    opr_mode_weight_options: AutodiffPtychographyOPRModeWeightsOptions = field(
        default_factory=AutodiffPtychographyOPRModeWeightsOptions
    )
