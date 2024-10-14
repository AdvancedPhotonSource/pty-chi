import dataclasses
from dataclasses import field


import ptychointerim.api.options.base as base
import ptychointerim.api.options.task as task_options
import ptychointerim.api.enums as enums


@dataclasses.dataclass
class AutodiffPtychographyReconstructorOptions(base.ReconstructorOptions):
        
    loss_function: enums.LossFunctions = enums.LossFunctions.MSE_SQRT
    """
    The loss function.
    """
    
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
    
    reconstructor_options: AutodiffPtychographyReconstructorOptions = field(default_factory=AutodiffPtychographyReconstructorOptions)
    
    object_options: AutodiffPtychographyObjectOptions = field(default_factory=AutodiffPtychographyObjectOptions)
    
    probe_options: AutodiffPtychographyProbeOptions = field(default_factory=AutodiffPtychographyProbeOptions)
    
    probe_position_options: AutodiffPtychographyProbePositionOptions = field(default_factory=AutodiffPtychographyProbePositionOptions)
    
    opr_mode_weight_options: AutodiffPtychographyOPRModeWeightsOptions = field(default_factory=AutodiffPtychographyOPRModeWeightsOptions)
