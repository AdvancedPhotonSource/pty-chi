import dataclasses
from dataclasses import field

import ptychi.api.options.base as base
import ptychi.api.options.task as task_options
import ptychi.api.enums as enums


@dataclasses.dataclass
class LSQMLReconstructorOptions(base.ReconstructorOptions):
    
    noise_model: enums.NoiseModels = enums.NoiseModels.GAUSSIAN
    """
    The noise model to use.
    """
    
    gaussian_noise_std: float = 0.5
    """
    The standard deviation of the gaussian noise. Only used when `noise_model == enums.NoiseModels.GAUSSIAN`.
    """
    
    solve_obj_prb_step_size_jointly_for_first_slice_in_multislice: bool = False
    """
    Whether to solve the simultaneous object/probe step length calculation;
    in FoldSlice they use independent (non-joint) step length calculation, but 
    we're adding the option of using simultaneous AND non-simultaneous step 
    length calculation.
    """
    
    solve_step_sizes_only_using_first_probe_mode: bool = True
    """
    If True, object and probe step sizes will only be calculated using the first probe mode.
    This is how it is done in PtychoShelves.
    """
    
    momentum_acceleration_gain: float = 0.0
    """The gain of momentum acceleration. If 0, momentum acceleration is not used."""
    
    def get_reconstructor_type(self) -> enums.Reconstructors:
        return enums.Reconstructors.LSQML
    

@dataclasses.dataclass
class LSQMLObjectOptions(base.ObjectOptions):
    
    solved_step_size_upper_bound: float = None
    """
    The upper bound on the solved step size for object. If None,
    no upper bound is used.
    """


@dataclasses.dataclass
class LSQMLProbeOptions(base.ProbeOptions):
    
    solved_step_size_upper_bound: float = None
    """
    The upper bound on the solved step size for probe. If None,
    no upper bound is used.
    """
    
    eigenmode_update_relaxation: float = 0.1
    """
    A separate step size for eigenmode update. For now, this is only used by 
    LSQMLReconstructor.
    """
    
    
@dataclasses.dataclass
class LSQMLProbePositionOptions(base.ProbePositionOptions):
    pass


@dataclasses.dataclass
class LSQMLOPRModeWeightsOptions(base.OPRModeWeightsOptions):
    
    update_relaxation: float = 0.1
    """
    A separate step size for eigenmode weight update. For now, this is only used by
    LSQMLReconstructor.
    """
    

@dataclasses.dataclass
class LSQMLOptions(task_options.PtychographyTaskOptions):

    reconstructor_options: LSQMLReconstructorOptions = field(default_factory=LSQMLReconstructorOptions)
    
    object_options: LSQMLObjectOptions = field(default_factory=LSQMLObjectOptions)
    
    probe_options: LSQMLProbeOptions = field(default_factory=LSQMLProbeOptions)
    
    probe_position_options: LSQMLProbePositionOptions = field(default_factory=LSQMLProbePositionOptions)
    
    opr_mode_weight_options: LSQMLOPRModeWeightsOptions = field(default_factory=LSQMLOPRModeWeightsOptions)
        