# Copyright © 2025 UChicago Argonne, LLC All right reserved
# Full license accessible at https://github.com//AdvancedPhotonSource/pty-chi/blob/main/LICENSE

from typing import Optional
import dataclasses
from dataclasses import field
import logging

import ptychi.api.options.base as base
import ptychi.api.options.task as task_options
import ptychi.api.enums as enums

logger = logging.getLogger(__name__)


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
    
    momentum_acceleration_gradient_mixing_factor: Optional[float] = 1
    """
    Controls how the current gradient is mixed with the accumulated velocity in LSQML
    momentum acceleration:
    
    `velocity = (1 - friction) * velocity + momentum_acceleration_gradient_mixing_factor * delta_o`
    
    If None, this mixing factor is automatically chosen to be `friction`:
    
    `velocity = (1 - friction) * velocity + friction * delta_o`
    
    Using `None` usually provides better stability. However, it may cause the speed of convergence to be
    too slow in some cases. Set this parameter to 1 to reproduce the behavior in PtychoShelves.
    """
    
    rescale_probe_intensity_in_first_epoch: bool = True
    """
    If True, probe intensity is rescaled in the first epoch using the average intensity of all
    diffraction patterns. Set this to False if you want the probe intensity to stay constant.
    You may also want to check `ObjectOptions.remove_object_probe_ambiguity`.
    """
    
    preconditioning_damping_factor: float = 0.1
    """
    The damping factor for applying preconditioning to the object update, which is calculated as::
    
        delta_o_hat = delta_o_hat / torch.sqrt(preconditioner ** 2 + (preconditioner.max() * mixing_factor) ** 2)
    """
    
    def check(self, options: "LSQMLOptions"):
        super().check(options)
        if self.rescale_probe_intensity_in_first_epoch:
            if options.probe_options.power_constraint.enabled:
                logger.warning(
                    "`rescale_probe_intensity_in_first_epoch` and `ProbeOptions.power_constraint` "
                    "are both enabled, which may lead to unexpected results."
                )
    
    def get_reconstructor_type(self) -> enums.Reconstructors:
        return enums.Reconstructors.LSQML
    

@dataclasses.dataclass
class LSQMLObjectOptions(base.ObjectOptions):
    
    optimal_step_size_scaler: float = 0.9
    """
    A scaler for the solved optimal step size (beta_LSQ in PtychoShelves).
    """
    
    multimodal_update: bool = True
    """
    If True, object update direction is calculated and summed over all probe modes. 
    Otherwise, only the first mode will be used for object update. However, forward
    propagation always uses all probe modes regardless of this option.
    """


@dataclasses.dataclass
class LSQMLProbeOptions(base.ProbeOptions):
    optimal_step_size_scaler: float = 0.9
    """
    A scaler for the solved optimal step size (beta_LSQ in PtychoShelves).
    """
    
    
@dataclasses.dataclass
class LSQMLProbePositionOptions(base.ProbePositionOptions):
    pass


@dataclasses.dataclass
class LSQMLOPRModeWeightsOptions(base.OPRModeWeightsOptions):
    pass


@dataclasses.dataclass
class LSQMLOptions(task_options.PtychographyTaskOptions):

    reconstructor_options: LSQMLReconstructorOptions = field(default_factory=LSQMLReconstructorOptions)
    
    object_options: LSQMLObjectOptions = field(default_factory=LSQMLObjectOptions)
    
    probe_options: LSQMLProbeOptions = field(default_factory=LSQMLProbeOptions)
    
    probe_position_options: LSQMLProbePositionOptions = field(default_factory=LSQMLProbePositionOptions)
    
    opr_mode_weight_options: LSQMLOPRModeWeightsOptions = field(default_factory=LSQMLOPRModeWeightsOptions)
