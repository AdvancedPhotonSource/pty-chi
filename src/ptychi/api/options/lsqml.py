# Copyright © 2025 UChicago Argonne, LLC All right reserved
# Full license accessible at https://github.com//AdvancedPhotonSource/pty-chi/blob/main/LICENSE

from typing import Optional
from dataclasses import field
import logging
from pydantic import Field as PydanticField
from pydantic.dataclasses import dataclass

import ptychi.api.options.base as base
import ptychi.api.options.task as task_options
import ptychi.api.enums as enums

logger = logging.getLogger(__name__)


@dataclass
class LSQMLReconstructorOptions(base.ReconstructorOptions):
    
    noise_model: enums.NoiseModels = enums.NoiseModels.GAUSSIAN
    """
    The noise model to use.
    """
    
    gaussian_noise_std: float = PydanticField(default=0.5, gt=0)
    """
    The standard deviation of the gaussian noise. Only used when `noise_model == enums.NoiseModels.GAUSSIAN`.
    """
    
    single_slice_solve_obj_prb_step_size_jointly: bool = True
    """
    Whether to solve the object/probe step size jointly for single-slice objects.
    For multislice objects, use `multislice_solve_obj_prb_step_size_jointly` instead. 
    """

    multislice_solve_obj_prb_step_size_jointly: bool = False
    """
    Whether to solve the object/probe step size jointly for multislice objects at the first slice.
    Slices other than the first are always solved independently. For single-slice objects, use
    `single_slice_solve_obj_prb_step_size_jointly` instead.
    """

    obj_prb_joint_step_size_method: enums.ObjProbeJointStepSizeMethods = (
        enums.ObjProbeJointStepSizeMethods.PTYCHOSHELVES
    )
    """
    How the 2x2 linear system is built when the object/probe step sizes are solved jointly
    (see `single_slice_solve_obj_prb_step_size_jointly` and
    `multislice_solve_obj_prb_step_size_jointly`):

    - `PTYCHOSHELVES`: Eq. 22 of Odstrcil (2018) as printed. The off-diagonal terms are complex,
      the diagonal is damped by 0.1 times its batch mean, the real part of the complex solution
      is taken, and negative step sizes are clipped to 0.
    - `REAL_LSQ`: least squares over real step sizes, so every matrix entry is real and the
      off-diagonal terms are the real parts of the cross terms. The diagonal is damped by 0.02
      times its batch mean, and negative step sizes are clipped to 0.
    """

    solve_step_sizes_only_using_first_probe_mode: bool = True
    """
    If True, object and probe step sizes will only be calculated using the first probe mode.
    This is how it is done in PtychoShelves.
    """
    
    momentum_acceleration_gain: float = PydanticField(default=0.0, ge=0)
    """The gain of momentum acceleration for object and probe. If 0, momentum acceleration is not used."""
    
    momentum_acceleration_gradient_mixing_factor: Optional[float] = PydanticField(
        default=1, ge=0
    )
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
    
    preconditioning_damping_factor: float = PydanticField(default=0.1, ge=0)
    """
    The damping factor for applying preconditioning to the object update, which is calculated as::
    
        delta_o_hat = delta_o_hat / torch.sqrt(preconditioner ** 2 + (preconditioner.max() * mixing_factor) ** 2)
    """
    
    def check(self, options: "LSQMLOptions"):
        super().check(options)
        if options.object_options.optimizable:
            # TODO: Remove this warning in 2.8.0 (two releases after 2.6.0).
            logger.warning(
                "Pty-Chi 2.6.0 initializes the LSQML object preconditioner with the "
                "probe illumination map and uses relative damping in the joint "
                "step-size solve. Existing `preconditioning_damping_factor` and SGD "
                "step sizes generally need no conversion, but iteration histories "
                "can change for all geometries, including angular-spectrum runs. "
                "See https://github.com/AdvancedPhotonSource/pty-chi/discussions/90"
            )
        if self.rescale_probe_intensity_in_first_epoch:
            if options.probe_options.power_constraint.enabled:
                logger.warning(
                    "`rescale_probe_intensity_in_first_epoch` and `ProbeOptions.power_constraint` "
                    "are both enabled, which may lead to unexpected results."
                )
        if self.batching_mode == enums.BatchingModes.COMPACT and self.momentum_acceleration_gain == 0:
            logger.warning(
                "`batching_mode` is set to COMPACT but `momentum_acceleration_gain` is 0. "
                "Momentum acceleration is strongly recommended for compact batching mode. "
                "Start with 0.5."
            )
    
    def get_reconstructor_type(self) -> enums.Reconstructors:
        return enums.Reconstructors.LSQML
    

@dataclass
class LSQMLObjectOptions(base.ObjectOptions):
    
    optimal_step_size_scaler: float = PydanticField(default=0.9, gt=0)
    """
    A scaler for the solved optimal step size (beta_LSQ in PtychoShelves).
    """
    
    multimodal_update: bool = True
    """
    If True, object update direction is calculated and summed over all probe modes. 
    Otherwise, only the first mode will be used for object update. However, forward
    propagation always uses all probe modes regardless of this option.
    """


@dataclass
class LSQMLProbeOptions(base.ProbeOptions):
    optimal_step_size_scaler: float = PydanticField(default=0.9, gt=0)
    """
    A scaler for the solved optimal step size (beta_LSQ in PtychoShelves).
    """
    
    
@dataclass
class LSQMLProbePositionOptions(base.ProbePositionOptions):
    momentum_acceleration_gain: float = PydanticField(default=0.0, ge=0)
    """
    The gain of momentum acceleration for probe positions. If 0, momentum
    acceleration is not used.
    """

    momentum_acceleration_gradient_mixing_factor: Optional[float] = PydanticField(
        default=1, ge=0
    )
    """
    Controls how the current position update is mixed with the accumulated
    velocity in probe-position momentum acceleration:

    `velocity = (1 - friction) * velocity + momentum_acceleration_gradient_mixing_factor * delta_pos`

    If None, this mixing factor is automatically chosen to be `friction`.
    Set this parameter to 1 to reproduce the behavior in foldslice.
    """

    momentum_acceleration_memory: int = PydanticField(default=3, ge=1)
    """
    Number of previous epochs used to estimate the friction of probe-position
    momentum acceleration.
    """

    def check(self, options: "LSQMLOptions"):
        super().check(options)
        if self.momentum_acceleration_gain > 0 and self.momentum_acceleration_memory < 1:
            raise ValueError(
                "`probe_position_options.momentum_acceleration_memory` must be positive "
                "when probe-position momentum acceleration is enabled."
            )


@dataclass
class LSQMLOPRModeWeightsOptions(base.OPRModeWeightsOptions):
    pass


@dataclass
class LSQMLOptions(task_options.PtychographyTaskOptions):

    reconstructor_options: LSQMLReconstructorOptions = field(default_factory=LSQMLReconstructorOptions)
    
    object_options: LSQMLObjectOptions = field(default_factory=LSQMLObjectOptions)
    
    probe_options: LSQMLProbeOptions = field(default_factory=LSQMLProbeOptions)
    
    probe_position_options: LSQMLProbePositionOptions = field(default_factory=LSQMLProbePositionOptions)
    
    opr_mode_weight_options: LSQMLOPRModeWeightsOptions = field(default_factory=LSQMLOPRModeWeightsOptions)
