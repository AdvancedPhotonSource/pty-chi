# Copyright © 2025 UChicago Argonne, LLC All right reserved
# Full license accessible at https://github.com//AdvancedPhotonSource/pty-chi/blob/main/LICENSE

import dataclasses
from dataclasses import field

import ptychi.api.options.base as base
import ptychi.api.options.task as task_options
import ptychi.api.enums as enums


@dataclasses.dataclass
class BHReconstructorOptions(base.ReconstructorOptions):
    method: str = "GD"
    """
    Reconstruction algorithm: 'GD' (Gradient Descent) or 'CG' (Conjugate Gradients). 
    Batch processing should be performed using the GD method, while CG is significantly 
    faster when using a single batch, i.e., when all positions are included in the update.
    """

    def get_reconstructor_type(self) -> enums.Reconstructors:
        return enums.Reconstructors.BH


@dataclasses.dataclass
class BHObjectOptions(base.ObjectOptions):
    pass


@dataclasses.dataclass
class BHProbeOptions(base.ProbeOptions):
    rho: float = 1
    """   
    Scaling factor for the probe variable relative to the object variable.
    It addresses numerical instabilities caused by disparities in gradient magnitudes between variables, 
    ensuring balanced convergence. The optimal value of rho is currently determined experimentally, 
    based on how closely the initial guess aligns with the final solution.
    For near-field ptychography, a typical value is 1, while for far-field ptychography, it is 0.1.
    """


@dataclasses.dataclass
class BHProbePositionOptions(base.ProbePositionOptions):
    rho: float = 0.1
    """   
    Scaling factor for the probe positions relative to the object variable.
    It addresses numerical instabilities caused by disparities in gradient magnitudes between variables, 
    ensuring balanced convergence. The optimal value of rho is currently determined experimentally, 
    based on how closely the initial guess aligns with the final solution.
    For near-field ptychography, a typical value is 0.1, while for far-field ptychography, it is 1.
    """


@dataclasses.dataclass
class BHOPRModeWeightsOptions(base.OPRModeWeightsOptions):
    pass


@dataclasses.dataclass
class BHOptions(task_options.PtychographyTaskOptions):
    reconstructor_options: BHReconstructorOptions = field(default_factory=BHReconstructorOptions)

    object_options: BHObjectOptions = field(default_factory=BHObjectOptions)

    probe_options: BHProbeOptions = field(default_factory=BHProbeOptions)

    probe_position_options: BHProbePositionOptions = field(default_factory=BHProbePositionOptions)

    opr_mode_weight_options: BHOPRModeWeightsOptions = field(
        default_factory=BHOPRModeWeightsOptions
    )
