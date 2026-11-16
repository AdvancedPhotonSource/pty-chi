# Copyright © 2025 UChicago Argonne, LLC All right reserved
# Full license accessible at https://github.com//AdvancedPhotonSource/pty-chi/blob/main/LICENSE

from dataclasses import field
import logging
from pydantic import Field as PydanticField
from pydantic.dataclasses import dataclass

import ptychi.api.options.base as base
import ptychi.api.options.task as task_options
import ptychi.api.enums as enums


logger = logging.getLogger(__name__)


@dataclass
class BHReconstructorOptions(base.ReconstructorOptions):
    method: str = "GD"
    """
    Reconstruction algorithm: 'GD' (Gradient Descent) or 'CG' (Conjugate Gradients). 
    Batch processing should be performed using the GD method, while CG is significantly 
    faster when using a single batch, i.e., when all positions are included in the update.
    """

    def get_reconstructor_type(self) -> enums.Reconstructors:
        return enums.Reconstructors.BH


@dataclass
class BHObjectOptions(base.ObjectOptions):
    pass


@dataclass
class BHProbeOptions(base.ProbeOptions):
    rho: float = PydanticField(default=1, gt=0)
    """   
    Scaling factor for the probe variable relative to the object variable.
    It addresses numerical instabilities caused by disparities in gradient magnitudes between variables, 
    ensuring balanced convergence. The optimal value of rho is currently determined experimentally, 
    based on how closely the initial guess aligns with the final solution.
    For near-field ptychography, a typical value is 1, while for far-field ptychography, it is 0.1.
    """

    def check(self, options: "BHOptions"):
        super().check(options)
        if self.optimizable:
            # TODO: Remove this warning in 2.8.0 (two releases after 2.6.0).
            logger.warning(
                "Pty-Chi 2.6.0 changed the scaling of `probe_options.rho` in BH. "
                "When migrating a tuned value from the old default FourierPropagator, "
                "multiply it by sqrt(H*W), matching the probe-amplitude conversion. "
                "Fresnel/Fraunhofer conversion depends on geometry and may require "
                "retuning; angular-spectrum units are unchanged. "
                "See https://github.com/AdvancedPhotonSource/pty-chi/discussions/90"
            )


@dataclass
class BHProbePositionOptions(base.ProbePositionOptions):
    rho: float = PydanticField(default=0.1, gt=0)
    """   
    Scaling factor for the probe positions relative to the object variable.
    It addresses numerical instabilities caused by disparities in gradient magnitudes between variables, 
    ensuring balanced convergence. The optimal value of rho is currently determined experimentally, 
    based on how closely the initial guess aligns with the final solution.
    For near-field ptychography, a typical value is 0.1, while for far-field ptychography, it is 1.
    """


@dataclass
class BHOPRModeWeightsOptions(base.OPRModeWeightsOptions):
    pass


@dataclass
class BHOptions(task_options.PtychographyTaskOptions):
    reconstructor_options: BHReconstructorOptions = field(default_factory=BHReconstructorOptions)

    object_options: BHObjectOptions = field(default_factory=BHObjectOptions)

    probe_options: BHProbeOptions = field(default_factory=BHProbeOptions)

    probe_position_options: BHProbePositionOptions = field(default_factory=BHProbePositionOptions)

    opr_mode_weight_options: BHOPRModeWeightsOptions = field(
        default_factory=BHOPRModeWeightsOptions
    )
