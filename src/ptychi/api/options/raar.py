# Copyright © 2025 UChicago Argonne, LLC All right reserved
# Full license accessible at https://github.com//AdvancedPhotonSource/pty-chi/blob/main/LICENSE

from dataclasses import field

from pydantic import Field as PydanticField
from pydantic.dataclasses import dataclass

import ptychi.api.enums as enums
import ptychi.api.options.base as base
import ptychi.api.options.task as task_options


@dataclass
class RAARReconstructorOptions(base.ReconstructorOptions):
    def get_reconstructor_type(self) -> enums.Reconstructors:
        return enums.Reconstructors.RAAR

    beta: float = PydanticField(default=0.75, gt=0.5, le=1)
    """RAAR relaxation parameter from equation (16) of Marchesini et al. (2016)."""

    chunk_length: int = PydanticField(default=1, ge=1)
    """Number of scan points processed together during exit-wave projections.
    Smaller values use less temporary memory, but can be slower."""


@dataclass
class RAARObjectOptions(base.ObjectOptions):
    amplitude_clamp_limit: float = PydanticField(default=1000, gt=0)
    """Maximum allowed amplitude for the object reconstruction."""

    inertia: float = PydanticField(default=0, ge=0, le=1)
    """Inertia of the object retrieval update."""


@dataclass
class RAARProbeOptions(base.ProbeOptions):
    inertia: float = PydanticField(default=0, ge=0, le=1)
    """Inertia of the probe retrieval update."""


@dataclass
class RAARProbePositionOptions(base.ProbePositionOptions):
    pass


@dataclass
class RAAROPRModeWeightsOptions(base.OPRModeWeightsOptions):
    pass


@dataclass
class RAAROptions(task_options.PtychographyTaskOptions):
    reconstructor_options: RAARReconstructorOptions = field(
        default_factory=RAARReconstructorOptions
    )

    object_options: RAARObjectOptions = field(default_factory=RAARObjectOptions)

    probe_options: RAARProbeOptions = field(default_factory=RAARProbeOptions)

    probe_position_options: RAARProbePositionOptions = field(
        default_factory=RAARProbePositionOptions
    )

    opr_mode_weight_options: RAAROPRModeWeightsOptions = field(
        default_factory=RAAROPRModeWeightsOptions
    )
