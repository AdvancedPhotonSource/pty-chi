# Copyright © 2025 UChicago Argonne, LLC All right reserved
# Full license accessible at https://github.com//AdvancedPhotonSource/pty-chi/blob/main/LICENSE

import dataclasses

import ptychi.data_structures.base as dsbase
import ptychi.data_structures.object as object
import ptychi.data_structures.opr_mode_weights as oprweights
import ptychi.data_structures.probe as probe
import ptychi.data_structures.probe_positions as probepos


@dataclasses.dataclass
class ParameterGroup:
    def get_all_parameters(self) -> list["dsbase.ReconstructParameter"]:
        return list(self.__dict__.values())

    def get_optimizable_parameters(self) -> list["dsbase.ReconstructParameter"]:
        ovs = []
        for var in self.get_all_parameters():
            if var.optimizable:
                ovs.append(var)
        return ovs

    def get_config_dict(self):
        return {var.name: var.get_config_dict() for var in self.get_all_parameters()}


@dataclasses.dataclass
class PtychographyParameterGroup(ParameterGroup):
    object: "object.Object"

    probe: "probe.Probe"

    probe_positions: "probepos.ProbePositions"

    opr_mode_weights: "oprweights.OPRModeWeights"

    def __post_init__(self):
        if self.probe.has_multiple_opr_modes and self.opr_mode_weights is None:
            raise ValueError(
                "OPRModeWeights must be provided when the probe has multiple OPR modes."
            )


@dataclasses.dataclass
class PlanarPtychographyParameterGroup(PtychographyParameterGroup):
    object: "object.PlanarObject"