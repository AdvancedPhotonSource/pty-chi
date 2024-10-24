from typing import Optional
import dataclasses

import ptychi.data_structures.base as ds
import ptychi.data_structures.object as object
import ptychi.data_structures.opr_mode_weights as oprweights
import ptychi.data_structures.probe as probe
import ptychi.data_structures.probe_positions as probepos


@dataclasses.dataclass
class ParameterGroup:
    def get_all_parameters(self) -> list["ds.ReconstructParameter"]:
        return list(self.__dict__.values())

    def get_optimizable_parameters(self) -> list["ds.ReconstructParameter"]:
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

    opr_mode_weights: Optional["oprweights.OPRModeWeights"] = dataclasses.field(default_factory=ds.DummyParameter)

    def __post_init__(self):
        if self.probe.has_multiple_opr_modes and self.opr_mode_weights is None:
            raise ValueError(
                "OPRModeWeights must be provided when the probe has multiple OPR modes."
            )


@dataclasses.dataclass
class Ptychography2DParameterGroup(PtychographyParameterGroup):
    object: "object.Object2D"