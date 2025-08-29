# Copyright © 2025 UChicago Argonne, LLC All right reserved
# Full license accessible at https://github.com//AdvancedPhotonSource/pty-chi/blob/main/LICENSE

import dataclasses

import torch.distributed as dist

import ptychi.data_structures.base as dsbase
import ptychi.data_structures.object as object
import ptychi.data_structures.opr_mode_weights as oprweights
import ptychi.data_structures.probe as probe
import ptychi.data_structures.probe_positions as probepos
from ptychi.parallel import MultiprocessMixin


@dataclasses.dataclass
class ParameterGroup(MultiprocessMixin):
    def get_all_parameters(self) -> list["dsbase.ReconstructParameter"]:
        return list(self.__dict__.values())

    def get_optimizable_parameters(self) -> list["dsbase.ReconstructParameter"]:
        ovs = []
        for var in self.get_all_parameters():
            if var.optimizable:
                ovs.append(var)
        return ovs
    
    def synchronize_optimizable_parameter_gradients(
        self, op: dist.ReduceOp = dist.ReduceOp.SUM
    ) -> None:
        """Synchronize the gradients stored in the `grad` attribute
        of the optimizable parameters across ranks in a multi-process
        environment.
        """
        for param in self.get_optimizable_parameters():
            g = param.get_grad()
            if g is not None:
                self.sync_buffer(g, op=op)
            param.set_grad(g)
            
    def synchronize_optimizable_parameter_data(
        self, 
        op: dist.ReduceOp = dist.ReduceOp.SUM,
        source_rank: int | None = None
    ) -> None:
        """Synchronize the data stored in the `data` attribute
        of the optimizable parameters across ranks in a multi-process
        environment.
        
        Parameters
        ----------
        op : dist.ReduceOp, optional
            The operation to take on the data when they are synchronized across
            ranks through all-reduce. If broadcasting from a designated rank
            is intended, use anything for this argument.
        source_rank : int | None, optional
            If given, data will be broadcasted from the given rank to all ranks.
            Otherwise, data will be synchronized across all ranks through all-reduce.
        """
        for param in self.get_optimizable_parameters():
            d = param.data
            if source_rank is not None:
                self.sync_buffer(d, source_rank=source_rank)
            else:
                self.sync_buffer(d, op=op)
            param.set_data(d)


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
