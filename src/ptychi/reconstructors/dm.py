# Copyright © 2025 UChicago Argonne, LLC All right reserved
# Full license accessible at https://github.com//AdvancedPhotonSource/pty-chi/blob/main/LICENSE

from typing import Optional, TYPE_CHECKING, Tuple, Union
import logging

import torch
from torch.utils.data import Dataset
from torch import Tensor
from torch.utils.data import DataLoader
import math

import ptychi.api as api
from ptychi.reconstructors.base import (
    AnalyticalIterativePtychographyReconstructor,
    LossTracker,
)
from ptychi.api.options.dm import DMReconstructorOptions
from ptychi.timing.timer_utils import timer
import ptychi.image_proc as ip

if TYPE_CHECKING:
    import ptychi.data_structures.parameter_group as pg

logger = logging.getLogger(__name__)


class DMReconstructor(AnalyticalIterativePtychographyReconstructor):
    """
    Difference map algorithm reconstructor class

    ### On memory usage
    The difference map algorithm takes up more memory than algorithms like LSQML
    and PIE because the entire exit wave `self.psi` must be stored in memory.

    For the least amount of memory usage, set the `chunk_length` option to 1.
    `chunk_length` has no effect on the convergence.

    ### On batching
    The `batch_size` option is not used by this reconstructor.
    """

    parameter_group: "pg.PlanarPtychographyParameterGroup"

    def __init__(
        self,
        parameter_group: "pg.PlanarPtychographyParameterGroup",
        dataset: Dataset,
        options: Optional["api.options.dm.DMReconstructorOptions"] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            parameter_group=parameter_group,
            dataset=dataset,
            options=options,
            *args,
            **kwargs,
        )
        self.forward_model.retain_intermediates = False
        self.options: DMReconstructorOptions = self.options

    def check_inputs(self, *args, **kwargs):
        if self.parameter_group.object.is_multislice:
            raise NotImplementedError("DMReconstructor only supports 2D objects.")
        if self.parameter_group.probe.has_multiple_opr_modes:
            raise NotImplementedError("DMReconstructor does not support multiple OPR modes yet.")
        if (
            self.parameter_group.probe_positions.position_correction.options.correction_type
            is not api.enums.PositionCorrectionTypes.GRADIENT
        ):
            raise NotImplementedError("DMReconstructor only supports gradient position correction.")
        if self.options.batch_size != DMReconstructorOptions.batch_size:
            logger.warning("Difference map reconstruction does not support batching!")

    def build_loss_tracker(self):
        if self.options.displayed_loss_function is not None:
            logger.warning(
                "The loss tracker is hard-coded to record the DM error. "
                "The specified metric function will not be used!"
            )
        self.loss_tracker = LossTracker(metric_function=None)

    def build_dataloader(self):
        """
        Build the dataloader for the DM reconstructor. There is no
        batching in this reconstructor. Each batch is always a full,
        sequential sampling of the dataset.
        """
        data_loader_kwargs = {
            "dataset": self.dataset,
            "generator": torch.Generator(device=torch.get_default_device()),
            "batch_size": self.parameter_group.probe_positions.n_scan_points,
            "shuffle": False,
        }
        self.dataloader = DataLoader(**data_loader_kwargs)
        self.dataset.move_attributes_to_device(torch.get_default_device())

    @timer()
    def run_minibatch(self, input_data, y_true, *args, **kwargs):
        dm_error_squared = self.compute_updates(y_true, self.dataset.valid_pixel_mask)
        self.loss_tracker.update_batch_loss(loss=dm_error_squared.sqrt())

    @timer()
    def compute_updates(self, y_true: Tensor, valid_pixel_mask: Tensor) -> Tensor:
        """
        Compute the updates to the object, probe, and exit wave using the procedure
        described here: [Probe retrieval in ptychographic coherent diffractive imaging
        ](https://doi.org/10.1016/j.ultramic.2008.12.011).
        """

        object_ = self.parameter_group.object
        probe = self.parameter_group.probe
        probe_positions = self.parameter_group.probe_positions

        # Get indices used for dividing data into chunks
        start_pts = list(range(0, probe_positions.n_scan_points, self.options.chunk_length))
        end_pts = list(
            range(
                self.options.chunk_length,
                probe_positions.n_scan_points + self.options.chunk_length,
                self.options.chunk_length,
            )
        )
        end_pts[-1] = min(end_pts[-1], probe_positions.n_scan_points)
        n_chunks = math.ceil(probe_positions.n_scan_points / self.options.chunk_length)

        # Initialize the exit wave
        if self.current_epoch == 0:
            self.psi = torch.zeros(
                (
                    probe_positions.n_scan_points,
                    probe.n_modes,
                    *probe.get_spatial_shape(),
                ),
                dtype=object_.data.dtype,
            )
            for i in range(n_chunks):
                self.psi[start_pts[i] : end_pts[i]] = self.calculate_exit_wave_chunk(
                    start_pts[i], end_pts[i]
                )

        # Calculate the dm exit wave and probe update in chunks
        probe_numerator = torch.zeros_like(probe.get_opr_mode(0))
        probe_denominator = torch.zeros_like(probe.get_opr_mode(0).abs())
        delta_pos = torch.zeros_like(probe_positions.data)
        dm_error_squared = 0
        for i in range(n_chunks):
            obj_patches, dm_error_squared, new_psi = self.apply_dm_update_to_exit_wave_chunk(
                start_pts[i], end_pts[i], y_true, valid_pixel_mask, dm_error_squared
            )
            if probe.optimization_enabled(self.current_epoch):
                self.add_to_probe_update_terms(
                    probe_numerator,
                    probe_denominator,
                    obj_patches,
                    start_pts[i],
                    end_pts[i],
                )
            if probe_positions.optimization_enabled(self.current_epoch):
                delta_pos[start_pts[i] : end_pts[i]] = self.get_positions_update_chunk(
                    indices=torch.Tensor(list(range(start_pts[i], end_pts[i]))).to(int),
                    obj_patches=obj_patches,
                    chi=self.psi[start_pts[i] : end_pts[i]] - new_psi,
                )

        # Update the probe
        if probe.optimization_enabled(self.current_epoch):
            probe_update = self.calculate_probe_update(probe_numerator, probe_denominator)
            self.update_probe(probe_update)

        # Update the object
        if object_.optimization_enabled(self.current_epoch):
            updated_object = self.calculate_object_update(start_pts, end_pts)
            self.update_object(updated_object)

        if probe_positions.optimization_enabled(self.current_epoch):
            probe_positions.set_grad(-delta_pos)
            probe_positions.step_optimizer()

        return dm_error_squared

    @timer()
    def calculate_probe_update(self, probe_numerator: Tensor, probe_denominator: Tensor) -> Tensor:
        probe = self.parameter_group.probe
        probe_update = probe_numerator / torch.sqrt(
            probe_denominator**2 + (0.05 * probe_denominator.max()) ** 2
        )
        probe_update = (
            probe.options.inertia * probe.data + (1 - probe.options.inertia) * probe_update
        )
        return probe_update

    @timer()
    def update_probe(self, probe_update: Tensor):
        probe = self.parameter_group.probe
        probe.set_data(probe_update)

    @timer()
    def calculate_exit_wave_chunk(
        self, start_pt: int, end_pt: int, return_obj_patches: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        object_ = self.parameter_group.object
        probe = self.parameter_group.probe
        positions = self.parameter_group.probe_positions.tensor

        obj_patches = object_.extract_patches(
            positions[start_pt:end_pt].round().int(), probe.get_spatial_shape(), integer_mode=True
        )
        psi = self.forward_model.forward_real_space(
            indices=torch.arange(start_pt, end_pt, device=obj_patches.device).long(),
            obj_patches=obj_patches,
        )

        if return_obj_patches:
            return psi, obj_patches
        else:
            return psi

    @timer()
    def apply_dm_update_to_exit_wave_chunk(
        self,
        start_pt: int,
        end_pt: int,
        y_true: Tensor,
        valid_pixel_mask: Tensor,
        dm_error_squared: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Calculate the updated exit wave according to equation (9) in
        this paper: https://doi.org/10.1016/j.ultramic.2008.12.011
        """
        # Mappings between this code and the paper:
        # - self.psi --> psi_n
        # - revised_psi --> 2 * Pi_o(psi_n)- psi_n
        # - new_psi --> Pi_o(psi_n)

        probe = self.parameter_group.probe

        # Get the update exit wave
        new_psi, obj_patches = self.calculate_exit_wave_chunk(
            start_pt, end_pt, return_obj_patches=True
        )
        # Propagate to detector plane
        revised_psi = self.forward_model.free_space_propagator.propagate_forward(
            2 * new_psi - self.psi[start_pt:end_pt]
        )
        # Replace intensities
        revised_psi = torch.where(
            valid_pixel_mask.repeat(revised_psi.shape[0], probe.n_modes, 1, 1),
            self.replace_propagated_exit_wave_magnitude(revised_psi, y_true[start_pt:end_pt]),
            revised_psi,
        )
        # Propagate back to sample plane
        revised_psi = self.forward_model.free_space_propagator.propagate_backward(revised_psi)
        # Update the exit wave
        psi_update = (revised_psi - new_psi) * self.options.exit_wave_update_relaxation
        self.psi[start_pt:end_pt] += psi_update

        dm_error_squared += (psi_update.abs() ** 2).sum()

        return obj_patches, dm_error_squared, new_psi

    @timer()
    def add_to_probe_update_terms(
        self,
        probe_numerator: Tensor,
        probe_denominator: Tensor,
        obj_patches: Tensor,
        start_pt: int,
        end_pt: int,
    ):
        "Add to the running totals for the probe update numerator and denominator"
        indices = torch.arange(start_pt, end_pt, device=probe_numerator.device).long()
        numerator_update = obj_patches.conj() * self.psi[start_pt:end_pt]
        denominator_update = obj_patches.abs() ** 2
        # Before summing in the batch dimension, the updates should be adjointly shifted to backpropagate
        # through the subpixel shifts of the probe.
        numerator_update = self.adjoint_shift_probe_update_direction(
            indices, numerator_update, first_mode_only=True
        )
        denominator_update = self.adjoint_shift_probe_update_direction(
            indices, denominator_update, first_mode_only=True
        )
        numerator_update = numerator_update.sum(0)
        denominator_update = denominator_update.sum(0)
        probe_numerator += numerator_update
        probe_denominator += denominator_update

    @timer()
    def calculate_object_update(self, start_pts: list[int], end_pts: list[int]) -> Tensor:
        "Calculate and apply the object update."

        object_ = self.parameter_group.object
        positions = self.parameter_group.probe_positions.tensor

        # Calculate object update
        # Iterating over the object numerator calculation is more memory efficient
        object_numerator = torch.zeros_like(object_.get_slice(0))
        object_denominator = torch.zeros_like(object_.get_slice(0), dtype=positions.dtype)
        for i in range(len(start_pts)):
            indices = torch.arange(start_pts[i], end_pts[i], device=object_numerator.device).long()
            p = self.forward_model.get_unique_probes(indices, always_return_probe_batch=True)
            p = self.forward_model.shift_unique_probes(indices, p, first_mode_only=True)

            object_numerator = ip.place_patches_integer(
                object_numerator,
                positions[start_pts[i] : end_pts[i]].round().int() + object_.pos_origin_coords,
                patches=(p.conj() * self.psi[start_pts[i] : end_pts[i]]).sum(1),
                op="add",
            )

            object_denominator = ip.place_patches_integer(
                object_denominator,
                positions[start_pts[i] : end_pts[i]].round().int() + object_.pos_origin_coords,
                patches=(p.abs() ** 2).sum(1),
                op="add",
            )

        # Calculate DM object update
        object_update = object_numerator / torch.sqrt(
            object_denominator**2 + (0.05 * object_denominator.max()) ** 2
        )
        # Apply inertia
        object_update = object_.get_slice(0) * object_.options.inertia + object_update * (
            1 - object_.options.inertia
        )
        # Clamp the object amplitude -- this is rarely needed
        idx = object_update.abs() > object_.options.amplitude_clamp_limit
        object_update[idx] = (
            object_update[idx] / (object_update[idx].abs())
        ) * object_.options.amplitude_clamp_limit

        return object_update

    @timer()
    def update_object(self, object_update: Tensor):
        # Apply object update
        self.parameter_group.object.set_data(object_update)

    @timer()
    def get_positions_update_chunk(
        self, indices: Tensor, obj_patches: Tensor, chi: Tensor
    ) -> Tensor:
        "Calculate the probe position update. Only gradient based updates are allowed for now."

        probe_positions = self.parameter_group.probe_positions
        probe = self.forward_model.get_unique_probes(indices, always_return_probe_batch=True)
        probe = self.forward_model.shift_unique_probes(indices, probe, first_mode_only=True)

        delta_pos = probe_positions.position_correction.get_update(
            chi,
            obj_patches,
            None,
            probe,
            None,
        )

        return delta_pos
