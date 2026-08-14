# Copyright © 2025 UChicago Argonne, LLC All right reserved
# Full license accessible at https://github.com//AdvancedPhotonSource/pty-chi/blob/main/LICENSE

import logging
from typing import Optional, TYPE_CHECKING, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

import ptychi.api as api
import ptychi.image_proc as ip
from ptychi.api.options.raar import RAARReconstructorOptions
from ptychi.reconstructors.base import (
    AnalyticalIterativePtychographyReconstructor,
    LossTracker,
)
from ptychi.timing.timer_utils import timer

if TYPE_CHECKING:
    import ptychi.data_structures.parameter_group as pg

logger = logging.getLogger(__name__)


class RAARReconstructor(AnalyticalIterativePtychographyReconstructor):
    """Relaxed averaged alternating reflections (RAAR) reconstructor.

    The exit-wave update follows equation (16) of Marchesini et al.,
    J. Appl. Cryst. 49, 1245–1252 (2016). Object and probe retrieval follow
    equations (14) and (15), respectively. OPR eigenmodes and weights are
    updated from the RAAR exit-wave increment using the shared OPR routine.

    The complete exit-wave stack is retained between epochs. ``batch_size`` is
    therefore unused; ``chunk_length`` controls the temporary working-set size.
    When an OPR update is enabled, its full exit-wave increment is also retained
    for the duration of that update.
    """

    parameter_group: "pg.PlanarPtychographyParameterGroup"

    def __init__(
        self,
        parameter_group: "pg.PlanarPtychographyParameterGroup",
        dataset: Dataset,
        options: Optional["api.options.raar.RAARReconstructorOptions"] = None,
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
        self.options: RAARReconstructorOptions = self.options

    def check_inputs(self, *args, **kwargs):
        if self.parameter_group.object.is_multislice:
            raise NotImplementedError("RAARReconstructor only supports 2D objects.")
        if (
            self.parameter_group.probe_positions.position_correction.options.correction_type
            is not api.enums.PositionCorrectionTypes.GRADIENT
        ):
            raise NotImplementedError(
                "RAARReconstructor only supports gradient position correction."
            )
        if self.options.batch_size != RAARReconstructorOptions.batch_size:
            logger.warning("RAAR reconstruction does not support batching!")

    def build_loss_tracker(self):
        if self.options.displayed_loss_function is not None:
            logger.warning(
                "The loss tracker is hard-coded to record the RAAR exit-wave update error. "
                "The specified metric function will not be used!"
            )
        self.loss_tracker = LossTracker(metric_function=None)

    def build_dataloader(self):
        data_loader_kwargs = {
            "dataset": self.dataset,
            "generator": torch.Generator(device=torch.get_default_device()),
            "batch_size": self.parameter_group.probe_positions.n_scan_points,
            "shuffle": False,
        }
        self.dataloader = DataLoader(**data_loader_kwargs)
        self.dataset.move_attributes_to_device(torch.get_default_device())

    def get_chunk_bounds(self) -> Tuple[list[int], list[int]]:
        n_scan_points = self.parameter_group.probe_positions.n_scan_points
        start_pts = list(range(0, n_scan_points, self.options.chunk_length))
        end_pts = [
            min(start_pt + self.options.chunk_length, n_scan_points) for start_pt in start_pts
        ]
        return start_pts, end_pts

    @timer()
    def run_minibatch(self, input_data, y_true, *args, **kwargs):
        raar_error_squared = self.compute_updates(y_true)
        self.loss_tracker.update_batch_loss(loss=raar_error_squared.sqrt())

    @timer()
    def compute_updates(self, y_true: Tensor) -> Tensor:
        object_ = self.parameter_group.object
        probe = self.parameter_group.probe
        probe_positions = self.parameter_group.probe_positions
        opr_mode_weights = self.parameter_group.opr_mode_weights
        start_pts, end_pts = self.get_chunk_bounds()
        object_update_enabled = object_.optimization_enabled(self.current_epoch)
        probe_update_enabled = probe.optimization_enabled(self.current_epoch)
        position_update_enabled = probe_positions.optimization_enabled(self.current_epoch)
        variable_probe_update_enabled = (
            probe.has_multiple_opr_modes
            and (
                probe_update_enabled
                or opr_mode_weights.eigenmode_weight_optimization_enabled(self.current_epoch)
            )
        ) or opr_mode_weights.intensity_variation_optimization_enabled(self.current_epoch)

        if self.current_epoch == 0:
            self.initialize_exit_wave(start_pts, end_pts)

        # P_Q z and P_Q P_a z use the same fixed probe within this RAAR step.
        if object_update_enabled:
            q_object = self.calculate_object_projection(self.psi, start_pts, end_pts)
            pa_object = self.calculate_data_projected_object_projection(y_true, start_pts, end_pts)
        else:
            q_object = object_.get_slice(0)
            pa_object = q_object

        raar_error_squared, exit_wave_update = self.apply_raar_exit_wave_update(
            y_true,
            q_object,
            pa_object,
            start_pts,
            end_pts,
            return_exit_wave_update=variable_probe_update_enabled,
        )

        object_update = None
        if object_update_enabled:
            object_update = self.calculate_object_projection(self.psi, start_pts, end_pts)

        if variable_probe_update_enabled:
            self.update_variable_probe(exit_wave_update)

        if object_update is not None:
            self.update_object(object_update)

        if probe_update_enabled:
            probe_update = self.calculate_probe_update(start_pts, end_pts)
            probe.set_data(probe_update, slicer=0)

        if position_update_enabled:
            self.update_probe_positions(start_pts, end_pts)

        return raar_error_squared

    @timer()
    def initialize_exit_wave(self, start_pts: list[int], end_pts: list[int]) -> None:
        object_ = self.parameter_group.object
        probe = self.parameter_group.probe
        probe_positions = self.parameter_group.probe_positions
        self.psi = torch.zeros(
            (
                probe_positions.n_scan_points,
                probe.n_modes,
                *probe.get_spatial_shape(),
            ),
            dtype=object_.data.dtype,
            device=object_.data.device,
        )
        for start_pt, end_pt in zip(start_pts, end_pts):
            self.psi[start_pt:end_pt] = self.calculate_exit_wave_chunk(start_pt, end_pt)

    @timer()
    def calculate_exit_wave_chunk(self, start_pt: int, end_pt: int) -> Tensor:
        object_ = self.parameter_group.object
        positions = self.parameter_group.probe_positions.tensor
        obj_patches = object_.extract_patches(
            positions[start_pt:end_pt].round().int(),
            self.parameter_group.probe.get_spatial_shape(),
            integer_mode=True,
        )
        indices = torch.arange(start_pt, end_pt, device=obj_patches.device).long()
        return self.forward_model.forward_real_space(indices=indices, obj_patches=obj_patches)

    @timer()
    def apply_data_projection(self, exit_wave: Tensor, y_true: Tensor) -> Tensor:
        propagated_exit_wave = self.forward_model.free_space_propagator.propagate_forward(exit_wave)
        propagated_exit_wave = self.replace_propagated_exit_wave_magnitude(
            propagated_exit_wave,
            y_true,
            constrained_pixel_mask=self.get_constrained_pixel_mask(y_true),
        )
        return self.forward_model.free_space_propagator.propagate_backward(propagated_exit_wave)

    def initialize_object_projection_terms(self) -> Tuple[Tensor, Tensor]:
        object_ = self.parameter_group.object
        positions = self.parameter_group.probe_positions.tensor
        numerator = torch.zeros_like(object_.get_slice(0))
        denominator = torch.zeros_like(object_.get_slice(0), dtype=positions.dtype)
        return numerator, denominator

    @timer()
    def add_to_object_projection_terms(
        self,
        numerator: Tensor,
        denominator: Tensor,
        exit_wave: Tensor,
        start_pt: int,
        end_pt: int,
    ) -> None:
        object_ = self.parameter_group.object
        positions = self.parameter_group.probe_positions.tensor
        indices = torch.arange(start_pt, end_pt, device=numerator.device).long()
        shifted_probe = self.forward_model.get_unique_probes(
            indices, always_return_probe_batch=True
        )
        shifted_probe = self.forward_model.shift_unique_probes(
            indices, shifted_probe, first_mode_only=True
        )
        placement_positions = positions[start_pt:end_pt].round().int() + object_.pos_origin_coords
        numerator.copy_(
            ip.place_patches_integer(
                numerator,
                placement_positions,
                patches=(shifted_probe.conj() * exit_wave).sum(1),
                op="add",
            )
        )
        denominator.copy_(
            ip.place_patches_integer(
                denominator,
                placement_positions,
                patches=(shifted_probe.abs() ** 2).sum(1),
                op="add",
            )
        )

    def finish_object_projection(self, numerator: Tensor, denominator: Tensor) -> Tensor:
        object_ = self.parameter_group.object
        tiny = torch.finfo(denominator.dtype).tiny
        projected_object = numerator / denominator.clamp_min(tiny)
        projected_object = torch.where(
            denominator > tiny,
            projected_object,
            object_.get_slice(0),
        )
        return projected_object

    @timer()
    def calculate_object_projection(
        self,
        exit_wave: Tensor,
        start_pts: list[int],
        end_pts: list[int],
    ) -> Tensor:
        numerator, denominator = self.initialize_object_projection_terms()
        for start_pt, end_pt in zip(start_pts, end_pts):
            self.add_to_object_projection_terms(
                numerator,
                denominator,
                exit_wave[start_pt:end_pt],
                start_pt,
                end_pt,
            )
        return self.finish_object_projection(numerator, denominator)

    @timer()
    def calculate_data_projected_object_projection(
        self,
        y_true: Tensor,
        start_pts: list[int],
        end_pts: list[int],
    ) -> Tensor:
        numerator, denominator = self.initialize_object_projection_terms()
        for start_pt, end_pt in zip(start_pts, end_pts):
            pa_exit_wave = self.apply_data_projection(
                self.psi[start_pt:end_pt], y_true[start_pt:end_pt]
            )
            self.add_to_object_projection_terms(
                numerator,
                denominator,
                pa_exit_wave,
                start_pt,
                end_pt,
            )
        return self.finish_object_projection(numerator, denominator)

    @timer()
    def synthesize_exit_wave_chunk(
        self, projected_object: Tensor, start_pt: int, end_pt: int
    ) -> Tensor:
        object_ = self.parameter_group.object
        positions = self.parameter_group.probe_positions.tensor
        indices = torch.arange(start_pt, end_pt, device=projected_object.device).long()
        obj_patches = ip.extract_patches_integer(
            projected_object,
            positions[start_pt:end_pt].round().int() + object_.pos_origin_coords,
            self.parameter_group.probe.get_spatial_shape(),
        )
        shifted_probe = self.forward_model.get_unique_probes(
            indices, always_return_probe_batch=True
        )
        shifted_probe = self.forward_model.shift_unique_probes(
            indices, shifted_probe, first_mode_only=True
        )
        return obj_patches[:, None] * shifted_probe

    @timer()
    def apply_raar_exit_wave_update(
        self,
        y_true: Tensor,
        q_object: Tensor,
        pa_object: Tensor,
        start_pts: list[int],
        end_pts: list[int],
        return_exit_wave_update: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Apply equation (16) of Marchesini et al. (2016)."""
        beta = self.options.beta
        error_squared = torch.zeros((), device=self.psi.device)
        exit_wave_update_all = torch.zeros_like(self.psi) if return_exit_wave_update else None
        for start_pt, end_pt in zip(start_pts, end_pts):
            previous_exit_wave = self.psi[start_pt:end_pt].clone()
            pa_exit_wave = self.apply_data_projection(previous_exit_wave, y_true[start_pt:end_pt])
            q_exit_wave = self.synthesize_exit_wave_chunk(q_object, start_pt, end_pt)
            q_pa_exit_wave = self.synthesize_exit_wave_chunk(pa_object, start_pt, end_pt)
            updated_exit_wave = (
                2 * beta * q_pa_exit_wave
                + (1 - 2 * beta) * pa_exit_wave
                + beta * (q_exit_wave - previous_exit_wave)
            )
            exit_wave_update = updated_exit_wave - previous_exit_wave
            self.psi[start_pt:end_pt] = updated_exit_wave
            error_squared += (exit_wave_update.abs() ** 2).sum()
            if exit_wave_update_all is not None:
                exit_wave_update_all[start_pt:end_pt] = exit_wave_update
        return error_squared, exit_wave_update_all

    @timer()
    def update_variable_probe(self, exit_wave_update: Tensor) -> None:
        """Update OPR eigenmodes and weights from the RAAR exit-wave update."""
        object_ = self.parameter_group.object
        probe = self.parameter_group.probe
        probe_positions = self.parameter_group.probe_positions
        opr_mode_weights = self.parameter_group.opr_mode_weights
        indices = torch.arange(
            probe_positions.n_scan_points, device=probe_positions.data.device
        ).long()
        obj_patches = object_.extract_patches(
            probe_positions.tensor.round().int(),
            probe.get_spatial_shape(),
            integer_mode=True,
        )

        delta_p_i = obj_patches.conj() * exit_wave_update
        delta_p_i = self.adjoint_shift_probe_update_direction(
            indices, delta_p_i, first_mode_only=True
        )
        delta_p_hat = delta_p_i.mean(0)
        opr_mode_weights.update_variable_probe(
            probe,
            indices,
            exit_wave_update,
            delta_p_i,
            delta_p_hat,
            obj_patches,
            self.current_epoch,
            probe_mode_index=0,
        )

    @timer()
    def update_object(self, projected_object: Tensor) -> None:
        object_ = self.parameter_group.object
        object_update = (
            object_.options.inertia * object_.get_slice(0)
            + (1 - object_.options.inertia) * projected_object
        )
        over_limit = object_update.abs() > object_.options.amplitude_clamp_limit
        object_update[over_limit] = (
            object_update[over_limit]
            / object_update[over_limit].abs()
            * object_.options.amplitude_clamp_limit
        )
        object_.set_data(object_update)

    @timer()
    def calculate_probe_update(self, start_pts: list[int], end_pts: list[int]) -> Tensor:
        object_ = self.parameter_group.object
        probe = self.parameter_group.probe
        positions = self.parameter_group.probe_positions.tensor
        numerator = torch.zeros_like(probe.get_opr_mode(0))
        denominator = torch.zeros_like(probe.get_opr_mode(0).abs())

        for start_pt, end_pt in zip(start_pts, end_pts):
            indices = torch.arange(start_pt, end_pt, device=numerator.device).long()
            obj_patches = object_.extract_patches(
                positions[start_pt:end_pt].round().int(),
                probe.get_spatial_shape(),
                integer_mode=True,
            )
            numerator_update = obj_patches.conj() * self.psi[start_pt:end_pt]
            denominator_update = obj_patches.abs() ** 2
            numerator_update = self.adjoint_shift_probe_update_direction(
                indices, numerator_update, first_mode_only=True
            )
            denominator_update = self.adjoint_shift_probe_update_direction(
                indices, denominator_update, first_mode_only=True
            )
            numerator += numerator_update.sum(0)
            denominator += denominator_update.sum(0)

        tiny = torch.finfo(denominator.dtype).tiny
        projected_probe = numerator / denominator.clamp_min(tiny)
        projected_probe = torch.where(
            denominator > tiny,
            projected_probe,
            probe.get_opr_mode(0),
        )
        return (
            probe.options.inertia * probe.get_opr_mode(0)
            + (1 - probe.options.inertia) * projected_probe
        )

    @timer()
    def update_probe_positions(self, start_pts: list[int], end_pts: list[int]) -> None:
        object_ = self.parameter_group.object
        probe = self.parameter_group.probe
        probe_positions = self.parameter_group.probe_positions
        positions = probe_positions.tensor
        delta_pos = torch.zeros_like(probe_positions.data)

        for start_pt, end_pt in zip(start_pts, end_pts):
            indices = torch.arange(start_pt, end_pt, device=positions.device).long()
            obj_patches = object_.extract_patches(
                positions[start_pt:end_pt].round().int(),
                probe.get_spatial_shape(),
                integer_mode=True,
            )
            model_exit_wave = self.forward_model.forward_real_space(
                indices=indices, obj_patches=obj_patches
            )
            delta_pos[start_pt:end_pt] = self.get_positions_update_chunk(
                indices=indices,
                obj_patches=obj_patches,
                chi=self.psi[start_pt:end_pt] - model_exit_wave,
            )

        probe_positions.set_grad(-delta_pos)
        probe_positions.step_optimizer()

    @timer()
    def get_positions_update_chunk(
        self, indices: Tensor, obj_patches: Tensor, chi: Tensor
    ) -> Tensor:
        probe_positions = self.parameter_group.probe_positions
        probe = self.forward_model.get_unique_probes(indices, always_return_probe_batch=True)
        probe = self.forward_model.shift_unique_probes(indices, probe, first_mode_only=True)
        return probe_positions.position_correction.get_update(
            chi,
            obj_patches,
            None,
            probe,
            None,
        )
