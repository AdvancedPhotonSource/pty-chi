# Copyright © 2025 UChicago Argonne, LLC All right reserved
# Full license accessible at https://github.com//AdvancedPhotonSource/pty-chi/blob/main/LICENSE

import math
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor

import ptychi.image_proc as image_proc
from ptychi.api.options.workflow import ProgressiveResolutionWorkflowOptions
from ptychi.api.task import PtychographyTask
from ptychi.workflows.base import BaseWorkflow


class ProgressiveResolutionWorkflow(BaseWorkflow):
    """Reconstruct diffraction data from coarse to full spatial resolution."""

    workflow_options: ProgressiveResolutionWorkflowOptions

    def run(self) -> None:
        if self.tasks:
            raise RuntimeError("This progressive-resolution workflow has already been run.")
        if not isinstance(self.workflow_options, ProgressiveResolutionWorkflowOptions):
            raise TypeError(
                "`workflow_options` must be a ProgressiveResolutionWorkflowOptions instance."
            )

        self._validate_spatial_shapes()
        self._validate_simulated_intensity_binning()
        self._completed = False
        for i_level in range(self.workflow_options.num_resolution_levels):
            factor = self._get_downsampling_factor(i_level)
            level_data = self._build_level_data(i_level=i_level, factor=factor)
            level_options = self._copy_task_options()
            level_options.reconstructor_options.num_epochs = (
                self.workflow_options.num_epochs_all_levels[i_level]
            )
            level_options.object_options.pixel_size_m = (
                self.task_options.object_options.pixel_size_m * factor
            )
            if self.task_options.probe_options.pixel_size_m is not None:
                level_options.probe_options.pixel_size_m = (
                    self.task_options.probe_options.pixel_size_m * factor
                )

            task = PtychographyTask(
                level_options,
                *self._task_args,
                diffraction_data=self._build_level_diffraction_data(factor),
                object_data=level_data["object_data"],
                probe_data=level_data["probe_data"],
                probe_position_x_px=level_data["probe_position_x_px"],
                probe_position_y_px=level_data["probe_position_y_px"],
                opr_mode_weights_data=level_data["opr_mode_weights_data"],
                valid_pixel_mask=self._build_level_valid_pixel_mask(factor),
                **self._task_kwargs,
            )
            self.tasks.append(task)
            try:
                task.run()
            finally:
                task.set_large_tensor_device("cpu")

        self._completed = True

    def get_full_resolution_task(self) -> PtychographyTask:
        """Return the completed task for the full-resolution level."""
        if not getattr(self, "_completed", False):
            raise RuntimeError(
                "The full-resolution task is not available until the workflow completes."
            )
        return self.tasks[-1]

    def _validate_spatial_shapes(self) -> None:
        expected_ndim = {
            "diffraction_data": 3,
            "object_data": 3,
            "probe_data": 4,
        }
        for name, ndim in expected_ndim.items():
            if getattr(self, name).ndim != ndim:
                raise ValueError(f"`{name}` must be {ndim}D.")

        if self.valid_pixel_mask is not None:
            if self.valid_pixel_mask.ndim != 2:
                raise ValueError("`valid_pixel_mask` must be a 2D boolean mask.")
            if tuple(self.valid_pixel_mask.shape) != tuple(self.diffraction_data.shape[-2:]):
                raise ValueError(
                    "`valid_pixel_mask.shape` must match the diffraction pattern shape."
                )

    def _validate_simulated_intensity_binning(self) -> None:
        """Check that the probe bins onto the diffraction patterns at every level.

        The probe and the diffraction patterns are rounded independently at each level,
        so a configuration that is valid at full resolution can be invalid at a coarser
        level. Checking all levels up front fails before any reconstruction runs.
        """
        binning = (
            self.task_options.reconstructor_options.forward_model_options
            .simulated_intensity_binning_factor
        )
        if binning == 1:
            return
        num_levels = self.workflow_options.num_resolution_levels
        for i_level in range(num_levels):
            factor = self._get_downsampling_factor(i_level)
            probe_shape = self._target_spatial_shape(self.probe_data, factor)
            pattern_shape = self._target_spatial_shape(self.diffraction_data, factor)
            level = (
                f"At progressive-resolution level {i_level + 1} of {num_levels} (downsampling "
                f"factor {factor}), the probe shape is {probe_shape} and the diffraction "
                f"pattern shape is {pattern_shape}"
            )
            if probe_shape[0] % binning or probe_shape[1] % binning:
                raise ValueError(
                    f"{level}; the probe shape is not divisible by "
                    f"`simulated_intensity_binning_factor` ({binning})."
                )
            binned_shape = (probe_shape[0] // binning, probe_shape[1] // binning)
            if binned_shape[0] < pattern_shape[0] or binned_shape[1] < pattern_shape[1]:
                raise ValueError(
                    f"{level}; with `simulated_intensity_binning_factor` ({binning}) the "
                    f"probe bins to {binned_shape}, smaller than the diffraction patterns."
                )

    def _get_downsampling_factor(self, i_level: int) -> int:
        return 2 ** (self.workflow_options.num_resolution_levels - 1 - i_level)

    def _build_level_data(self, i_level: int, factor: int) -> dict[str, Tensor | None]:
        object_target = self._target_spatial_shape(self.object_data, factor)
        probe_target = self._target_spatial_shape(self.probe_data, factor)
        if i_level == 0:
            return {
                "object_data": self._resize_spatial_data(self.object_data, object_target),
                "probe_data": self._resize_spatial_data(self.probe_data, probe_target),
                "probe_position_x_px": self.probe_position_x_px / factor,
                "probe_position_y_px": self.probe_position_y_px / factor,
                "opr_mode_weights_data": (
                    None
                    if self.opr_mode_weights_data is None
                    else self.opr_mode_weights_data.clone()
                ),
            }

        previous_task = self.tasks[-1]
        previous_positions = self._get_task_tensor(previous_task, "probe_positions")
        previous_factor = factor * 2
        position_scale = previous_factor / factor
        return {
            "object_data": self._resize_spatial_data(
                self._get_task_tensor(previous_task, "object"), object_target
            ),
            "probe_data": self._resize_spatial_data(
                self._get_task_tensor(previous_task, "probe"), probe_target
            ),
            "probe_position_x_px": previous_positions[:, 1] * position_scale,
            "probe_position_y_px": previous_positions[:, 0] * position_scale,
            "opr_mode_weights_data": self._get_task_tensor(
                previous_task, "opr_mode_weights"
            ),
        }

    @staticmethod
    def _get_task_tensor(
        task: PtychographyTask,
        name: Literal["object", "probe", "probe_positions", "opr_mode_weights"],
    ) -> Tensor:
        data = task.get_data_to_cpu(name)
        if not isinstance(data, Tensor):
            raise TypeError(f"Expected tensor data for `{name}`.")
        return data.detach().cpu().clone()

    @staticmethod
    def _target_spatial_shape(data: Tensor, factor: int) -> tuple[int, int]:
        height, width = data.shape[-2:]
        return (
            max(1, math.floor(height / factor + 0.5)),
            max(1, math.floor(width / factor + 0.5)),
        )

    @staticmethod
    def _resize_spatial_data(data: Tensor, size: tuple[int, int]) -> Tensor:
        if tuple(data.shape[-2:]) == size:
            return data.detach().cpu().clone()

        leading_shape = data.shape[:-2]
        flattened = data.detach().cpu().reshape(-1, 1, *data.shape[-2:])
        if flattened.is_complex():
            resized = F.interpolate(
                flattened.real, size=size, mode="bilinear", align_corners=False
            ) + 1j * F.interpolate(
                flattened.imag, size=size, mode="bilinear", align_corners=False
            )
        else:
            if not flattened.is_floating_point():
                flattened = flattened.to(torch.get_default_dtype())
            resized = F.interpolate(
                flattened, size=size, mode="bilinear", align_corners=False
            )
        return resized.reshape(*leading_shape, *size)

    def _crop_reciprocal_data(self, data: Tensor, factor: int) -> Tensor:
        target_size = self._target_spatial_shape(data, factor)
        if not self.task_options.data_options.fft_shift:
            data = torch.fft.ifftshift(data, dim=(-2, -1))
            data = image_proc.central_crop(data, target_size)
            data = torch.fft.fftshift(data, dim=(-2, -1))
        else:
            data = image_proc.central_crop(data, target_size)
        return data.clone()

    def _build_level_diffraction_data(self, factor: int) -> Tensor:
        if math.isfinite(
            self.task_options.data_options.free_space_propagation_distance_m
        ):
            target_size = self._target_spatial_shape(self.diffraction_data, factor)
            return self._resize_spatial_data(self.diffraction_data, target_size)
        return self._crop_reciprocal_data(self.diffraction_data, factor)

    def _build_level_valid_pixel_mask(self, factor: int) -> Tensor | None:
        if self.valid_pixel_mask is None:
            return None
        if not math.isfinite(
            self.task_options.data_options.free_space_propagation_distance_m
        ):
            return self._crop_reciprocal_data(self.valid_pixel_mask, factor)

        target_size = self._target_spatial_shape(self.valid_pixel_mask, factor)
        if tuple(self.valid_pixel_mask.shape[-2:]) == target_size:
            return self.valid_pixel_mask.clone()
        resized = F.interpolate(
            self.valid_pixel_mask[None, None].to(torch.float32),
            size=target_size,
            mode="nearest",
        )
        return resized[0, 0].to(torch.bool)
