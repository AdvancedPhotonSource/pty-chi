# Copyright © 2025 UChicago Argonne, LLC All right reserved
# Full license accessible at https://github.com//AdvancedPhotonSource/pty-chi/blob/main/LICENSE

from abc import ABC, abstractmethod
import copy
import dataclasses
from dataclasses import dataclass
from typing import Optional
import warnings

import numpy as np
import torch
from numpy import ndarray
from torch import Tensor

from ptychi.api.options.task import PtychographyTaskOptions
from ptychi.api.options.workflow import WorkflowOptions
from ptychi.api.task import PtychographyTask, TaskArray


class _UnsetWorkflowData:
    pass


_UNSET = _UnsetWorkflowData()


@dataclass(frozen=True)
class _WorkflowData:
    diffraction_data: TaskArray
    object_data: TaskArray
    probe_data: TaskArray
    probe_position_x_px: TaskArray
    probe_position_y_px: TaskArray
    opr_mode_weights_data: Optional[TaskArray]
    valid_pixel_mask: Optional[TaskArray]


@dataclass(frozen=True)
class _WorkflowTensorData:
    diffraction_data: Tensor
    object_data: Tensor
    probe_data: Tensor
    probe_position_x_px: Tensor
    probe_position_y_px: Tensor
    opr_mode_weights_data: Optional[Tensor]
    valid_pixel_mask: Optional[Tensor]


class BaseWorkflow(ABC):
    """Base class for workflows composed of one or more ptychography tasks."""

    def __init__(
        self,
        task_options: PtychographyTaskOptions,
        *args,
        diffraction_data: Optional[TaskArray] | _UnsetWorkflowData = _UNSET,
        object_data: Optional[TaskArray] | _UnsetWorkflowData = _UNSET,
        probe_data: Optional[TaskArray] | _UnsetWorkflowData = _UNSET,
        probe_position_x_px: Optional[TaskArray] | _UnsetWorkflowData = _UNSET,
        probe_position_y_px: Optional[TaskArray] | _UnsetWorkflowData = _UNSET,
        opr_mode_weights_data: Optional[TaskArray] | _UnsetWorkflowData = _UNSET,
        valid_pixel_mask: Optional[TaskArray] | _UnsetWorkflowData = _UNSET,
        workflow_options: WorkflowOptions,
        **kwargs,
    ) -> None:
        if not isinstance(task_options, PtychographyTaskOptions):
            raise TypeError("`task_options` must be a PtychographyTaskOptions instance.")
        if not isinstance(workflow_options, WorkflowOptions):
            raise TypeError("`workflow_options` must be a WorkflowOptions instance.")

        self.task_options = task_options
        self.workflow_options = workflow_options
        self.tasks: list[PtychographyTask] = []
        self._task_args = args
        self._task_kwargs = kwargs

        data = self._resolve_workflow_data(
            diffraction_data=diffraction_data,
            object_data=object_data,
            probe_data=probe_data,
            probe_position_x_px=probe_position_x_px,
            probe_position_y_px=probe_position_y_px,
            opr_mode_weights_data=opr_mode_weights_data,
            valid_pixel_mask=valid_pixel_mask,
        )
        self._warn_for_gpu_data(data)
        cpu_data = self._copy_workflow_data_to_cpu(data)
        self.diffraction_data = cpu_data.diffraction_data
        self.object_data = cpu_data.object_data
        self.probe_data = cpu_data.probe_data
        self.probe_position_x_px = cpu_data.probe_position_x_px
        self.probe_position_y_px = cpu_data.probe_position_y_px
        self.opr_mode_weights_data = cpu_data.opr_mode_weights_data
        self.valid_pixel_mask = cpu_data.valid_pixel_mask
        self.task_options = self._copy_task_options()

    def _resolve_workflow_data(
        self,
        *,
        task_options: Optional[PtychographyTaskOptions] = None,
        diffraction_data: Optional[TaskArray] | _UnsetWorkflowData,
        object_data: Optional[TaskArray] | _UnsetWorkflowData,
        probe_data: Optional[TaskArray] | _UnsetWorkflowData,
        probe_position_x_px: Optional[TaskArray] | _UnsetWorkflowData,
        probe_position_y_px: Optional[TaskArray] | _UnsetWorkflowData,
        opr_mode_weights_data: Optional[TaskArray] | _UnsetWorkflowData,
        valid_pixel_mask: Optional[TaskArray] | _UnsetWorkflowData,
    ) -> _WorkflowData:
        task_options = self.task_options if task_options is None else task_options
        return _WorkflowData(
            diffraction_data=self._resolve_data_field(
                value=diffraction_data,
                option_owner=task_options.data_options,
                option_field_name="data",
                option_path="task_options.data_options.data",
                kwarg_name="diffraction_data",
                required=True,
            ),
            object_data=self._resolve_data_field(
                value=object_data,
                option_owner=task_options.object_options,
                option_field_name="initial_guess",
                option_path="task_options.object_options.initial_guess",
                kwarg_name="object_data",
                required=True,
            ),
            probe_data=self._resolve_data_field(
                value=probe_data,
                option_owner=task_options.probe_options,
                option_field_name="initial_guess",
                option_path="task_options.probe_options.initial_guess",
                kwarg_name="probe_data",
                required=True,
            ),
            probe_position_x_px=self._resolve_data_field(
                value=probe_position_x_px,
                option_owner=task_options.probe_position_options,
                option_field_name="position_x_px",
                option_path="task_options.probe_position_options.position_x_px",
                kwarg_name="probe_position_x_px",
                required=True,
            ),
            probe_position_y_px=self._resolve_data_field(
                value=probe_position_y_px,
                option_owner=task_options.probe_position_options,
                option_field_name="position_y_px",
                option_path="task_options.probe_position_options.position_y_px",
                kwarg_name="probe_position_y_px",
                required=True,
            ),
            opr_mode_weights_data=self._resolve_data_field(
                value=opr_mode_weights_data,
                option_owner=task_options.opr_mode_weight_options,
                option_field_name="initial_weights",
                option_path="task_options.opr_mode_weight_options.initial_weights",
                kwarg_name="opr_mode_weights_data",
                required=False,
            ),
            valid_pixel_mask=self._resolve_data_field(
                value=valid_pixel_mask,
                option_owner=task_options.data_options,
                option_field_name="valid_pixel_mask",
                option_path="task_options.data_options.valid_pixel_mask",
                kwarg_name="valid_pixel_mask",
                required=False,
            ),
        )

    @staticmethod
    def _resolve_data_field(
        *,
        value,
        option_owner,
        option_field_name: str,
        option_path: str,
        kwarg_name: str,
        required: bool,
    ):
        option_value = getattr(option_owner, option_field_name)
        if value is not _UNSET:
            if option_value is not None:
                warnings.warn(
                    f"`{option_path}` is deprecated and was ignored because "
                    f"`{kwarg_name}` was supplied to the workflow.",
                    DeprecationWarning,
                    stacklevel=4,
                )
            resolved_value = value
        elif option_value is not None:
            warnings.warn(
                f"Passing workflow data via `{option_path}` is deprecated; pass "
                f"`{kwarg_name}` to the workflow instead.",
                DeprecationWarning,
                stacklevel=4,
            )
            resolved_value = option_value
        else:
            resolved_value = None

        if required and resolved_value is None:
            raise ValueError(
                f"`{kwarg_name}` is required. Passing it through `{option_path}` is "
                "temporarily supported but deprecated."
            )
        return resolved_value

    @staticmethod
    def _warn_for_gpu_data(data: _WorkflowData) -> None:
        gpu_fields = [
            field.name
            for field in dataclasses.fields(data)
            if isinstance(getattr(data, field.name), Tensor)
            and getattr(data, field.name).device.type != "cpu"
        ]
        if gpu_fields:
            warnings.warn(
                "Workflow inputs were copied to CPU, but the original tensors remain on "
                "their accelerator devices and continue to occupy accelerator memory: "
                f"{', '.join(gpu_fields)}.",
                UserWarning,
                stacklevel=3,
            )

    @staticmethod
    def _copy_tensor_to_cpu(data: TaskArray) -> Tensor:
        if isinstance(data, Tensor):
            return data.detach().to(device="cpu").clone()
        if isinstance(data, ndarray):
            return torch.from_numpy(np.array(data, copy=True))
        return torch.tensor(data, device="cpu")

    @classmethod
    def _copy_optional_tensor_to_cpu(cls, data: Optional[TaskArray]) -> Optional[Tensor]:
        if data is None:
            return None
        return cls._copy_tensor_to_cpu(data)

    @classmethod
    def _copy_workflow_data_to_cpu(cls, data: _WorkflowData) -> _WorkflowTensorData:
        return _WorkflowTensorData(
            diffraction_data=cls._copy_tensor_to_cpu(data.diffraction_data),
            object_data=cls._copy_tensor_to_cpu(data.object_data),
            probe_data=cls._copy_tensor_to_cpu(data.probe_data),
            probe_position_x_px=cls._copy_tensor_to_cpu(data.probe_position_x_px),
            probe_position_y_px=cls._copy_tensor_to_cpu(data.probe_position_y_px),
            opr_mode_weights_data=cls._copy_optional_tensor_to_cpu(
                data.opr_mode_weights_data
            ),
            valid_pixel_mask=cls._copy_optional_tensor_to_cpu(data.valid_pixel_mask),
        )

    def _copy_task_options(
        self, task_options: Optional[PtychographyTaskOptions] = None
    ) -> PtychographyTaskOptions:
        task_options = self.task_options if task_options is None else task_options
        data_fields = (
            task_options.data_options.data,
            task_options.data_options.valid_pixel_mask,
            task_options.object_options.initial_guess,
            task_options.probe_options.initial_guess,
            task_options.probe_position_options.position_x_px,
            task_options.probe_position_options.position_y_px,
            task_options.opr_mode_weight_options.initial_weights,
        )
        memo = {id(value): None for value in data_fields if value is not None}
        options = copy.deepcopy(task_options, memo)
        options.data_options.data = None
        options.data_options.valid_pixel_mask = None
        options.object_options.initial_guess = None
        options.probe_options.initial_guess = None
        options.probe_position_options.position_x_px = None
        options.probe_position_options.position_y_px = None
        options.opr_mode_weight_options.initial_weights = None
        return options

    @abstractmethod
    def run(self) -> None:
        """Run the workflow."""
