# Copyright © 2025 UChicago Argonne, LLC All right reserved
# Full license accessible at https://github.com//AdvancedPhotonSource/pty-chi/blob/main/LICENSE

from typing import Optional

import ptychi.api as api
from ptychi.api.options.task import PtychographyTaskOptions
from ptychi.api.options.workflow import MultiscanSharedObjectWorkflowOptions
from ptychi.api.task import PtychographyTask, TaskArray
from ptychi.workflows.base import BaseWorkflow, _UNSET, _UnsetWorkflowData


class MultiscanSharedObjectWorkflow(BaseWorkflow):
    """Reconstruct multiple scans by passing one object between scan-specific tasks."""

    task_options: list[PtychographyTaskOptions]  # type: ignore[assignment]
    workflow_options: MultiscanSharedObjectWorkflowOptions

    def __init__(
        self,
        task_options: list[PtychographyTaskOptions],
        *args,
        diffraction_data: list[TaskArray] | _UnsetWorkflowData = _UNSET,
        object_data: list[TaskArray] | _UnsetWorkflowData = _UNSET,
        probe_data: list[TaskArray] | _UnsetWorkflowData = _UNSET,
        probe_position_x_px: list[TaskArray] | _UnsetWorkflowData = _UNSET,
        probe_position_y_px: list[TaskArray] | _UnsetWorkflowData = _UNSET,
        opr_mode_weights_data: (
            list[Optional[TaskArray]] | None | _UnsetWorkflowData
        ) = _UNSET,
        valid_pixel_mask: list[Optional[TaskArray]] | None | _UnsetWorkflowData = _UNSET,
        workflow_options: MultiscanSharedObjectWorkflowOptions,
        **kwargs,
    ) -> None:
        if not isinstance(task_options, list):
            raise TypeError("`task_options` must be a list.")
        if not task_options:
            raise ValueError("`task_options` must contain at least one options object.")
        if not all(isinstance(options, PtychographyTaskOptions) for options in task_options):
            raise TypeError(
                "Every member of `task_options` must be a PtychographyTaskOptions instance."
            )
        if not isinstance(workflow_options, MultiscanSharedObjectWorkflowOptions):
            raise TypeError(
                "`workflow_options` must be a "
                "MultiscanSharedObjectWorkflowOptions instance."
            )

        n_tasks = len(task_options)
        values_by_name = {
            "diffraction_data": self._expand_task_data(
                "diffraction_data", diffraction_data, n_tasks
            ),
            "object_data": self._expand_task_data("object_data", object_data, n_tasks),
            "probe_data": self._expand_task_data("probe_data", probe_data, n_tasks),
            "probe_position_x_px": self._expand_task_data(
                "probe_position_x_px", probe_position_x_px, n_tasks
            ),
            "probe_position_y_px": self._expand_task_data(
                "probe_position_y_px", probe_position_y_px, n_tasks
            ),
            "opr_mode_weights_data": self._expand_task_data(
                "opr_mode_weights_data", opr_mode_weights_data, n_tasks, optional=True
            ),
            "valid_pixel_mask": self._expand_task_data(
                "valid_pixel_mask", valid_pixel_mask, n_tasks, optional=True
            ),
        }

        copied_options = []
        copied_data = []
        for i_task, options in enumerate(task_options):
            data = self._resolve_workflow_data(
                task_options=options,
                **{name: values[i_task] for name, values in values_by_name.items()},
            )
            self._warn_for_gpu_data(data)
            copied_data.append(self._copy_workflow_data_to_cpu(data))
            copied_options.append(self._copy_task_options(options))

        self.workflow_options = workflow_options
        self.task_options = copied_options
        self.tasks: list[PtychographyTask] = []
        self._task_args = args
        self._task_kwargs = kwargs
        self._workflow_task_data = copied_data
        self._completed = False

        for field_name in values_by_name:
            setattr(self, field_name, [getattr(data, field_name) for data in copied_data])

        self._validate_shared_object_geometry()
        total_epochs = workflow_options.num_outer_epochs * workflow_options.num_inner_epochs
        for options in self.task_options:
            options.reconstructor_options.num_epochs = total_epochs

    @staticmethod
    def _expand_task_data(
        name: str,
        value,
        n_tasks: int,
        *,
        optional: bool = False,
    ) -> list:
        if value is _UNSET:
            return [_UNSET] * n_tasks
        if value is None:
            if optional:
                return [None] * n_tasks
            raise ValueError(f"`{name}` is required.")
        if not isinstance(value, list):
            raise TypeError(f"`{name}` must be a list.")
        if len(value) != n_tasks:
            raise ValueError(
                f"`{name}` must contain one member for each task "
                f"({len(value)} != {n_tasks})."
            )
        return value

    def _validate_shared_object_geometry(self) -> None:
        if any(
            options.object_options.determine_position_origin_coords_by
            != api.ObjectPosOriginCoordsMethods.SUPPORT
            for options in self.task_options
        ):
            raise ValueError(
                "All task options must set "
                "`object_options.determine_position_origin_coords_by` to `SUPPORT`."
            )

        reference_shape = tuple(self._workflow_task_data[0].object_data.shape)
        reference_options = self.task_options[0].object_options
        for i_task, (data, options) in enumerate(
            zip(self._workflow_task_data[1:], self.task_options[1:]), start=1
        ):
            object_data = data.object_data
            if tuple(object_data.shape) != reference_shape:
                raise ValueError(
                    "All members of `object_data` must have the same shape; "
                    f"task 0 has {reference_shape} and task {i_task} has "
                    f"{tuple(object_data.shape)}."
                )
            object_options = options.object_options
            geometry = (
                object_options.pixel_size_m,
                object_options.pixel_size_aspect_ratio,
                object_options.slice_spacings_m,
            )
            reference_geometry = (
                reference_options.pixel_size_m,
                reference_options.pixel_size_aspect_ratio,
                reference_options.slice_spacings_m,
            )
            if geometry != reference_geometry:
                raise ValueError(
                    "All task options must use matching object pixel and slice geometry."
                )

    def run(self) -> None:
        if self.tasks:
            raise RuntimeError("This multiscan shared-object workflow has already been run.")

        self._completed = False
        self._build_tasks()
        for _ in range(self.workflow_options.num_outer_epochs):
            for i_task, task in enumerate(self.tasks):
                task.build_default_device()
                task.build_default_dtype()
                task.set_large_tensor_device()
                try:
                    task.run(self.workflow_options.num_inner_epochs)
                    if len(self.tasks) > 1:
                        next_task = self.tasks[(i_task + 1) % len(self.tasks)]
                        next_task.copy_data_from_task(task, params_to_copy=("object",))
                finally:
                    task.set_large_tensor_device("cpu")

        final_task = self.tasks[-1]
        for task in self.tasks[:-1]:
            task.copy_data_from_task(final_task, params_to_copy=("object",))
        self._completed = True

    def _build_tasks(self) -> None:
        for i_task, options in enumerate(self.task_options):
            data = self._workflow_task_data[i_task]
            task = PtychographyTask(
                options,
                *self._task_args,
                diffraction_data=data.diffraction_data,
                object_data=data.object_data,
                probe_data=data.probe_data,
                probe_position_x_px=data.probe_position_x_px,
                probe_position_y_px=data.probe_position_y_px,
                opr_mode_weights_data=data.opr_mode_weights_data,
                valid_pixel_mask=data.valid_pixel_mask,
                **self._task_kwargs,
            )
            self.tasks.append(task)
            if i_task > 0 and task.reconstructor is not None:
                pbar = getattr(task.reconstructor, "pbar", None)
                if pbar is not None:
                    pbar.disable = True
            task.set_large_tensor_device("cpu")
