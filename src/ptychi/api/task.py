# Copyright © 2025 UChicago Argonne, LLC All right reserved
# Full license accessible at https://github.com//AdvancedPhotonSource/pty-chi/blob/main/LICENSE

from typing import Literal, Optional, Union, overload
import dataclasses
from dataclasses import dataclass
from types import TracebackType
import random
import logging
import os
import warnings

import torch
import numpy as np
from torch import Tensor
from numpy import ndarray

import ptychi.api as api
import ptychi.data_structures.object as object
import ptychi.data_structures.opr_mode_weights as oprweights
import ptychi.data_structures.probe as probe
import ptychi.data_structures.probe_positions as probepos
import ptychi.data_structures.parameter_group as paramgrp
import ptychi.maps as maps
from ptychi.io_handles import PtychographyDataset
from ptychi.reconstructors.base import Reconstructor
from ptychi.utils import to_tensor
import ptychi.utils as utils
import ptychi.maths as pmath
from ptychi.timing import timer_utils
import ptychi.movies as movies
from ptychi.device import AcceleratorModuleWrapper
from ptychi.parallel import MultiprocessMixin

logger = logging.getLogger(__name__)


class _UnsetTaskData:
    pass


_UNSET = _UnsetTaskData()


TaskArray = Union[Tensor, ndarray, list, tuple]


@dataclass(frozen=True)
class _PtychographyTaskData:
    diffraction_data: TaskArray
    object_data: TaskArray
    probe_data: TaskArray
    probe_position_x_px: TaskArray
    probe_position_y_px: TaskArray
    opr_mode_weights_data: Optional[TaskArray]
    valid_pixel_mask: Optional[TaskArray]


class Task(MultiprocessMixin):
    def __init__(self, options: api.options.base.TaskOptions, *args, **kwargs) -> None:
        pass

    def __enter__(self) -> "Task":
        return self

    @overload
    def __exit__(self, exception_type: None, exception_value: None, traceback: None) -> None: ...

    @overload
    def __exit__(
        self,
        exception_type: type[BaseException],
        exception_value: BaseException,
        traceback: TracebackType,
    ) -> None: ...

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        AcceleratorModuleWrapper.get_module().empty_cache()


class PtychographyTask(Task):
    def __init__(
        self,
        options: api.options.task.PtychographyTaskOptions,
        *args,
        diffraction_data: Optional[TaskArray] | _UnsetTaskData = _UNSET,
        object_data: Optional[TaskArray] | _UnsetTaskData = _UNSET,
        probe_data: Optional[TaskArray] | _UnsetTaskData = _UNSET,
        probe_position_x_px: Optional[TaskArray] | _UnsetTaskData = _UNSET,
        probe_position_y_px: Optional[TaskArray] | _UnsetTaskData = _UNSET,
        opr_mode_weights_data: Optional[TaskArray] | _UnsetTaskData = _UNSET,
        valid_pixel_mask: Optional[TaskArray] | _UnsetTaskData = _UNSET,
        **kwargs,
    ) -> None:
        super().__init__(options, *args, **kwargs)
        self.options = options
        self.data_options = options.data_options
        self.object_options = options.object_options
        self.probe_options = options.probe_options
        self.position_options = options.probe_position_options
        self.opr_mode_weight_options = options.opr_mode_weight_options
        self.reconstructor_options = options.reconstructor_options

        self.dataset = None
        self.object = None
        self.probe = None
        self.probe_positions = None
        self.opr_mode_weights = None
        self.reconstructor: Reconstructor | None = None

        self._task_data = self._resolve_task_data(
            diffraction_data=diffraction_data,
            object_data=object_data,
            probe_data=probe_data,
            probe_position_x_px=probe_position_x_px,
            probe_position_y_px=probe_position_y_px,
            opr_mode_weights_data=opr_mode_weights_data,
            valid_pixel_mask=valid_pixel_mask,
        )

        self.check_options()
        self.check_task_data()
        self.build()
        
    def check_options(self):
        self.options.check()

    def _resolve_task_data(
        self,
        *,
        diffraction_data: Optional[TaskArray] | _UnsetTaskData,
        object_data: Optional[TaskArray] | _UnsetTaskData,
        probe_data: Optional[TaskArray] | _UnsetTaskData,
        probe_position_x_px: Optional[TaskArray] | _UnsetTaskData,
        probe_position_y_px: Optional[TaskArray] | _UnsetTaskData,
        opr_mode_weights_data: Optional[TaskArray] | _UnsetTaskData,
        valid_pixel_mask: Optional[TaskArray] | _UnsetTaskData,
    ) -> _PtychographyTaskData:
        return _PtychographyTaskData(
            diffraction_data=self._resolve_data_field(
                value=diffraction_data,
                option_owner=self.data_options,
                option_field_name="data",
                option_path="options.data_options.data",
                kwarg_name="diffraction_data",
                required=True,
            ),
            object_data=self._resolve_data_field(
                value=object_data,
                option_owner=self.object_options,
                option_field_name="initial_guess",
                option_path="options.object_options.initial_guess",
                kwarg_name="object_data",
                required=True,
            ),
            probe_data=self._resolve_data_field(
                value=probe_data,
                option_owner=self.probe_options,
                option_field_name="initial_guess",
                option_path="options.probe_options.initial_guess",
                kwarg_name="probe_data",
                required=True,
            ),
            probe_position_x_px=self._resolve_data_field(
                value=probe_position_x_px,
                option_owner=self.position_options,
                option_field_name="position_x_px",
                option_path="options.probe_position_options.position_x_px",
                kwarg_name="probe_position_x_px",
                required=True,
            ),
            probe_position_y_px=self._resolve_data_field(
                value=probe_position_y_px,
                option_owner=self.position_options,
                option_field_name="position_y_px",
                option_path="options.probe_position_options.position_y_px",
                kwarg_name="probe_position_y_px",
                required=True,
            ),
            opr_mode_weights_data=self._resolve_data_field(
                value=opr_mode_weights_data,
                option_owner=self.opr_mode_weight_options,
                option_field_name="initial_weights",
                option_path="options.opr_mode_weight_options.initial_weights",
                kwarg_name="opr_mode_weights_data",
                required=False,
            ),
            valid_pixel_mask=self._resolve_data_field(
                value=valid_pixel_mask,
                option_owner=self.data_options,
                option_field_name="valid_pixel_mask",
                option_path="options.data_options.valid_pixel_mask",
                kwarg_name="valid_pixel_mask",
                required=False,
            ),
        )

    def _resolve_data_field(
        self,
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
                    f"`{kwarg_name}` was supplied to `PtychographyTask`.",
                    DeprecationWarning,
                    stacklevel=4,
                )
            resolved_value = value
        elif option_value is not None:
            warnings.warn(
                f"Passing task data via `{option_path}` is deprecated; pass "
                f"`{kwarg_name}` to `PtychographyTask` instead.",
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

    def check_task_data(self):
        data = self._task_data

        diffraction_shape = self._shape(data.diffraction_data)
        object_shape = self._shape(data.object_data)
        probe_shape = self._shape(data.probe_data)
        position_x_shape = self._shape(data.probe_position_x_px)
        position_y_shape = self._shape(data.probe_position_y_px)

        if len(diffraction_shape) != 3:
            raise ValueError("`diffraction_data` must have shape (n_positions, height, width).")
        if len(object_shape) != 3:
            raise ValueError("`object_data` must have shape (n_slices, height, width).")
        if len(probe_shape) != 4:
            raise ValueError(
                "`probe_data` must have shape "
                "(n_opr_modes, n_incoherent_modes, height, width)."
            )
        if len(position_x_shape) != 1 or len(position_y_shape) != 1:
            raise ValueError("`probe_position_x_px` and `probe_position_y_px` must be 1D arrays.")
        if position_x_shape != position_y_shape:
            raise ValueError(
                "`probe_position_x_px` and `probe_position_y_px` must have matching shapes."
            )

        n_positions = position_x_shape[0]
        if n_positions < 1:
            raise ValueError("At least one probe position is required.")
        if diffraction_shape[0] != n_positions:
            raise ValueError(
                "`diffraction_data.shape[0]` must match the number of probe positions "
                f"({diffraction_shape[0]} != {n_positions})."
            )

        if data.valid_pixel_mask is not None:
            valid_pixel_mask_shape = self._shape(data.valid_pixel_mask)
            if len(valid_pixel_mask_shape) != 2:
                raise ValueError("`valid_pixel_mask` must be a 2D boolean mask.")
            if valid_pixel_mask_shape != diffraction_shape[-2:]:
                raise ValueError(
                    "`valid_pixel_mask.shape` must match the diffraction pattern shape "
                    f"({valid_pixel_mask_shape} != {diffraction_shape[-2:]})."
                )

        self._check_opr_mode_weights_shape(
            weights=data.opr_mode_weights_data,
            n_positions=n_positions,
            n_opr_modes=probe_shape[0],
        )
        self._check_object_position_coverage(
            object_shape=object_shape,
            probe_shape=probe_shape,
            position_x_px=data.probe_position_x_px,
            position_y_px=data.probe_position_y_px,
        )

    @staticmethod
    def _shape(value) -> tuple[int, ...]:
        if isinstance(value, torch.Tensor):
            return tuple(value.shape)
        if isinstance(value, np.ndarray):
            return value.shape
        return np.shape(value)

    @staticmethod
    def _as_numpy(value) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def _check_opr_mode_weights_shape(
        self,
        *,
        weights: Optional[TaskArray],
        n_positions: int,
        n_opr_modes: int,
    ) -> None:
        if weights is None:
            if n_opr_modes > 1:
                raise ValueError(
                    f"You have {n_opr_modes} OPR modes in `probe_data`, but "
                    "`opr_mode_weights_data` is not provided."
                )
            logging.info("Unspecified OPR weight initial guess will be automatically populated with 1s.")
            return

        weights_shape = self._shape(weights)
        if len(weights_shape) not in (1, 2):
            raise ValueError("`opr_mode_weights_data` must be a 1D or 2D array.")
        if weights_shape[-1] != n_opr_modes:
            raise ValueError(
                f"You have {n_opr_modes} OPR modes in `probe_data`, but the number of "
                f"modes in `opr_mode_weights_data` is {weights_shape[-1]}."
            )
        if len(weights_shape) == 2 and weights_shape[0] != n_positions:
            raise ValueError(
                "`opr_mode_weights_data.shape[0]` must match the number of probe positions "
                f"({weights_shape[0]} != {n_positions})."
            )
        if self.opr_mode_weight_options.optimizable:
            logging.warning(
                "The default value of OPRModeWeightsOptions has been changed to False. "
                "You have provided initial OPR weights, but optimizable is set to False. "
                "Is this intended?"
            )

    def _check_object_position_coverage(
        self,
        *,
        object_shape: tuple[int, ...],
        probe_shape: tuple[int, ...],
        position_x_px: TaskArray,
        position_y_px: TaskArray,
    ) -> None:
        pos_y = self._as_numpy(position_y_px)
        pos_x = self._as_numpy(position_x_px)
        obj_lateral_shape = object_shape[-2:]
        probe_lateral_shape = probe_shape[-2:]
        min_size = [
            int(np.ceil(pos_y.max() - pos_y.min() + probe_lateral_shape[-2])) + 2,
            int(np.ceil(pos_x.max() - pos_x.min() + probe_lateral_shape[-1])) + 2,
        ]
        if any(min_size[i] > obj_lateral_shape[i] for i in range(2)):
            logging.warning(
                f"An object tensor with a lateral size of at least {min_size} is "
                "required to avoid padding when extracting/placing patches, but the provided "
                f"object size is {list(obj_lateral_shape)}."
            )
        if (
            self.object_options.determine_position_origin_coords_by
            == api.ObjectPosOriginCoordsMethods.SUPPORT
        ):
            buffer_center = np.array([np.round(x / 2) + 0.5 for x in obj_lateral_shape])
            if (
                pos_y.max() + buffer_center[0] + probe_lateral_shape[-2] // 2 > obj_lateral_shape[-2]
                or pos_y.min() + buffer_center[0] - probe_lateral_shape[-2] // 2 < 0
                or pos_x.max() + buffer_center[1] + probe_lateral_shape[-1] // 2 > obj_lateral_shape[-1]
                or pos_x.min() + buffer_center[1] - probe_lateral_shape[-1] // 2 < 0
            ):
                logging.warning(
                    "`object_options.determine_center_coords_by` is set to `SUPPORT`. This assumes "
                    "that the probe positions are approximately zero-centered, i.e., "
                    "`-pos_y.min() ~ pos_y.max()` and `-pos_x.min() ~ pos_x.max()`. "
                    "However, the given probe positions will cause the reconstructor to access pixels "
                    "out of the object support. Please provide probe positions that are approximately "
                    "zero-centered, or set `object_options.determine_center_coords_by` to `POSITIONS`."
                )

    def build(self):
        self.build_random_seed()
        self.build_default_device()
        self.build_default_dtype()
        self.build_logger()
        self.build_data()
        self.build_object()
        self.build_probe()
        self.build_probe_positions()
        self.build_opr_mode_weights()
        self.build_reconstructor()

    def build_random_seed(self):
        if self.reconstructor_options.random_seed is not None:
            torch.manual_seed(self.reconstructor_options.random_seed)
            np.random.seed(self.reconstructor_options.random_seed)
            random.seed(self.reconstructor_options.random_seed)
        pmath.set_allow_nondeterministic_algorithms(self.reconstructor_options.allow_nondeterministic_algorithms)

    def build_default_device(self):
        accelerator_module = AcceleratorModuleWrapper.get_module()
        default_device = self._get_default_device()
        
        if self.detect_launcher() is None:
            torch.set_default_device(default_device)
        else:
            self.init_process_group()
            
            if self.backend == "nccl" and self.n_ranks > accelerator_module.device_count():
                raise ValueError(
                    f"Number of ranks ({self.n_ranks}) is greater than the number of devices "
                    f"({accelerator_module.device_count()}). This is not allowed with NCCL backend."
                )
            
            if self.n_ranks == 1:
                torch.set_default_device(default_device)
            else:
                logging.info(f"Multi-processing mode detected with {self.n_ranks} ranks.")
                torch.set_default_device(
                    f"{AcceleratorModuleWrapper.get_to_device_string()}:{self.rank % accelerator_module.device_count()}"
                )
            
        if accelerator_module.device_count() > 0:
            cuda_visible_devices_str = "(unset)"
            if "CUDA_VISIBLE_DEVICES" in os.environ.keys():
                cuda_visible_devices_str = os.environ["CUDA_VISIBLE_DEVICES"]
            logger.info(
                "Using device: {} (CUDA_VISIBLE_DEVICES=\"{}\")".format(
                    [accelerator_module.get_device_name(i) for i in range(accelerator_module.device_count())],
                    cuda_visible_devices_str,
                )
            )
        else:
            logger.info("Using device: {}".format(torch.get_default_device()))

    def _get_default_device(self) -> str:
        accelerator_module = AcceleratorModuleWrapper.get_module()
        if (
            self.reconstructor_options.default_device == api.Devices.GPU
            and not accelerator_module.is_available()
        ):
            logger.warning(
                "GPU default device was requested, but no accelerator is available. "
                "Falling back to CPU."
            )
            return "cpu"
        return maps.get_device_by_enum(self.reconstructor_options.default_device)

    def build_logger(self):
        if self.rank != 0:
            logger.setLevel(level=logging.ERROR)

    def build_default_dtype(self):
        torch.set_default_dtype(maps.get_dtype_by_enum(self.reconstructor_options.default_dtype))
        utils.set_default_complex_dtype(
            maps.get_complex_dtype_by_enum(self.reconstructor_options.default_dtype)
        )
        pmath.set_use_double_precision_for_fft(
            self.reconstructor_options.use_double_precision_for_fft
        )

    def build_data(self):
        if self.data_options.free_space_propagation_distance_m < np.inf and self.data_options.fft_shift:
            logger.warning(
                "It seems that you are reconstructing near-field data with FFT-shifted diffraction data. "
                "Is this intended? If not, set `data_options.fft_shift=False`."
            )
            
        save_on_device = self.data_options.save_data_on_device
        if self.n_ranks > 1:
            if save_on_device:
                logging.warning(
                    "Data must be saved on CPU in multi-processing mode "
                    "but `data_options.save_data_on_device` is set to `True`. "
                    "The current setting will be ignored."
                )
            save_on_device = False

        self.dataset = PtychographyDataset(
            self._task_data.diffraction_data,
            wavelength_m=self.data_options.wavelength_m,
            free_space_propagation_distance_m=self.data_options.free_space_propagation_distance_m,
            fft_shift=self.data_options.fft_shift,
            save_data_on_device=save_on_device,
            valid_pixel_mask=self._task_data.valid_pixel_mask,
        )

    def build_object(self):
        data = to_tensor(self._task_data.object_data)
        kwargs = {
            "data": data,
            "options": self.object_options,
        }
        if (
            isinstance(self.object_options, api.options.AutodiffPtychographyObjectOptions)
        ) and (
            self.object_options.experimental.deep_image_prior_options.enabled
        ):
            self.object = object.DIPPlanarObject(**kwargs)
        else:
            self.object = object.PlanarObject(**kwargs)

    def build_probe(self):
        data = to_tensor(self._task_data.probe_data)
        kwargs = {
            "data": data,
            "options": self.probe_options,
        }
        if (
            isinstance(self.probe_options, api.options.AutodiffPtychographyProbeOptions)
        ) and (
            self.probe_options.experimental.deep_image_prior_options.enabled
        ):
            self.probe = probe.DIPProbe(**kwargs)
        elif (
            isinstance(self.probe_options, api.options.PIEProbeOptions)
        ) and (
            self.probe_options.experimental.sdl_probe_options.enabled
        ):
            self.probe = probe.SynthesisDictLearnProbe(**kwargs)
        else:
            self.probe = probe.Probe(**kwargs)

    def build_probe_positions(self):
        pos_y = to_tensor(self._task_data.probe_position_y_px)
        pos_x = to_tensor(self._task_data.probe_position_x_px)
        data = torch.stack([pos_y, pos_x], dim=1)
        self.probe_positions = probepos.ProbePositions(data=data, options=self.position_options)

    def build_opr_mode_weights(self):
        if self._task_data.opr_mode_weights_data is None:
            initial_weights = torch.ones([self._shape(self._task_data.diffraction_data)[0], 1])
        else:
            initial_weights = to_tensor(self._task_data.opr_mode_weights_data)
        if initial_weights.ndim == 1:
            # If a 1D array is given, expand it to all scan points.
            initial_weights = initial_weights.unsqueeze(0).repeat(
                self._shape(self._task_data.probe_position_x_px)[0], 1
            )
        self.opr_mode_weights = oprweights.OPRModeWeights(
            data=initial_weights, options=self.opr_mode_weight_options
        )

    def build_reconstructor(self):
        par_group = paramgrp.PlanarPtychographyParameterGroup(
            object=self.object,
            probe=self.probe,
            probe_positions=self.probe_positions,
            opr_mode_weights=self.opr_mode_weights,
        )

        if self.n_ranks == 1:
            reconstructor_class = maps.get_reconstructor_by_enum(
                self.reconstructor_options.get_reconstructor_type()
            )
        else:
            reconstructor_class = maps.get_multiprocess_reconstructor_by_enum(
                self.reconstructor_options.get_reconstructor_type()
            )

        reconstructor_kwargs = {
            "parameter_group": par_group,
            "dataset": self.dataset,
            "options": self.reconstructor_options,
        }

        self.reconstructor = reconstructor_class(**reconstructor_kwargs)
        self.reconstructor.build()

    def run(self, n_epochs: int = None, reset_timer_globals: bool = True):
        """
        Run reconstruction either for `n_epochs` (if given), or for the number of epochs given
        in the options. The internal states of the Task object persists when this function
        finishes. To run more epochs continuing from the last run, call this function again.

        Parameters
        ----------
        n_epochs : int, optional
            The number of epochs to run. If None, use the number of epochs specified in the
            option object.
        reset_timer_globals : bool, optional
            When True (default) the global timing accumulators are cleared before the run. Set to
            False to continue accumulating timing data across successive calls.
        """
        if movies.MOVIES_INSTALLED and self.reconstructor.current_epoch == 0:
            movies.api.reset_movie_builders()
        if reset_timer_globals:
            timer_utils.clear_timer_globals()
        self.reconstructor.run(n_epochs=n_epochs)

    def get_data(
        self, name: Literal["object", "probe", "probe_positions", "opr_mode_weights"]
    ) -> Tensor:
        """Get a detached copy of the data of the given name.

        Parameters
        ----------
        name : Literal["object", "probe", "probe_positions", "opr_mode_weights"]
            The name of the data to get.

        Returns
        -------
        Tensor
            The data of the given name.
        """
        # Deep image prior objects and probes need to be generated 
        # before fetching to avoid issues with multi-GPU.
        if name == "object" and isinstance(self.object, object.DIPPlanarObject):
            self.object.generate()
        elif name == "probe" and isinstance(self.probe, probe.DIPProbe):
            self.probe.generate()
        return getattr(self, name).data.detach()

    def get_data_to_cpu(
        self,
        name: Literal["object", "probe", "probe_positions", "opr_mode_weights"],
        as_numpy: bool = False,
    ) -> Union[Tensor, ndarray]:
        data = self.get_data(name).cpu()
        if as_numpy:
            data = data.numpy()
        return data
    
    def get_probe_positions_y(self, as_numpy: bool = False) -> Union[Tensor, ndarray]:
        data = self.probe_positions.data[:, 0].detach()
        if as_numpy:
            data = data.cpu().numpy()
        return data

    def get_probe_positions_x(self, as_numpy: bool = False) -> Union[Tensor, ndarray]:
        data = self.probe_positions.data[:, 1].detach()
        if as_numpy:
            data = data.cpu().numpy()
        return data
    
    def copy_data_from_task(
        self, 
        task: "PtychographyTask",
        params_to_copy: tuple[str, ...] = ("object", "probe", "probe_positions", "opr_mode_weights")
    ) -> None:
        """Copy data of reconstruction parameters from another task object.

        Parameters
        ----------
        task : PtychographyTask
            The task object to copy from.
        params_to_copy : tuple[str, ...], optional
            The parameters to copy. By default, copy all parameters.
        """
        with torch.no_grad():
            for param in params_to_copy:
                if param == "object":
                    self.reconstructor.parameter_group.object.set_data(
                        task.get_data("object")
                    )
                elif param == "probe":
                    self.reconstructor.parameter_group.probe.set_data(
                        task.get_data("probe")
                    )
                elif param == "probe_positions":
                    self.reconstructor.parameter_group.probe_positions.set_data(
                        task.get_data("probe_positions")
                    )
                elif param == "opr_mode_weights":
                    self.reconstructor.parameter_group.opr_mode_weights.set_data(
                        task.get_data("opr_mode_weights")
                    )
                else:
                    raise ValueError(f"Invalid parameter name: {param}")

    def set_large_tensor_device(
        self,
        device: Literal["cpu", "cuda"] | torch.device | None = None,
    ) -> None:
        """Move task tensors and reconstruction state between devices.

        This helper is aimed at multi-task workflows where only one task is
        active on the accelerator at a time. Call it with ``device="cpu"`` to
        offload task tensors and optimizer/reconstructor state to system memory,
        and call it again before resuming the task to restore that state to the
        accelerator.

        Parameters
        ----------
        device : str | torch.device | None, optional
            Target device for task state. If None, tensors are moved back
            to the current default device. If a string is given, it must be either
            "cpu" or "cuda".
        """

        if device is None:
            device = torch.get_default_device()
        device = torch.device(device)

        if self.reconstructor is None or self.dataset is None:
            raise RuntimeError("Task is not built yet.")

        parameter_group = self.reconstructor.parameter_group
        with torch.no_grad():
            for parameter in parameter_group.get_all_reconstruct_parameters():
                parameter.to(device)
                if parameter.optimizer is not None:
                    for key, value in list(parameter.optimizer.state.items()):
                        parameter.optimizer.state[key] = utils.move_nested_tensors_to_device(
                            value, device
                        )

            self.dataset.patterns = self.dataset.patterns.to(device)
            self.dataset.move_attributes_to_device(device)
            self.dataset.save_data_on_device = device.type != "cpu"

            buffers = self.reconstructor.reconstructor_buffers
            for name in buffers.get_all_names():
                if not hasattr(buffers, name):
                    continue
                buffers.__setattr__(
                    name,
                    utils.move_nested_tensors_to_device(getattr(buffers, name), device),
                    bypass_check=True,
                )

            for name, value in vars(self.reconstructor).items():
                if not name.endswith("_momentum_params") or not dataclasses.is_dataclass(value):
                    continue
                for field in dataclasses.fields(value):
                    setattr(
                        value,
                        field.name,
                        utils.move_nested_tensors_to_device(getattr(value, field.name), device),
                    )

            self.reconstructor.forward_model.move_intermediate_variables_to_device(device)

        if device.type == "cpu":
            AcceleratorModuleWrapper.get_module().empty_cache()
                
    def get_options_as_dict(self) -> dict:
        return self.options.get_dict()
    
    def load_options_from_dict(self, d: dict) -> None:
        self.options.load_from_dict(d)
        self.data_options = self.options.data_options
        self.object_options = self.options.object_options
        self.probe_options = self.options.probe_options
        self.position_options = self.options.probe_position_options
        self.opr_mode_weight_options = self.options.opr_mode_weight_options
        self.reconstructor_options = self.options.reconstructor_options

    def __exit__(self, exc_type, exc_value, exc_tb):
        del self.object
        del self.probe
        del self.probe_positions
        del self.opr_mode_weights
        del self.reconstructor
        del self.dataset

        super().__exit__(exc_type, exc_value, exc_tb)
