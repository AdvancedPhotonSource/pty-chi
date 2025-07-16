# Copyright © 2025 UChicago Argonne, LLC All right reserved
# Full license accessible at https://github.com//AdvancedPhotonSource/pty-chi/blob/main/LICENSE

from typing import Literal, Union, overload
from types import TracebackType
import random
import logging
import os

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

logger = logging.getLogger(__name__)


class Task:
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
    def __init__(self, options: api.options.task.PtychographyTaskOptions, *args, **kwargs) -> None:
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

        self.check_options()
        self.build()
        
    def check_options(self):
        self.options.check()

    def build(self):
        self.build_random_seed()
        self.build_default_device()
        self.build_default_dtype()
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
        
        torch.set_default_device(maps.get_device_by_enum(self.reconstructor_options.default_device))
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
        self.dataset = PtychographyDataset(
            self.data_options.data, 
            wavelength_m=self.data_options.wavelength_m,
            free_space_propagation_distance_m=self.data_options.free_space_propagation_distance_m,
            fft_shift=self.data_options.fft_shift,
            save_data_on_device=self.data_options.save_data_on_device,
            valid_pixel_mask=self.data_options.valid_pixel_mask
        )

    def build_object(self):
        data = to_tensor(self.object_options.initial_guess)
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
        data = to_tensor(self.probe_options.initial_guess)
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
        pos_y = to_tensor(self.position_options.position_y_px)
        pos_x = to_tensor(self.position_options.position_x_px)
        data = torch.stack([pos_y, pos_x], dim=1)
        self.probe_positions = probepos.ProbePositions(data=data, options=self.position_options)

    def build_opr_mode_weights(self):
        if self.opr_mode_weight_options.initial_weights is None:
            initial_weights = torch.ones([self.data_options.data.shape[0], 1])
        else:
            initial_weights = to_tensor(self.opr_mode_weight_options.initial_weights)
        if initial_weights.ndim == 1:
            # If a 1D array is given, expand it to all scan points.
            initial_weights = initial_weights.unsqueeze(0).repeat(
                len(self.position_options.position_x_px), 1
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

        reconstructor_class = maps.get_reconstructor_by_enum(
            self.reconstructor_options.get_reconstructor_type()
        )

        reconstructor_kwargs = {
            "parameter_group": par_group,
            "dataset": self.dataset,
            "options": self.reconstructor_options,
        }

        self.reconstructor = reconstructor_class(**reconstructor_kwargs)
        self.reconstructor.build()

    def run(self, n_epochs: int = None):
        """
        Run reconstruction either for `n_epochs` (if given), or for the number of epochs given
        in the options. The internal states of the Task object persists when this function
        finishes. To run more epochs continuing from the last run, call this function again.

        Parameters
        ----------
        n_epochs : int, optional
            The number of epochs to run. If None, use the number of epochs specified in the
            option object.
        """
        if movies.MOVIES_INSTALLED and self.reconstructor.current_epoch == 0:
            movies.api.reset_movie_builders()
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
