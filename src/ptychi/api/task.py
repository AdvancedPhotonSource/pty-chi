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
from ptychi.data_structures.base import DummyParameter
from ptychi.ptychotorch.io_handles import PtychographyDataset
from ptychi.ptychotorch.utils import to_tensor
import ptychi.ptychotorch.utils as utils


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
        torch.cuda.empty_cache()


class PtychographyTask(Task):
    def __init__(self, options: api.options.task.PtychographyTaskOptions, *args, **kwargs) -> None:
        super().__init__(options, *args, **kwargs)
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
        self.reconstructor = None

        self.build()

    def build(self):
        self.build_random_seed()
        self.build_logger()
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

    def build_logger(self):
        logging.basicConfig(level=self.reconstructor_options.log_level)

    def build_default_device(self):
        torch.set_default_device(maps.get_device_by_enum(self.reconstructor_options.default_device))
        if len(self.reconstructor_options.gpu_indices) > 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
                map(str, self.reconstructor_options.gpu_indices)
            )

    def build_default_dtype(self):
        torch.set_default_dtype(maps.get_dtype_by_enum(self.reconstructor_options.default_dtype))
        utils.set_default_complex_dtype(
            maps.get_complex_dtype_by_enum(self.reconstructor_options.default_dtype)
        )

    def build_data(self):
        self.dataset = PtychographyDataset(
            self.data_options.data, wavelength_m=self.data_options.wavelength_m
        )

    def build_object(self):
        data = to_tensor(self.object_options.initial_guess)
        kwargs = {
            "data": data,
            "options": self.object_options,
        }
        self.object = object.PlanarObject(**kwargs)

    def build_probe(self):
        data = to_tensor(self.probe_options.initial_guess)
        self.probe = probe.Probe(data=data, options=self.probe_options)

    def build_probe_positions(self):
        pos_y = to_tensor(self.position_options.position_y_m)
        pos_x = to_tensor(self.position_options.position_x_m)
        data = torch.stack([pos_y, pos_x], dim=1)
        data = data / self.position_options.pixel_size_m
        self.probe_positions = probepos.ProbePositions(data=data, options=self.position_options)

    def build_opr_mode_weights(self):
        n_opr_modes = self.probe_options.initial_guess.shape[0]
        if n_opr_modes == 1:
            self.opr_mode_weights = DummyParameter()
            return
        if self.opr_mode_weight_options.initial_weights is None:
            self.opr_mode_weights = DummyParameter()
            return
        else:
            initial_weights = to_tensor(self.opr_mode_weight_options.initial_weights)
            if self.opr_mode_weight_options.initial_weights.ndim == 1:
                # If a 1D array is given, expand it to all scan points.
                initial_weights = initial_weights.unsqueeze(0).repeat(
                    len(self.position_options.position_x_m), 1
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

    def run(self):
        self.reconstructor.run()

    def iterate(self, n_epochs: int):
        self.reconstructor.run(n_epochs=n_epochs)

    def get_data(
        self, name: Literal["object", "probe", "probe_positions", "opr_mode_weights"]
    ) -> Tensor:
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

    def __exit__(self, exc_type, exc_value, exc_tb):
        del self.object
        del self.probe
        del self.probe_positions
        del self.opr_mode_weights
        del self.reconstructor
        del self.dataset

        super().__exit__(exc_type, exc_value, exc_tb)
