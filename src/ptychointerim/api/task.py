from typing import Optional, Literal, Union, overload
from types import TracebackType
import random
import logging
import os

import torch
import numpy as np
from torch import Tensor
from numpy import ndarray

import ptychointerim.api as api
import ptychointerim.maps as maps
from ptychointerim.ptychotorch.data_structures import *
from ptychointerim.ptychotorch.io_handles import PtychographyDataset
from ptychointerim.forward_models import Ptychography2DForwardModel
from ptychointerim.ptychotorch.utils import to_tensor
import ptychointerim.ptychotorch.utils as utils
from ptychointerim.ptychotorch.reconstructors import *


class Task:
    
    def __init__(self, options: api.options.base.TaskOptions, *args, **kwargs) -> None:
        pass
    
    def __enter__(self) -> 'Task':
        return self
    
    @overload
    def __exit__(self, exception_type: None, exception_value: None, traceback: None) -> None:
        ...
 
    @overload
    def __exit__(self, exception_type: type[BaseException], exception_value: BaseException,
                 traceback: TracebackType) -> None:
        ...
    
    def __exit__(self, exception_type: type[BaseException] | None,
                 exception_value: BaseException | None, traceback: TracebackType | None) -> None:
        torch.cuda.empty_cache()
    

class PtychographyTask(Task):
    
    def __init__(self,
                 options: api.options.task.PtychographyTaskOptions,
                 *args, **kwargs
    ) -> None:
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
        torch.set_default_device(maps.device_dict[self.reconstructor_options.default_device])
        if len(self.reconstructor_options.gpu_indices) > 0:
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, self.reconstructor_options.gpu_indices))
        
    def build_default_dtype(self):
        torch.set_default_dtype(maps.dtype_dict[self.reconstructor_options.default_dtype])
        utils.set_default_complex_dtype(maps.complex_dtype_dict[self.reconstructor_options.default_dtype])
        
    def build_data(self):
        self.dataset = PtychographyDataset(self.data_options.data)
        
    def build_object(self):
        data = to_tensor(self.object_options.initial_guess)
        self.object = Object2D(
            data=data, 
            pixel_size_m=self.object_options.pixel_size_m,
            optimizable=self.object_options.optimizable,
            optimizer_class=maps.optimizer_dict[self.object_options.optimizer],
            optimizer_params={'lr': self.object_options.step_size},
            **self.object_options.uninherited_fields()
        )
        
    def build_probe(self):
        data = to_tensor(self.probe_options.initial_guess)
        self.probe = Probe(
            data=data, 
            optimizable=self.probe_options.optimizable,
            optimizer_class=maps.optimizer_dict[self.probe_options.optimizer],
            optimizer_params={'lr': self.probe_options.step_size},
            **self.probe_options.uninherited_fields()
        )
        
    def build_probe_positions(self):
        pos_y = to_tensor(self.position_options.position_y_m)
        pos_x = to_tensor(self.position_options.position_x_m)
        data = torch.stack([pos_y, pos_x], dim=1)
        data = data / self.position_options.pixel_size_m

        self.probe_positions = ProbePositions(
            data=data,
            optimizable=self.position_options.optimizable,
            optimizer_class=maps.optimizer_dict[self.position_options.optimizer],
            optimizer_params={'lr': self.position_options.step_size},
            **self.position_options.uninherited_fields()
        )
        
    def build_opr_mode_weights(self):
        n_opr_modes = self.probe_options.initial_guess.shape[0]
        if n_opr_modes == 1:
            self.opr_mode_weights = DummyVariable()
            return
        if self.opr_mode_weight_options.initial_weights is None:
            self.opr_mode_weights = DummyVariable()
            return
        else:
            initial_weights = to_tensor(self.opr_mode_weight_options.initial_weights)
            if self.opr_mode_weight_options.initial_weights.ndim == 1:
                # If a 1D array is given, expand it to all scan points.
                initial_weights = initial_weights.unsqueeze(0).repeat(len(self.position_options.position_x_m), 1)
            self.opr_mode_weights = OPRModeWeights(
                data=initial_weights,
                optimizable=self.opr_mode_weight_options.optimizable,
                optimizer_class=maps.optimizer_dict[self.opr_mode_weight_options.optimizer],
                optimizer_params={'lr': self.opr_mode_weight_options.step_size},
                optimize_intensity_variation=self.opr_mode_weight_options.optimize_intensity_variation,
                **self.opr_mode_weight_options.uninherited_fields()
            )
            
    def build_reconstructor(self):
        var_group = Ptychography2DVariableGroup(
            object=self.object,
            probe=self.probe,
            probe_positions=self.probe_positions,
            opr_mode_weights=self.opr_mode_weights
        )
        
        reconstructor_class = maps.reconstructor_dict[self.reconstructor_options.get_reconstructor_type()]
        
        reconstructor_kwargs = {
            'variable_group': var_group,
            'dataset': self.dataset,
            'batch_size': self.reconstructor_options.batch_size,
            'n_epochs': self.reconstructor_options.num_epochs,
            'metric_function': maps.loss_function_dict[self.reconstructor_options.metric_function](),
            **self.reconstructor_options.uninherited_fields()
        }
        # Special handling. We should change the expected input type of the reconstructor so that no conversion
        # needs to be done here. 
        if reconstructor_class == AutodiffPtychographyReconstructor:
            reconstructor_kwargs['forward_model_class'] = Ptychography2DForwardModel
            reconstructor_kwargs['loss_function'] = maps.loss_function_dict[self.reconstructor_options.loss_function]()
        
        self.reconstructor = reconstructor_class(**reconstructor_kwargs)
        self.reconstructor.build()
        
    def run(self):
        self.reconstructor.run()
        
    def iterate(self, n_epochs: int):
        self.reconstructor.run(n_epochs=n_epochs)
        
    def get_data(self, name: Literal['object', 'probe', 'probe_positions', 'opr_mode_weights']) -> Tensor:
        return getattr(self, name).data.detach()
        
    def get_data_to_cpu(self, 
                        name: Literal['object', 'probe', 'probe_positions', 'opr_mode_weights'], 
                        as_numpy: bool = False) -> Union[Tensor, ndarray]:
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
        