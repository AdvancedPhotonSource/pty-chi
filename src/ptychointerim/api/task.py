from typing import Literal, Union, overload
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
from ptychointerim.ptychotorch.data_structures import (Object2D, MultisliceObject, Probe, ProbePositions, OPRModeWeights,
                                                       DummyParameter, Ptychography2DParameterGroup)
from ptychointerim.ptychotorch.io_handles import PtychographyDataset
from ptychointerim.forward_models import Ptychography2DForwardModel, MultislicePtychographyForwardModel
from ptychointerim.ptychotorch.utils import to_tensor
import ptychointerim.ptychotorch.utils as utils
from ptychointerim.ptychotorch.reconstructors import AutodiffPtychographyReconstructor


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
        torch.set_default_device(maps.device_dict[self.reconstructor_options.default_device])
        if len(self.reconstructor_options.gpu_indices) > 0:
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, self.reconstructor_options.gpu_indices))
        
    def build_default_dtype(self):
        torch.set_default_dtype(maps.dtype_dict[self.reconstructor_options.default_dtype])
        utils.set_default_complex_dtype(maps.complex_dtype_dict[self.reconstructor_options.default_dtype])
        
    def build_data(self):
        self.dataset = PtychographyDataset(self.data_options.data, wavelength_m=self.data_options.wavelength_m)
        
    def build_object(self):
        data = to_tensor(self.object_options.initial_guess)
        kwargs = {
            'data': data, 
            'pixel_size_m': self.object_options.pixel_size_m,
            'optimizable': self.object_options.optimizable,
            'optimization_plan': self.object_options.optimization_plan,
            'optimizer_class': maps.optimizer_dict[self.object_options.optimizer],
            'optimizer_params': dict({'lr': self.object_options.step_size}, **self.object_options.optimizer_params),
            'l1_norm_constraint_weight': self.object_options.l1_norm_constraint_weight,
            'l1_norm_constraint_stride': self.object_options.l1_norm_constraint_stride,
            'smoothness_constraint_alpha': self.object_options.smoothness_constraint_alpha,
            'smoothness_constraint_stride': self.object_options.smoothness_constraint_stride,
            'total_variation_weight': self.object_options.total_variation_weight,
            'total_variation_stride': self.object_options.total_variation_stride
        }
        kwargs.update(self.object_options.uninherited_fields())
        if self.object_options.type == api.ObjectTypes.MULTISLICE:
            kwargs['slice_spacings_m'] = self.object_options.slice_spacings_m
        if self.object_options.type == api.ObjectTypes.TWO_D:
            self.object = Object2D(**kwargs)
        elif self.object_options.type == api.ObjectTypes.MULTISLICE:
            self.object = MultisliceObject(**kwargs)
        
    def build_probe(self):
        data = to_tensor(self.probe_options.initial_guess)
        self.probe = Probe(
            data=data, 
            optimizable=self.probe_options.optimizable,
            optimization_plan=self.probe_options.optimization_plan,
            optimizer_class=maps.optimizer_dict[self.probe_options.optimizer],
            optimizer_params=dict({'lr': self.probe_options.step_size}, **self.probe_options.optimizer_params),
            probe_power=self.probe_options.probe_power,
            probe_power_constraint_stride=self.probe_options.probe_power_constraint_stride,
            orthogonalize_incoherent_modes=self.probe_options.orthogonalize_incoherent_modes,
            orthogonalize_incoherent_modes_stride=self.probe_options.orthogonalize_incoherent_modes_stride,
            orthogonalize_incoherent_modes_method=self.probe_options.orthogonalize_incoherent_modes_method,
            orthogonalize_opr_modes=self.probe_options.orthogonalize_opr_modes,
            orthogonalize_opr_modes_stride=self.probe_options.orthogonalize_opr_modes_stride,
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
            optimization_plan=self.position_options.optimization_plan,
            optimizer_class=maps.optimizer_dict[self.position_options.optimizer],
            optimizer_params=dict({'lr': self.position_options.step_size}, **self.position_options.optimizer_params),
            pixel_size_m=self.position_options.pixel_size_m,
            update_magnitude_limit=self.position_options.update_magnitude_limit,
            **self.position_options.uninherited_fields()
        )
        
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
                initial_weights = initial_weights.unsqueeze(0).repeat(len(self.position_options.position_x_m), 1)
            self.opr_mode_weights = OPRModeWeights(
                data=initial_weights,
                optimizable=self.opr_mode_weight_options.optimizable,
                optimization_plan=self.opr_mode_weight_options.optimization_plan,
                optimizer_class=maps.optimizer_dict[self.opr_mode_weight_options.optimizer],
                optimizer_params=dict({'lr': self.opr_mode_weight_options.step_size}, **self.opr_mode_weight_options.optimizer_params),
                optimize_eigenmode_weights=self.opr_mode_weight_options.optimize_eigenmode_weights,
                optimize_intensity_variation=self.opr_mode_weight_options.optimize_intensity_variation,
                **self.opr_mode_weight_options.uninherited_fields()
            )
            
    def build_reconstructor(self):
        par_group = Ptychography2DParameterGroup(
            object=self.object,
            probe=self.probe,
            probe_positions=self.probe_positions,
            opr_mode_weights=self.opr_mode_weights
        )
        
        reconstructor_class = maps.reconstructor_dict[self.reconstructor_options.get_reconstructor_type()]
        
        reconstructor_kwargs = {
            'parameter_group': par_group,
            'dataset': self.dataset,
            'batch_size': self.reconstructor_options.batch_size,
            'n_epochs': self.reconstructor_options.num_epochs,
            'metric_function': (None if self.reconstructor_options.metric_function is None 
                                else maps.loss_function_dict[self.reconstructor_options.metric_function]()),
            **self.reconstructor_options.uninherited_fields()
        }
        # Special handling. We should change the expected input type of the reconstructor so that no conversion
        # needs to be done here. 
        if reconstructor_class == AutodiffPtychographyReconstructor:
            if self.object_options.type == api.ObjectTypes.TWO_D:
                reconstructor_kwargs['forward_model_class'] = Ptychography2DForwardModel
            elif self.object_options.type == api.ObjectTypes.MULTISLICE:
                reconstructor_kwargs['forward_model_class'] = MultislicePtychographyForwardModel
                reconstructor_kwargs['forward_model_params'] = {
                    'wavelength_m': self.data_options.wavelength_m, 
                    'propagation_distance_m': self.data_options.propagation_distance_m
                }
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
        