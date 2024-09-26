from typing import Optional, Literal, Union
import random
import logging
import os

import torch
import numpy as np
from torch import Tensor
from numpy import ndarray

import ptychointerim.configs as configs
from ptychointerim.ptychotorch.data_structures import *
from ptychointerim.ptychotorch.io_handles import PtychographyDataset
from ptychointerim.forward_models import Ptychography2DForwardModel
from ptychointerim.ptychotorch.utils import to_tensor
import ptychointerim.ptychotorch.utils as utils
from ptychointerim.ptychotorch.reconstructors import *
from ptychointerim.metrics import MSELossOfSqrt


def get_optimizer_class(optimizer_name: str):
    if optimizer_name == 'adam':
        return torch.optim.Adam
    elif optimizer_name == 'sgd':
        return torch.optim.SGD
    elif optimizer_name == 'rmsprop':
        return torch.optim.RMSprop
    elif optimizer_name == 'adagrad':
        return torch.optim.Adagrad
    elif optimizer_name == 'adadelta':
        return torch.optim.Adadelta
    elif optimizer_name == 'lbfgs':
        return torch.optim.LBFGS
    elif optimizer_name == 'asgd':
        return torch.optim.ASGD
    elif optimizer_name == 'sparse_adam':
        return torch.optim.SparseAdam
    elif optimizer_name == 'adamax':
        return torch.optim.Adamax
    elif optimizer_name == 'radam':
        return torch.optim.RAdam
    elif optimizer_name == 'adamw':
        return torch.optim.AdamW
    else:
        raise ValueError(f'Unknown optimizer: {optimizer_name}')
    

def get_loss_function(loss_name: str):
    if loss_name == 'mse':
        return torch.nn.MSELoss()
    elif loss_name == 'poisson':
        return torch.nn.PoissonNLLLoss()
    elif loss_name == 'mse_sqrt':
        return MSELossOfSqrt()
    
    
def get_reconstructor_class(config_class: Type[configs.Config]):
    if config_class is configs.LSQMLReconstructorConfig:
        return LSQMLReconstructor
    elif config_class is configs.AutodiffPtychographyReconstructorConfig:
        return AutodiffPtychographyReconstructor
    elif config_class is configs.PIEReconstructorConfig:
        return EPIEReconstructor


class Job:
    
    def __init__(self, *args, **kwargs) -> None:
        pass
    

class PtychographyJob(Job):
    
    def __init__(self,
                 config: configs.PtychographyJobConfig,
                 *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.data_config = config.data_config
        self.object_config = config.object_config
        self.probe_config = config.probe_config
        self.position_config = config.probe_position_config
        self.opr_mode_weight_config = config.opr_mode_weight_config
        self.reconstructor_config = config.reconstructor_config
        
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
        if self.reconstructor_config.random_seed is not None:
            torch.manual_seed(self.reconstructor_config.random_seed)
            np.random.seed(self.reconstructor_config.random_seed)
            random.seed(self.reconstructor_config.random_seed)
            
    def build_logger(self):
        ldict = {'error': logging.ERROR, 'warning': logging.WARNING, 'info': logging.INFO, 'debug': logging.DEBUG}
        logging.basicConfig(level=ldict[self.reconstructor_config.log_level])
    
    def build_default_device(self):
        dmap = {'cpu': 'cpu', 'gpu': 'cuda'}
        torch.set_default_device(dmap[self.reconstructor_config.default_device])
        if self.reconstructor_config.gpu_indices is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, self.reconstructor_config.gpu_indices))
        
    def build_default_dtype(self):
        dmap = {'float32': torch.float32, 'float64': torch.float64}
        cdmap = {'float32': torch.complex64, 'float64': torch.complex128}
        torch.set_default_dtype(dmap[self.reconstructor_config.default_dtype])
        utils.set_default_complex_dtype(cdmap[self.reconstructor_config.default_dtype])
        
    def build_data(self):
        self.dataset = PtychographyDataset(self.data_config.data)
        
    def build_object(self):
        data = to_tensor(self.object_config.initial_guess)
        self.object = Object2D(
            data=data, 
            pixel_size_m=self.object_config.pixel_size_m,
            optimizable=self.object_config.optimizable,
            optimizer_class=get_optimizer_class(self.object_config.optimizer),
            optimizer_params={'lr': self.object_config.step_size},
            **self.object_config.uninherited_fields()
        )
        
    def build_probe(self):
        data = to_tensor(self.probe_config.initial_guess)
        self.probe = Probe(
            data=data, 
            optimizable=self.probe_config.optimizable,
            optimizer_class=get_optimizer_class(self.probe_config.optimizer),
            optimizer_params={'lr': self.probe_config.step_size},
            **self.probe_config.uninherited_fields()
        )
        
    def build_probe_positions(self):
        pos_y = to_tensor(self.position_config.position_y_m)
        pos_x = to_tensor(self.position_config.position_x_m)
        data = torch.stack([pos_y, pos_x], dim=1)
        data = data / self.position_config.pixel_size_m

        self.probe_positions = ProbePositions(
            data=data,
            optimizable=self.position_config.optimizable,
            optimizer_class=get_optimizer_class(self.position_config.optimizer),
            optimizer_params={'lr': self.position_config.step_size},
            **self.position_config.uninherited_fields()
        )
        
    def build_opr_mode_weights(self):
        n_opr_modes = self.probe_config.initial_guess.shape[0]
        if n_opr_modes == 1:
            self.opr_mode_weights = DummyVariable()
        else:
            self.opr_mode_weights = OPRModeWeights(
                data=utils.generate_initial_opr_mode_weights(
                    n_points=len(self.position_config.position_x_m), 
                    n_opr_modes=n_opr_modes, 
                    eigenmode_weight=self.opr_mode_weight_config.initial_eigenmode_weights),
                optimizable=self.opr_mode_weight_config.optimizable,
                optimize_intensity_variation=self.opr_mode_weight_config.optimize_intensity_variation,
                **self.opr_mode_weight_config.uninherited_fields()
            )
            
    def build_reconstructor(self):
        var_group = Ptychography2DVariableGroup(
            object=self.object,
            probe=self.probe,
            probe_positions=self.probe_positions,
            opr_mode_weights=self.opr_mode_weights
        )
        
        reconstructor_class = get_reconstructor_class(self.reconstructor_config.__class__)
        
        reconstructor_kwargs = {
            'variable_group': var_group,
            'dataset': self.dataset,
            'batch_size': self.reconstructor_config.batch_size,
            'n_epochs': self.reconstructor_config.num_epochs,
            'metric_function': get_loss_function(self.reconstructor_config.metric_function),
            **self.reconstructor_config.uninherited_fields()
        }
        # Special treatments.
        if reconstructor_class == AutodiffPtychographyReconstructor:
            reconstructor_kwargs['forward_model_class'] = Ptychography2DForwardModel
            reconstructor_kwargs['loss_function'] = get_loss_function(self.reconstructor_config.loss_function)
        
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
        