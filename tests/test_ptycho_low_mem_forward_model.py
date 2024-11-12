import argparse

import torch
import numpy as np

import ptychi.api as api
from ptychi.api.task import PtychographyTask
from ptychi.ptychotorch.utils import get_suggested_object_size, get_default_complex_dtype, generate_initial_opr_mode_weights

import test_utils as tutils


def test_ptycho_low_mem_forward_model(pytestconfig, generate_gold=False, debug=False, high_tol=False):
    if pytestconfig is not None:
        high_tol = pytestconfig.getoption("high_tol")
        
    name = 'test_ptycho_low_mem_forward_model'
    
    tutils.setup(name, cpu_only=False, gpu_indices=[0])
    
    data, probe, pixel_size_m, positions_px = tutils.load_tungsten_data(pos_type='true')

    options = api.LSQMLOptions()
    options.data_options.data = data
        
    options.object_options.initial_guess = torch.ones([3, *get_suggested_object_size(positions_px, probe.shape[-2:], extra=100)], dtype=get_default_complex_dtype())
    options.object_options.pixel_size_m = pixel_size_m
    options.object_options.slice_spacings_m = np.array([3e-6, 3e-6])
    options.object_options.optimizable = True
    options.object_options.optimizer = api.Optimizers.SGD
    options.object_options.step_size = 1
    
    options.probe_options.initial_guess = probe
    options.probe_options.optimizable = True
    options.probe_options.optimizer = api.Optimizers.SGD
    options.probe_options.step_size = 1

    options.probe_position_options.position_x_px = positions_px[:, 1]
    options.probe_position_options.position_y_px = positions_px[:, 0]
    options.probe_position_options.optimizable = False
    
    options.reconstructor_options.batch_size = 96
    options.reconstructor_options.noise_model = api.NoiseModels.GAUSSIAN
    options.reconstructor_options.num_epochs = 8
    
    task = PtychographyTask(options)
    fm = task.reconstructor.forward_model
    
    y_normal = fm.forward(torch.tensor([0, 1]).long())
    y_lowmem = fm.forward_low_memory(torch.tensor([0, 1]).long())
    
    assert torch.allclose(y_normal, y_lowmem)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate-gold', action='store_true')
    parser.add_argument('--high-tol', action='store_true')
    args = parser.parse_args()

    test_ptycho_low_mem_forward_model(None, generate_gold=args.generate_gold, debug=True, high_tol=args.high_tol)
