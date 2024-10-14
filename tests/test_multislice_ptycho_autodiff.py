import argparse

import torch
import numpy as np

import ptychointerim.api as api
from ptychointerim.api.task import PtychographyTask
from ptychointerim.ptychotorch.utils import get_suggested_object_size, get_default_complex_dtype

import test_utils as tutils


def test_multislice_ptycho_autodiff(generate_gold=False, debug=False):
    name = 'test_multislice_ptycho_autodiff'
    
    tutils.setup(name, cpu_only=True, gpu_indices=[0])
    
    data, probe, pixel_size_m, positions_px = tutils.load_data_ptychodus(
        *tutils.get_default_input_data_file_paths('multislice_ptycho_AuNi'),
        subtract_position_mean=True
    )
    wavelength_m = 1.03e-10
    
    options = api.AutodiffPtychographyOptions()
    
    options.data_options.data = data
    options.data_options.wavelength_m = wavelength_m
    
    options.object_options.initial_guess = torch.ones([2, *get_suggested_object_size(positions_px, probe.shape[-2:], extra=50)], dtype=get_default_complex_dtype())
    options.object_options.pixel_size_m = pixel_size_m
    options.object_options.type = api.ObjectTypes.MULTISLICE
    options.object_options.slice_spacings_m = np.array([1e-5])
    options.object_options.optimizable = True
    options.object_options.optimizer = api.Optimizers.ADAM
    options.object_options.step_size = 1e-3
    
    options.probe_options.initial_guess = probe
    options.probe_options.optimizable = True
    options.probe_options.optimizer = api.Optimizers.ADAM
    options.probe_options.step_size = 1e-3
    
    options.probe_position_options.position_x_m = positions_px[:, 1]
    options.probe_position_options.position_y_m = positions_px[:, 0]
    options.probe_position_options.optimizable = True
    options.probe_position_options.optimizer = api.Optimizers.SGD
    options.probe_position_options.step_size = 1e-1
    options.probe_position_options.update_magnitude_limit = 1.0
    
    options.reconstructor_options.loss_function = api.LossFunctions.MSE_SQRT
    options.reconstructor_options.batch_size = 101
    options.reconstructor_options.num_epochs = 32
    options.reconstructor_options.default_device = api.Devices.CPU
    options.reconstructor_options.random_seed = 123
    
    task = PtychographyTask(options)
    task.run()

    recon = task.get_data_to_cpu('object', as_numpy=True)
    
    if generate_gold:
        tutils.save_gold_data(name, recon)
    else:
        tutils.run_comparison(name, recon)
        
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate-gold', action='store_true')
    args = parser.parse_args()

    test_multislice_ptycho_autodiff(generate_gold=args.generate_gold, debug=True)
