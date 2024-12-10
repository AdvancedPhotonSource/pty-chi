import argparse
import logging

import torch
import numpy as np

import ptychi.api as api
from ptychi.api.task import PtychographyTask
from ptychi.api import LSQMLOptions
import ptychi.utils as utils

import test_utils as tutils


def test_2d_ptycho_opt_plan(pytestconfig, generate_gold=False, debug=False, high_tol=False):
    if pytestconfig is not None:
        high_tol = pytestconfig.getoption("high_tol")
        
    name = 'test_2d_ptycho_opt_plan'
    
    tutils.setup(name, cpu_only=False, gpu_indices=[0])

    data, probe, pixel_size_m, positions_px = tutils.load_tungsten_data(pos_type='true', additional_opr_modes=3)
    object_init = torch.ones(
        [1, *utils.get_suggested_object_size(positions_px, probe.shape[-2:], extra=100)], 
        dtype=utils.get_default_complex_dtype()
    )
    
    options = LSQMLOptions()
    options.data_options.data = data
    
    options.object_options.initial_guess = object_init
    options.object_options.pixel_size_m = pixel_size_m
    options.object_options.optimizable = True
    options.object_options.optimizer = api.Optimizers.SGD
    options.object_options.step_size = 1
    
    options.probe_options.initial_guess = probe
    options.probe_options.optimizable = True
    options.probe_options.optimization_plan.start = 2
    options.probe_options.optimization_plan.end = None
    options.probe_options.optimizer = api.Optimizers.SGD
    options.probe_options.step_size = 1
    
    options.probe_position_options.position_x_px = positions_px[:, 1]
    options.probe_position_options.position_y_px = positions_px[:, 0]
    options.probe_position_options.update_magnitude_limit = 1.0
    options.probe_position_options.optimizable = True
    options.probe_position_options.optimization_plan.start = 1
    options.probe_position_options.optimization_plan.end = 3
    options.probe_position_options.optimization_plan.stride = 2
    options.probe_position_options.optimizer = api.Optimizers.ADAM
    options.probe_position_options.step_size = 1e-1
    
    options.opr_mode_weight_options.initial_weights = np.array([1, 1e-6, 1e-6, 1e-6])
    options.opr_mode_weight_options.optimize_intensity_variation = True
    options.opr_mode_weight_options.optimizable = True
    
    options.reconstructor_options.num_epochs = 4
    options.reconstructor_options.batch_size = 40
    options.reconstructor_options.default_device = api.Devices.GPU
    options.reconstructor_options.displayed_loss_function = api.LossFunctions.MSE_SQRT
    options.reconstructor_options.log_level = logging.DEBUG
    
    with PtychographyTask(options) as task:
        task.run()
        
        recon = task.get_data_to_cpu(name='object', as_numpy=True)[0]
        
        if debug and not generate_gold:
            tutils.plot_complex_image(recon)
        if generate_gold:
            tutils.save_gold_data(name, recon)
        else:
            tutils.run_comparison(name, recon, high_tol=high_tol)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate-gold', action='store_true')
    args = parser.parse_args()

    test_2d_ptycho_opt_plan(None, generate_gold=args.generate_gold, debug=True)
