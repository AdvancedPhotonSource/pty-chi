import argparse
import logging

import torch
import numpy as np

import ptychi.api as api
from ptychi.api.task import PtychographyTask
from ptychi.api import LSQMLOptions, AutodiffPtychographyOptions
import ptychi.ptychotorch.utils as utils

import test_utils as tutils


def test_2d_ptycho_probe_power_constraint_lsqml(pytestconfig, generate_gold=False, debug=False, high_tol=False):
    if pytestconfig is not None:
        high_tol = pytestconfig.getoption("high_tol")
        
    name = 'test_2d_ptycho_probe_power_constraint_lsqml'
    
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
    options.probe_options.optimizer = 'sgd'
    options.probe_options.step_size = 1
    options.probe_options.probe_power = data[0].sum()
    options.probe_options.probe_power_constraint_stride = 1
    
    options.probe_position_options.position_x_px = positions_px[:, 1]
    options.probe_position_options.position_y_px = positions_px[:, 0]
    options.probe_position_options.pixel_size_m = pixel_size_m
    options.probe_position_options.update_magnitude_limit = 1.0
    options.probe_position_options.optimizable = True
    options.probe_position_options.optimizer = api.Optimizers.ADAM
    options.probe_position_options.step_size = 1e-1
    
    options.opr_mode_weight_options.initial_weights = np.array([1, 0.1, 0.1, 0.1])
    options.opr_mode_weight_options.optimize_intensity_variation = True
    options.opr_mode_weight_options.optimizable = True
    
    options.reconstructor_options.num_epochs = 4
    options.reconstructor_options.batch_size = 40
    options.reconstructor_options.default_device = api.Devices.GPU
    options.reconstructor_options.displayed_loss_function = api.LossFunctions.MSE_SQRT
    
    with PtychographyTask(options) as task:
        task.run()
        # This should be equivalent to:
        # for _ in range(64):
        #     task.run(1)
        
        recon = task.get_data_to_cpu(name='object', as_numpy=True)[0]
        
        if debug and not generate_gold:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(abs(recon))       
            ax[1].imshow(np.angle(recon))
            plt.show()    
    
        if generate_gold:
            tutils.save_gold_data(name, recon)
        else:
            tutils.run_comparison(name, recon, high_tol=high_tol)
    
    
def test_2d_ptycho_probe_power_constraint_ad(pytestconfig, generate_gold=False, debug=False, high_tol=False):
    if pytestconfig is not None:
        high_tol = pytestconfig.getoption("high_tol")
        
    name = 'test_2d_ptycho_probe_power_constraint_ad'
    
    tutils.setup(name, cpu_only=False, gpu_indices=[0])

    data, probe, pixel_size_m, positions_px = tutils.load_tungsten_data(pos_type='true', additional_opr_modes=3)
    object_init = torch.ones(
        [1, *utils.get_suggested_object_size(positions_px, probe.shape[-2:], extra=100)], 
        dtype=utils.get_default_complex_dtype()
    )
    
    options = AutodiffPtychographyOptions()
    options.data_options.data = data
    
    options.object_options.initial_guess = object_init
    options.object_options.pixel_size_m = pixel_size_m
    options.object_options.optimizable = True
    options.object_options.optimizer = api.Optimizers.SGD
    options.object_options.step_size = 1e-1
    
    options.probe_options.initial_guess = probe
    options.probe_options.optimizable = True
    options.probe_options.optimizer = api.Optimizers.SGD
    options.probe_options.step_size = 1e-1
    options.probe_options.probe_power = data[0].sum()
    options.probe_options.probe_power_constraint_stride = 1
    
    options.probe_position_options.position_x_px = positions_px[:, 1]
    options.probe_position_options.position_y_px = positions_px[:, 0]
    options.probe_position_options.pixel_size_m = pixel_size_m
    options.probe_position_options.update_magnitude_limit = 1.0
    options.probe_position_options.optimizable = True
    options.probe_position_options.optimizer = api.Optimizers.ADAM
    options.probe_position_options.step_size = 1e-1
    
    options.opr_mode_weight_options.initial_weights = np.array([1, 0.1, 0.1, 0.1])
    options.opr_mode_weight_options.optimize_intensity_variation = True
    options.opr_mode_weight_options.optimizable = True
    options.opr_mode_weight_options.optimizer = api.Optimizers.ADAM
    options.opr_mode_weight_options.step_size = 1e-2
    
    options.reconstructor_options.num_epochs = 4
    options.reconstructor_options.batch_size = 40
    options.reconstructor_options.default_device = api.Devices.GPU
    options.reconstructor_options.displayed_loss_function = api.LossFunctions.MSE_SQRT
    
    with PtychographyTask(options) as task:
        task.run()
        # This should be equivalent to:
        # for _ in range(64):
        #     task.run(1)
        
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
    parser.add_argument('--high-tol', action='store_true')
    args = parser.parse_args()

    test_2d_ptycho_probe_power_constraint_lsqml(None, generate_gold=args.generate_gold, debug=True, high_tol=args.high_tol)
    test_2d_ptycho_probe_power_constraint_ad(None, generate_gold=args.generate_gold, debug=True, high_tol=args.high_tol)
    