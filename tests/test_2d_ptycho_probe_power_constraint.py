import argparse
import os
import logging

import torch
import numpy as np

import ptychointerim.api as api
from ptychointerim.api.task import PtychographyTask
from ptychointerim.api import LSQMLOptions, AutodiffPtychographyOptions
import ptychointerim.ptychotorch.utils as utils

import test_utils as tutils


def compare_results(recon, gold_dir, generate_gold=False, high_tol=False):
    if generate_gold:
        np.save(os.path.join(gold_dir, 'recon.npy'), recon)
    else:
        recon_gold = np.load(os.path.join(gold_dir, 'recon.npy'))
        recon = recon[300:400, 300:400]
        recon_gold = recon_gold[300:400, 300:400]
        print(recon)
        print(recon_gold)
        diff = np.abs(recon - recon_gold)
        amax = np.unravel_index(np.argmax(diff), diff.shape)
        print('value of max diff in recon: ', recon[amax[0], amax[1]])
        print('value of max diff in recon_gold: ', recon_gold[amax[0], amax[1]])
        if not high_tol:
            assert np.allclose(recon, recon_gold)
        else:
            assert np.allclose(recon.real, recon_gold.real, rtol=1e-2, atol=1e-1)
            assert np.allclose(recon.imag, recon_gold.imag, rtol=1e-2, atol=1e-1)


def test_2d_ptycho_probe_power_constraint_lsqml(pytestconfig, generate_gold=False, debug=False, high_tol=False):
    if pytestconfig is not None:
        high_tol = pytestconfig.getoption("high_tol")
        
    gold_dir = os.path.join('gold_data', 'test_2d_ptycho_probe_power_constraint_lsqml')
    
    tutils.setup(gold_dir, cpu_only=True)

    dataset, probe, pixel_size_m, positions_px = tutils.load_tungsten_data(pos_type='true', additional_opr_modes=3)
    data = dataset.patterns
    object_init = torch.ones(
        utils.get_suggested_object_size(positions_px, probe.shape[-2:], extra=100), 
        dtype=utils.get_default_complex_dtype()
    )
    positions_m = positions_px * pixel_size_m
    
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
    options.probe_options.probe_power = dataset[0][-1].sum()
    options.probe_options.probe_power_constraint_stride = 1
    
    options.probe_position_options.position_x_m = positions_m[:, 1]
    options.probe_position_options.position_y_m = positions_m[:, 0]
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
    options.reconstructor_options.default_device = api.Devices.CPU
    # options.reconstructor_options.gpu_indices = [0]
    options.reconstructor_options.metric_function = api.LossFunctions.MSE_SQRT
    options.reconstructor_options.log_level = logging.INFO
    
    with PtychographyTask(options) as task:
        task.run()
        # This should be equivalent to:
        # for _ in range(64):
        #     task.iterate(1)
        
        recon = task.get_data_to_cpu(name='object')
        
        if debug and not generate_gold:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(abs(recon))       
            ax[1].imshow(np.angle(recon))
            plt.show()    
    
    compare_results(recon, gold_dir, generate_gold=generate_gold, high_tol=high_tol)
    
    
def test_2d_ptycho_probe_power_constraint_ad(pytestconfig, generate_gold=False, debug=False, high_tol=False):
    if pytestconfig is not None:
        high_tol = pytestconfig.getoption("high_tol")
        
    gold_dir = os.path.join('gold_data', 'test_2d_ptycho_probe_power_constraint_ad')
    
    tutils.setup(gold_dir, cpu_only=True)

    dataset, probe, pixel_size_m, positions_px = tutils.load_tungsten_data(pos_type='true', additional_opr_modes=3)
    data = dataset.patterns
    object_init = torch.ones(
        utils.get_suggested_object_size(positions_px, probe.shape[-2:], extra=100), 
        dtype=utils.get_default_complex_dtype()
    )
    positions_m = positions_px * pixel_size_m
    
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
    options.probe_options.probe_power = dataset[0][-1].sum()
    options.probe_options.probe_power_constraint_stride = 1
    
    options.probe_position_options.position_x_m = positions_m[:, 1]
    options.probe_position_options.position_y_m = positions_m[:, 0]
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
    options.reconstructor_options.default_device = api.Devices.CPU
    # options.reconstructor_options.gpu_indices = [0]
    options.reconstructor_options.metric_function = api.LossFunctions.MSE_SQRT
    options.reconstructor_options.log_level = logging.INFO
    
    with PtychographyTask(options) as task:
        task.run()
        # This should be equivalent to:
        # for _ in range(64):
        #     task.iterate(1)
        
        recon = task.get_data_to_cpu(name='object')
        
        if debug and not generate_gold:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(abs(recon))       
            ax[1].imshow(np.angle(recon))
            plt.show()    
    
    compare_results(recon, gold_dir, generate_gold=generate_gold, high_tol=high_tol)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate-gold', action='store_true')
    parser.add_argument('--high-tol', action='store_true')
    args = parser.parse_args()

    test_2d_ptycho_probe_power_constraint_lsqml(None, generate_gold=args.generate_gold, debug=True, high_tol=args.high_tol)
    test_2d_ptycho_probe_power_constraint_ad(None, generate_gold=args.generate_gold, debug=True, high_tol=args.high_tol)
    