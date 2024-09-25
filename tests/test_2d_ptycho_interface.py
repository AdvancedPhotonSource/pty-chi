import argparse
import os

import torch
import numpy as np

from ptychointerim.interface import PtychographyJob
import ptychointerim.configs as configs
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


def test_2d_ptycho_interface_lsqml(pytestconfig, generate_gold=False, debug=False, high_tol=False):
    if pytestconfig is not None:
        high_tol = pytestconfig.getoption("high_tol")
        
    gold_dir = os.path.join('gold_data', 'test_2d_ptycho_interface_lsqml')
    
    tutils.setup(gold_dir, cpu_only=True)

    dataset, probe, pixel_size_m, positions_px = tutils.load_tungsten_data(pos_type='true', additional_opr_modes=3)
    data = dataset.patterns
    object_init = torch.ones(
        utils.get_suggested_object_size(positions_px, probe.shape[-2:], extra=100), 
        dtype=utils.get_default_complex_dtype()
    )
    positions_m = positions_px * pixel_size_m
    
    data_config = configs.PtychographyDataConfig(
        data=data
    )
    
    object_config = configs.ObjectConfig(
        initial_guess=object_init,
        pixel_size_m=pixel_size_m,
        optimizable=True,
        optimizer='sgd',
        step_size=1
    )
    
    probe_config = configs.ProbeConfig(
        initial_guess=probe,
        optimizable=True,
        optimizer='sgd',
        step_size=1
    )
    
    position_config = configs.ProbePositionConfig(
        position_x_m=positions_m[:, 1],
        position_y_m=positions_m[:, 0],
        pixel_size_m=pixel_size_m,
        update_magnitude_limit=1.0,
        optimizable=True,
        optimizer='adam',
        step_size=1e-1
    )
    
    opr_mode_weight_config = configs.OPRModeWeightsConfig(
        initial_eigenmode_weights=0.1,
        optimize_intensity_variation=True,
        optimizable=True
    )
    
    reconstructor_config = configs.LSQMLReconstructorConfig(
        num_epochs=64,
        batch_size=40,
        default_device='gpu',
        gpu_indices=[0],
        metric_function='mse_sqrt',
        log_level='info'
    )
    
    job = PtychographyJob(
        data_config=data_config,
        object_config=object_config,
        probe_config=probe_config,
        position_config=position_config,
        opr_mode_weight_config=opr_mode_weight_config,
        reconstructor_config=reconstructor_config
    )
    job.build()
    job.run()
    # This should be equivalent to:
    # for _ in range(64):
    #     job.iterate(1)
    
    recon = job.get_data_to_cpu(name='object')
    
    if debug:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(abs(recon))       
        ax[1].imshow(np.angle(recon))
        plt.show()    
    
    compare_results(recon, gold_dir, generate_gold=generate_gold, high_tol=high_tol)
    
    
def test_2d_ptycho_interface_ad(pytestconfig, generate_gold=False, debug=False, high_tol=False):
    if pytestconfig is not None:
        high_tol = pytestconfig.getoption("high_tol")
        
    gold_dir = os.path.join('gold_data', 'test_2d_ptycho_interface_ad')
    
    tutils.setup(gold_dir, cpu_only=True)

    dataset, probe, pixel_size_m, positions_px = tutils.load_tungsten_data(pos_type='true', additional_opr_modes=3)
    data = dataset.patterns
    object_init = torch.ones(
        utils.get_suggested_object_size(positions_px, probe.shape[-2:], extra=100), 
        dtype=utils.get_default_complex_dtype()
    )
    positions_m = positions_px * pixel_size_m
    
    data_config = configs.PtychographyDataConfig(
        data=data
    )
    
    object_config = configs.ObjectConfig(
        initial_guess=object_init,
        pixel_size_m=pixel_size_m,
        optimizable=True,
        optimizer='sgd',
        step_size=1e-1
    )
    
    probe_config = configs.ProbeConfig(
        initial_guess=probe,
        optimizable=True,
        optimizer='sgd',
        step_size=1e-1
    )
    
    position_config = configs.ProbePositionConfig(
        position_x_m=positions_m[:, 1],
        position_y_m=positions_m[:, 0],
        pixel_size_m=pixel_size_m,
        update_magnitude_limit=1.0,
        optimizable=True,
        optimizer='adam',
        step_size=1e-1
    )
    
    opr_mode_weight_config = configs.OPRModeWeightsConfig(
        initial_eigenmode_weights=0.1,
        optimize_intensity_variation=True,
        optimizable=True,
        optimizer='adam',
        step_size=1e-2
    )
    
    reconstructor_config = configs.AutodiffPtychographyReconstructorConfig(
        num_epochs=64,
        batch_size=96,
        default_device='gpu',
        gpu_indices=[0],
        metric_function='mse_sqrt',
        loss_function='mse_sqrt',
        log_level='info'
    )
    
    job = PtychographyJob(
        data_config=data_config,
        object_config=object_config,
        probe_config=probe_config,
        position_config=position_config,
        opr_mode_weight_config=opr_mode_weight_config,
        reconstructor_config=reconstructor_config
    )
    job.build()
    job.run()
    # This should be equivalent to:
    # for _ in range(64):
    #     job.iterate(1)
    
    recon = job.get_data_to_cpu(name='object')
    
    if debug:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(abs(recon))       
        ax[1].imshow(np.angle(recon))
        plt.show()    
    
    compare_results(recon, gold_dir, generate_gold=generate_gold, high_tol=high_tol)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate-gold', action='store_true')
    args = parser.parse_args()

    test_2d_ptycho_interface_lsqml(None, generate_gold=args.generate_gold, debug=True)
    test_2d_ptycho_interface_ad(None, generate_gold=args.generate_gold, debug=True)
