import argparse

import torch
import numpy as np

import ptychi.api as api
from ptychi.api.task import PtychographyTask
from ptychi.utils import get_suggested_object_size, get_default_complex_dtype, generate_initial_opr_mode_weights

import test_utils as tutils


def test_2d_ptycho_lsqml_compact(pytestconfig, generate_gold=False, debug=False, high_tol=False):
    if pytestconfig is not None:
        high_tol = pytestconfig.getoption("high_tol")
        
    name = 'test_2d_ptycho_lsqml_compact'
    
    tutils.setup(name, cpu_only=False, gpu_indices=[0])

    data, probe, pixel_size_m, positions_px = tutils.load_tungsten_data(pos_type='true')
    
    options = api.LSQMLOptions()
    options.data_options.data = data
    
    options.object_options.initial_guess = torch.ones([1, *get_suggested_object_size(positions_px, probe.shape[-2:], extra=100)], dtype=get_default_complex_dtype())
    options.object_options.pixel_size_m = pixel_size_m
    options.object_options.optimizable = True
    options.object_options.optimizer = api.Optimizers.SGD
    options.object_options.step_size = 1
    
    options.probe_options.initial_guess = probe
    options.probe_options.optimizable = True
    options.probe_options.optimizer = api.Optimizers.SGD
    options.probe_options.step_size = 1

    options.probe_position_options.position_x_px = positions_px[:, 1]
    options.probe_position_options.position_y_px = positions_px[:, 0]
    options.probe_position_options.step_size = 0.1
    options.probe_position_options.optimizable = True
    
    options.reconstructor_options.batch_size = 96
    options.reconstructor_options.noise_model = api.NoiseModels.GAUSSIAN
    options.reconstructor_options.num_epochs = 8
    options.reconstructor_options.batching_mode = api.BatchingModes.COMPACT
    options.reconstructor_options.compact_mode_update_clustering = True
    
    task = PtychographyTask(options)
    task.run()
    
    recon = task.get_data_to_cpu('object', as_numpy=True)[0]

    if debug:
        tutils.plot_complex_image(recon)
    if generate_gold:
        tutils.save_gold_data(name, recon)
    else:
        tutils.run_comparison(name, recon, high_tol=high_tol)
        

def test_2d_ptycho_lsqml_compact_multislice(pytestconfig, generate_gold=False, debug=False, high_tol=False):
    if pytestconfig is not None:
        high_tol = pytestconfig.getoption("high_tol")
        
    name = 'test_2d_ptycho_lsqml_compact_multislice'
    
    tutils.setup(name, cpu_only=False, gpu_indices=[0])

    data, probe, pixel_size_m, positions_px = tutils.load_tungsten_data(pos_type='true')
    
    options = api.LSQMLOptions()
    options.data_options.data = data
    
    options.object_options.initial_guess = torch.ones([2, *get_suggested_object_size(positions_px, probe.shape[-2:], extra=100)], dtype=get_default_complex_dtype())
    options.object_options.pixel_size_m = pixel_size_m
    options.object_options.optimizable = True
    options.object_options.optimizer = api.Optimizers.SGD
    options.object_options.step_size = 0.5
    options.object_options.slice_spacings_m = np.array([2e-7])
    
    options.probe_options.initial_guess = probe
    options.probe_options.optimizable = True
    options.probe_options.optimizer = api.Optimizers.SGD
    options.probe_options.step_size = 0.5

    options.probe_position_options.position_x_px = positions_px[:, 1]
    options.probe_position_options.position_y_px = positions_px[:, 0]
    options.probe_position_options.step_size = 0.1
    options.probe_position_options.optimizable = True
    
    options.reconstructor_options.batch_size = 96
    options.reconstructor_options.noise_model = api.NoiseModels.GAUSSIAN
    options.reconstructor_options.num_epochs = 8
    options.reconstructor_options.batching_mode = api.BatchingModes.COMPACT
    options.reconstructor_options.compact_mode_update_clustering = True
    
    task = PtychographyTask(options)
    task.run()
    
    recon = task.get_data_to_cpu('object', as_numpy=True)

    if debug:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(np.angle(recon[0]))
        ax[1].imshow(np.angle(recon[1]))
        plt.show()
    if generate_gold:
        tutils.save_gold_data(name, recon)
    else:
        tutils.run_comparison(name, recon, high_tol=high_tol)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate-gold', action='store_true')
    parser.add_argument('--high-tol', action='store_true')
    args = parser.parse_args()

    test_2d_ptycho_lsqml_compact(None, generate_gold=args.generate_gold, debug=True, high_tol=args.high_tol)
    test_2d_ptycho_lsqml_compact_multislice(None, generate_gold=args.generate_gold, debug=True, high_tol=args.high_tol)
