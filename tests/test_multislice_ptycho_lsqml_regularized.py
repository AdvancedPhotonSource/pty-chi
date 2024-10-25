import argparse

import torch
import numpy as np

import ptychi.api as api
from ptychi.api.task import PtychographyTask
from ptychi.ptychotorch.utils import get_suggested_object_size, get_default_complex_dtype

import test_utils as tutils


def test_multislice_ptycho_lsqml_regularized(generate_gold=False, debug=False):
    name = 'test_multislice_ptycho_lsqml_regularized'
    
    tutils.setup(name, cpu_only=False, gpu_indices=[0])
    
    data, probe, pixel_size_m, positions_px = tutils.load_data_ptychodus(
        *tutils.get_default_input_data_file_paths('multislice_ptycho_AuNi'),
        subtract_position_mean=True
    )
    wavelength_m = 1.03e-10
    
    options = api.LSQMLOptions()
    
    options.data_options.data = data
    options.data_options.wavelength_m = wavelength_m
    
    options.object_options.initial_guess = torch.ones([2, *get_suggested_object_size(positions_px, probe.shape[-2:], extra=50)], dtype=get_default_complex_dtype())
    options.object_options.pixel_size_m = pixel_size_m
    options.object_options.slice_spacings_m = np.array([1e-5])
    options.object_options.optimizable = True
    options.object_options.optimizer = api.Optimizers.SGD
    options.object_options.step_size = 1
    options.object_options.multislice_regularization_weight = 0.1
    options.object_options.multislice_regularization_unwrap_phase = True
    options.object_options.multislice_regularization_unwrap_image_grad_method = api.enums.ImageGradientMethods.FOURIER_SHIFT
    
    options.probe_options.initial_guess = probe
    options.probe_options.optimizable = True
    options.probe_options.optimizer = api.Optimizers.SGD
    options.probe_options.step_size = 1
    
    options.probe_position_options.position_x_m = positions_px[:, 1]
    options.probe_position_options.position_y_m = positions_px[:, 0]
    options.probe_position_options.optimizable = False
    options.probe_position_options.optimizer = api.Optimizers.SGD
    options.probe_position_options.step_size = 1e-1
    options.probe_position_options.update_magnitude_limit = 1.0
    
    options.reconstructor_options.metric_function = api.LossFunctions.MSE_SQRT
    options.reconstructor_options.batch_size = 101
    options.reconstructor_options.num_epochs = 32
    options.reconstructor_options.default_device = api.Devices.GPU
    options.reconstructor_options.gpu_indices = (0,)
    options.reconstructor_options.random_seed = 123
    
    task = PtychographyTask(options)
    task.run()

    recon = task.get_data_to_cpu('object', as_numpy=True)
    
    if debug and not generate_gold:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(np.angle(recon[0]))
        ax[1].imshow(np.angle(recon[1]))
        plt.show()
    if generate_gold:
        tutils.save_gold_data(name, recon)
    else:
        tutils.run_comparison(name, recon)
        
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate-gold', action='store_true')
    args = parser.parse_args()

    test_multislice_ptycho_lsqml_regularized(generate_gold=args.generate_gold, debug=True)
