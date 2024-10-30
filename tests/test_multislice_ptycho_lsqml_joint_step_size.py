# pip install -U --no-deps .
# pip install -e .

import argparse

import torch
import torchvision.transforms.functional as F

import numpy as np
from PIL import Image

import ptychi.maths
import ptychi.api as api
from ptychi.api.task import PtychographyTask
from ptychi.ptychotorch.utils import get_suggested_object_size, get_default_complex_dtype
from ptychi.ptychotorch.utils import rescale_probe

import test_utils as tutils
import os
import tifffile

def test_multislice_ptycho_lsqml_joint_step_size(generate_gold=False, debug=False):
    name = 'test_multislice_ptycho_lsqml_joint_step_size'
    
    tutils.setup(name, cpu_only=False, gpu_indices=[0])
    
    data, probe0, pixel_size_m, positions_px = tutils.load_data_ptychodus(
        *tutils.get_default_input_data_file_paths('multislice_ptycho_AuNi'),
        subtract_position_mean=True
    )

    #'''

    ########################################################
    # Add more shared probe modes to the probe tensor array:

    # Define the affine transformation parameters
    angle = 25              # Rotation angle in degrees
    translate = [0, 0]      # Translation in x and y directions
    scale = 1.3             # Scaling factor
    shear = [5, 10]         # Shear angle in degrees

    # Apply the affine transformation
    transformed_image_re = F.affine( probe0.real, angle, translate, scale, shear )
    transformed_image_im = F.affine( probe0.imag, angle, translate, scale, shear )
    transformed_image_1 =  transformed_image_re + 1j *  transformed_image_im

    # Define the affine transformation parameters
    angle = -10              # Rotation angle in degrees
    translate = [-4, 6]      # Translation in x and y directions
    scale = 0.9             # Scaling factor
    shear = [-4, +4]         # Shear angle in degrees

    # Apply the affine transformation
    transformed_image_re = F.affine( probe0.real, angle, translate, scale, shear )
    transformed_image_im = F.affine( probe0.imag, angle, translate, scale, shear )
    transformed_image_2 =  transformed_image_re + 1j *  transformed_image_im

    N_shared_modes = 3
    N_variable_modes = probe0.shape[0]

    probe = torch.zeros( N_variable_modes, N_shared_modes, *probe0.shape[-2:], dtype = torch.complex64)
    probe[ :, 0, ... ] = probe0[ :, 0, ... ]
    probe[ :, 1, ... ] = transformed_image_1[ :, 0, ... ]
    probe[ :, 2, ... ] = transformed_image_2[ :, 0, ... ]

    probe = ptychi.maths.orthogonalize_gs( probe, group_dim = 1, dim = (-1,-2) )
    #probe = ptychi.maths.orthogonalize_svd( probe, group_dim = 1, dim = (-1,-2) )

    new_probe_occ = np.exp( -0.5 * np.arange( 0, probe.shape[-3], 1, dtype=np.float32 ))
    new_probe_occ = new_probe_occ / np.linalg.norm( new_probe_occ, ord = 1 )
    new_probe_occ = torch.from_numpy( new_probe_occ )
    new_probe_occ = new_probe_occ.cuda()

    total_photons = torch.sum( torch.square( torch.abs( probe )))
    #total_photons = 9.0e6

    probeF = 1.2 * rescale_probe(probe[0, ...], data)
    probeF = probeF[ torch.newaxis, ... ]
    new_probe_photons = torch.sum( torch.square( torch.abs( probeF )), dim = (-1, -2) )

    probe = probeF * torch.sqrt( new_probe_occ * total_photons / new_probe_photons )[ ..., None, None ]

    total_photons2 = torch.sum( torch.abs( probe ) ** 2 )
    new_probe_photons2 = torch.sum( torch.abs( probe ) ** 2, (-1, -2) )
    probe_occ = new_probe_photons2 / total_photons2
    print(f"probe occupancy = {probe_occ}" )

    wavelength_m = 1.03e-10
    
    options = api.LSQMLOptions()
    
    options.data_options.data = data
    options.data_options.wavelength_m = wavelength_m
    
    options.object_options.initial_guess = torch.ones([2, *get_suggested_object_size(positions_px, probe.shape[-2:], extra=50)], dtype=get_default_complex_dtype())
    options.object_options.pixel_size_m = pixel_size_m
    options.object_options.slice_spacings_m = np.array([10e-6])
    options.object_options.optimizable = True
    options.object_options.optimizer = api.Optimizers.SGD
    options.object_options.step_size = 1
    options.object_options.multislice_regularization_weight = 0e-2
    options.object_options.multislice_regularization_stride = 5
    options.object_options.solve_obj_prb_step_size_jointly_for_first_slice_in_multislice = True

    options.probe_options.initial_guess = probe
    options.probe_options.optimizable = True
    options.probe_options.optimizer = api.Optimizers.SGD
    options.probe_options.orthogonalize_incoherent_modes = True
    options.probe_options.step_size = 1
    
    options.probe_position_options.position_x_px = positions_px[:, 1]
    options.probe_position_options.position_y_px = positions_px[:, 0]

    options.probe_options.orthogonalize_incoherent_modes_method = api.OrthogonalizationMethods.SVD
    # options.probe_options.orthogonalize_incoherent_modes_method = api.OrthogonalizationMethods.GS

    options.probe_position_options.optimizable = False
    options.probe_position_options.optimizer = api.Optimizers.SGD
    options.probe_position_options.step_size = 1e-1
    options.probe_position_options.update_magnitude_limit = 1.0
    
    options.reconstructor_options.metric_function = api.LossFunctions.MSE_SQRT
    options.reconstructor_options.batch_size = 101
    options.reconstructor_options.num_epochs = 32
    options.reconstructor_options.default_device = api.Devices.GPU
    options.reconstructor_options.random_seed = 123
    #options.reconstructor_options.noise_model = api.NoiseModels.POISSON

    task = PtychographyTask(options)
    task.run( 500 )

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

    test_multislice_ptycho_lsqml_joint_step_size(generate_gold=args.generate_gold, debug=True)
