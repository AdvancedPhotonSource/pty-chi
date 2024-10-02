import random
import os
from pathlib import Path

import torch
import h5py
import numpy as np

from ptychointerim.ptychotorch.io_handles import PtychographyDataset
from ptychointerim.forward_models import Ptychography2DForwardModel
from ptychointerim.ptychotorch.utils import rescale_probe, add_additional_opr_probe_modes_to_probe, set_default_complex_dtype, to_tensor


def setup(gold_dir, cpu_only=True):
    torch.manual_seed(123)
    random.seed(123)
    np.random.seed(123)
    
    torch.set_default_device('cpu' if cpu_only else 'cuda')
    torch.set_default_dtype(torch.float32)
    set_default_complex_dtype(torch.complex64)
    
    if not os.path.exists(gold_dir):
        os.makedirs(gold_dir)
        
        
def load_data_ptychodus(diffraction_pattern_file, parameter_file, additional_opr_modes=0):
    patterns = h5py.File(diffraction_pattern_file, 'r')['dp'][...]
    dataset = PtychographyDataset(patterns)

    f_meta = h5py.File(parameter_file, 'r')
    probe = f_meta['probe'][...]
    probe = rescale_probe(probe, patterns)
    probe = probe[None, :, :, :]
    probe = to_tensor(probe)
    if additional_opr_modes > 0:
        probe = add_additional_opr_probe_modes_to_probe(probe, n_opr_modes_to_add=additional_opr_modes)
    
    positions = np.stack([f_meta['probe_position_y_m'][...], f_meta['probe_position_x_m'][...]], axis=1)
    pixel_size_m = f_meta['object'].attrs['pixel_height_m']
    positions_px = positions / pixel_size_m
    
    return dataset, probe, pixel_size_m, positions_px
    
    
def load_tungsten_data(additional_opr_modes=0, pos_type='true'):
    return load_data_ptychodus(
        'data/2d_ptycho/dp_250.hdf5', 
        'data/2d_ptycho/metadata_250_{}Pos.hdf5'.format(pos_type), 
        additional_opr_modes=additional_opr_modes
    )