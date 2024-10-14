import random
import os
import datetime

import torch
import h5py
import numpy as np

from ptychointerim.ptychotorch.utils import rescale_probe, add_additional_opr_probe_modes_to_probe, set_default_complex_dtype, to_tensor


def get_timestamp():
    return datetime.datetime.now().strftime('%Y%m%d%H%M')

def get_ci_data_dir():
    try:
        dir = os.environ['PTYCHO_CI_DATA_DIR']
    except KeyError:
        raise KeyError('PTYCHO_CI_DATA_DIR not set. Please set it to the path to the data folder.')
    return dir


def get_ci_input_data_dir():
    return os.path.join(get_ci_data_dir(), 'data')


def get_ci_gold_data_dir():
    return os.path.join(get_ci_data_dir(), 'gold_data')


def get_default_input_data_file_paths(name):
    dp = os.path.join(get_ci_input_data_dir(), name, 'ptychodus_dp.hdf5')
    para = os.path.join(get_ci_input_data_dir(), name, 'ptychodus_para.hdf5')
    return dp, para


def setup(name, cpu_only=True, gpu_indices=(0,)):
    torch.manual_seed(123)
    random.seed(123)
    np.random.seed(123)
    
    torch.set_default_device('cpu' if cpu_only else 'cuda')
    torch.set_default_dtype(torch.float32)
    set_default_complex_dtype(torch.complex64)
    
    if not cpu_only:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_indices))
    
    name = os.path.join(get_ci_gold_data_dir(), name)
    if not os.path.exists(name):
        os.makedirs(name)
        
        
def load_data_ptychodus(diffraction_pattern_file, parameter_file, subtract_position_mean=False, additional_opr_modes=0):
    patterns = h5py.File(diffraction_pattern_file, 'r')['dp'][...]

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
    if subtract_position_mean:
        positions_px -= positions_px.mean(axis=0)
    
    return patterns, probe, pixel_size_m, positions_px
    
    
def load_tungsten_data(additional_opr_modes=0, pos_type='true'):
    return load_data_ptychodus(
        os.path.join(get_ci_input_data_dir(), '2d_ptycho', 'dp_250.hdf5'), 
        os.path.join(get_ci_input_data_dir(), '2d_ptycho', 'metadata_250_{}Pos.hdf5'.format(pos_type)), 
        additional_opr_modes=additional_opr_modes
    )
    

def save_gold_data(name, data):
    fname = os.path.join(get_ci_gold_data_dir(), name, 'recon.npy')
    np.save(fname, data)
    
    
def load_gold_data(name):
    fname = os.path.join(get_ci_gold_data_dir(), name, 'recon.npy')
    return np.load(fname)


def save_test_data(name, data):
    dirname = os.path.join(name + '_dump_{}'.format(get_timestamp()))
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    fname = os.path.join(dirname, 'recon.npy')
    np.save(fname, data)


def run_comparison(name, test_data, high_tol=False):
    gold_data = load_gold_data(name)
    atol = 1e-3
    rtol = 1e-2 if high_tol else 1e-4
    if not np.allclose(gold_data.shape, test_data.shape):
        print('{} FAILED [SHAPE MISMATCH]'.format(name))
        print('  Gold shape: {}'.format(gold_data.shape))
        print('  Test shape: {}'.format(test_data.shape))
        raise AssertionError
    if not np.allclose(gold_data, test_data, atol=atol, rtol=rtol):
        print('{} FAILED [MISMATCH]'.format(name))
        abs_diff = np.abs(gold_data - test_data)
        loc_max_diff = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)
        loc_max_diff = [a.item() for a in loc_max_diff]
        print('  Mean abs diff: {}'.format(abs_diff.mean()))
        print('  Location of max diff: {}'.format(loc_max_diff))
        print('  Max abs diff: {}'.format(abs_diff[*loc_max_diff].item()))
        print('  Value at max abs diff (test): {}'.format(test_data[*loc_max_diff].item()))
        print('  Value at max abs diff (gold): {}'.format(gold_data[*loc_max_diff].item()))
        raise AssertionError
    print('{} PASSED'.format(name))
    return

def plot_complex_image(img):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(np.abs(img))
    ax[0].set_title('magnitude')
    ax[1].imshow(np.angle(img))
    ax[1].set_title('phase')
    plt.show()
