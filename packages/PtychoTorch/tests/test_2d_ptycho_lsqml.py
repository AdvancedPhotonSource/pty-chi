import argparse
import logging.config
import os
import random
import pytest

import torch
import h5py
import numpy as np

from ptychotorch.data_structures import *
from ptychotorch.io_handles import PtychographyDataset
from ptychotorch.forward_models import Ptychography2DForwardModel
from ptychotorch.utils import (get_suggested_object_size, set_default_complex_dtype, get_default_complex_dtype, 
                            rescale_probe)
from ptychotorch.reconstructors import *


def setup(gold_dir):
    if not os.path.exists(gold_dir):
        os.makedirs(gold_dir)
    
    torch.manual_seed(123)
    random.seed(123)
    torch.use_deterministic_algorithms(True)
    
    torch.set_default_device('cpu')
    torch.set_default_dtype(torch.float32)
    set_default_complex_dtype(torch.complex64)


def load_data(pos_type='nominal'):
    patterns = h5py.File('data/2d_ptycho/dp_250.hdf5', 'r')['dp'][...]
    dataset = PtychographyDataset(patterns)

    f_meta = h5py.File('data/2d_ptycho/metadata_250_{}Pos.hdf5'.format(pos_type), 'r')
    probe = f_meta['probe'][...]
    probe = rescale_probe(probe, patterns)
    
    positions = np.stack([f_meta['probe_position_y_m'][...], f_meta['probe_position_x_m'][...]], axis=1)
    pixel_size_m = 8e-9
    positions_px = positions / pixel_size_m
    
    return dataset, probe, positions_px, pixel_size_m
    

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


def test_2d_ptycho_lsqml(pytestconfig, generate_gold=False, debug=False, high_tol=False):
    if pytestconfig is not None:
        high_tol = pytestconfig.getoption("high_tol")
        
    gold_dir = os.path.join('gold_data', 'test_2d_ptycho_lsqml')
    setup(gold_dir)

    dataset, probe, positions_px, pixel_size_m = load_data(pos_type='true')
    
    object = Object2D(
        data=torch.ones(get_suggested_object_size(positions_px, probe.shape[-2:], extra=100), dtype=get_default_complex_dtype()), 
        pixel_size_m=pixel_size_m,
        optimizable=True,
        optimizer_class=torch.optim.SGD,
        optimizer_params={'lr': 1}
    )

    probe = Probe(
        data=probe,
        optimizable=True,
        optimizer_class=torch.optim.SGD,
        optimizer_params={'lr': 1}
    )
    
    probe_positions = ProbePositions(
        data=positions_px,
        optimizable=False,
    )

    reconstructor = LSQMLReconstructor(
        variable_group=Ptychography2DVariableGroup(object=object, probe=probe, probe_positions=probe_positions),
        dataset=dataset,
        batch_size=96,
        noise_model='gaussian',
        n_epochs=8
    )
    reconstructor.build()
    reconstructor.run()

    recon = reconstructor.variable_group.object.tensor.complex().detach().cpu().numpy()
    compare_results(recon, gold_dir, generate_gold=generate_gold, high_tol=high_tol)
    
    
def test_2d_ptycho_lsqml_poscorr(pytestconfig, generate_gold=False, debug=False, high_tol=False):
    if pytestconfig is not None:
        high_tol = pytestconfig.getoption("high_tol")
        
    gold_dir = os.path.join('gold_data', 'test_2d_ptycho_lsqml_poscorr')
    setup(gold_dir)

    dataset, probe, positions_px, pixel_size_m = load_data(pos_type='nominal')
    
    object = Object2D(
        data=torch.ones(get_suggested_object_size(positions_px, probe.shape[-2:], extra=100), dtype=get_default_complex_dtype()), 
        pixel_size_m=pixel_size_m,
        optimizable=True,
        optimizer_class=torch.optim.SGD,
        optimizer_params={'lr': 1}
    )

    probe = Probe(
        data=probe,
        optimizable=True,
        optimizer_class=torch.optim.SGD,
        optimizer_params={'lr': 1}
    )
    
    probe_positions = ProbePositions(
        data=positions_px,
        optimizable=True,
        optimizer_class=torch.optim.SGD,
        optimizer_params={'lr': 0.1},
        update_magnitude_limit=2.0
    )

    reconstructor = LSQMLReconstructor(
        variable_group=Ptychography2DVariableGroup(object=object, probe=probe, probe_positions=probe_positions),
        dataset=dataset,
        batch_size=96,
        noise_model='gaussian',
        n_epochs=8
    )
    reconstructor.build()
    reconstructor.run()

    recon = reconstructor.variable_group.object.tensor.complex().detach().cpu().numpy()
    
    if debug:
        import matplotlib.pyplot as plt
        pos = reconstructor.variable_group.probe_positions.tensor.detach().cpu().numpy()
        pos_true = load_data(pos_type='true')[2]
        plt.figure()
        plt.plot(pos[:, 1], pos[:, 0], label='corrected')
        plt.plot(pos_true[:, 1], pos_true[:, 0], label='true')
        plt.show()
    
    compare_results(recon, gold_dir, generate_gold=generate_gold, high_tol=high_tol)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate-gold', action='store_true')
    parser.add_argument('--high-tol', action='store_true')
    args = parser.parse_args()

    test_2d_ptycho_lsqml(None, generate_gold=args.generate_gold, debug=True, high_tol=args.high_tol)
    test_2d_ptycho_lsqml_poscorr(None, generate_gold=args.generate_gold, debug=True, high_tol=args.high_tol)
