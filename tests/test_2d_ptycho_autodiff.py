import argparse
import os
import random

import torch
import h5py
import numpy as np

from ptychointerim.ptychotorch.data_structures import *
from ptychointerim.ptychotorch.io_handles import PtychographyDataset
from ptychointerim.ptychotorch.forward_models import Ptychography2DForwardModel
from ptychointerim.ptychotorch.utils import (get_suggested_object_size, set_default_complex_dtype, get_default_complex_dtype, 
                                             rescale_probe, add_additional_opr_probe_modes_to_probe, generate_initial_opr_mode_weights)
from ptychointerim.ptychotorch.reconstructors import *
from ptychointerim.ptychotorch.metrics import MSELossOfSqrt


def test_2d_ptycho_autodiff(generate_gold=False, debug=False):
    gold_dir = os.path.join('gold_data', 'test_2d_ptycho_autodiff')
    
    torch.manual_seed(123)
    random.seed(123)
    
    torch.set_default_device('cpu')
    torch.set_default_dtype(torch.float32)
    set_default_complex_dtype(torch.complex64)
    
    patterns = h5py.File('data/2d_ptycho/dp_250.hdf5', 'r')['dp'][...]
    dataset = PtychographyDataset(patterns)

    f_meta = h5py.File('data/2d_ptycho/metadata_250_truePos.hdf5', 'r')
    probe = f_meta['probe'][...]
    probe = rescale_probe(probe, patterns)
    probe = probe[None, :, :, :]
    
    positions = np.stack([f_meta['probe_position_y_m'][...], f_meta['probe_position_x_m'][...]], axis=1)
    pixel_size_m = 8e-9
    positions_px = positions / pixel_size_m
    
    object = Object2D(
        data=torch.ones(get_suggested_object_size(positions_px, probe.shape[-2:], extra=100), dtype=get_default_complex_dtype()), 
        pixel_size_m=pixel_size_m,
        optimizable=True,
        optimizer_class=torch.optim.SGD,
        optimizer_params={'lr': 1e-1}
    )

    probe = Probe(
        data=probe,
        optimizable=True,
        optimizer_class=torch.optim.SGD,
        optimizer_params={'lr': 1e-1}
    )
    
    probe_positions = ProbePositions(
        data=positions_px,
        optimizable=False,
        optimizer_class=torch.optim.Adam,
        optimizer_params={'lr': 1e-3}
    )

    reconstructor = AutodiffPtychographyReconstructor(
        variable_group=Ptychography2DVariableGroup(object=object, probe=probe, probe_positions=probe_positions),
        dataset=dataset,
        forward_model_class=Ptychography2DForwardModel,
        batch_size=96,
        loss_function=MSELossOfSqrt(),
        n_epochs=32
    )
    reconstructor.build()
    reconstructor.run()

    recon = reconstructor.variable_group.object.tensor.complex().detach().cpu().numpy()
    
    if generate_gold:
        np.save(os.path.join(gold_dir, 'recon.npy'), recon)
    else:
        recon_gold = np.load(os.path.join(gold_dir, 'recon.npy'))
        assert np.allclose(recon, recon_gold)
        
        
def test_2d_ptycho_autodiff_opr(generate_gold=False, debug=False):
    gold_dir = os.path.join('gold_data', 'test_2d_ptycho_autodiff_opr')
    
    torch.manual_seed(123)
    random.seed(123)
    
    torch.set_default_device('cpu')
    torch.set_default_dtype(torch.float32)
    set_default_complex_dtype(torch.complex64)
    
    patterns = h5py.File('data/2d_ptycho/dp_250.hdf5', 'r')['dp'][...]
    dataset = PtychographyDataset(patterns)

    f_meta = h5py.File('data/2d_ptycho/metadata_250_truePos.hdf5', 'r')
    probe = f_meta['probe'][...]
    probe = rescale_probe(probe, patterns)
    probe = probe[None, :, :, :]
    probe = add_additional_opr_probe_modes_to_probe(to_tensor(probe), n_opr_modes_to_add=3)
    
    positions = np.stack([f_meta['probe_position_y_m'][...], f_meta['probe_position_x_m'][...]], axis=1)
    pixel_size_m = 8e-9
    positions_px = positions / pixel_size_m
    
    object = Object2D(
        data=torch.ones(get_suggested_object_size(positions_px, probe.shape[-2:], extra=100), dtype=get_default_complex_dtype()), 
        pixel_size_m=pixel_size_m,
        optimizable=True,
        optimizer_class=torch.optim.SGD,
        optimizer_params={'lr': 1e-1}
    )

    probe = Probe(
        data=probe,
        optimizable=True,
        optimizer_class=torch.optim.SGD,
        optimizer_params={'lr': 1e-1}
    )
    
    probe_positions = ProbePositions(
        data=positions_px,
        optimizable=False,
        optimizer_class=torch.optim.Adam,
        optimizer_params={'lr': 1e-3}
    )
    
    opr_mode_weights = OPRModeWeights(
        data=generate_initial_opr_mode_weights(len(positions), probe.shape[0]),
        optimizable=True,
        optimizer_class=torch.optim.Adam,
        optimizer_params={'lr': 1e-2}
    )

    reconstructor = AutodiffPtychographyReconstructor(
        variable_group=Ptychography2DVariableGroup(object=object, probe=probe, probe_positions=probe_positions, opr_mode_weights=opr_mode_weights),
        dataset=dataset,
        forward_model_class=Ptychography2DForwardModel,
        batch_size=96,
        loss_function=MSELossOfSqrt(),
        n_epochs=32
    )
    reconstructor.build()
    reconstructor.run()

    recon = reconstructor.variable_group.object.tensor.complex().detach().cpu().numpy()
    
    if generate_gold:
        np.save(os.path.join(gold_dir, 'recon.npy'), recon)
    else:
        recon_gold = np.load(os.path.join(gold_dir, 'recon.npy'))
        assert np.allclose(recon, recon_gold, atol=1e-3, rtol=1e-3)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate-gold', action='store_true')
    args = parser.parse_args()

    test_2d_ptycho_autodiff(generate_gold=args.generate_gold, debug=True)
    test_2d_ptycho_autodiff_opr(generate_gold=args.generate_gold, debug=True)
