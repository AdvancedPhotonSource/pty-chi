import argparse
import os

import torch
import numpy as np

from ptychointerim.ptychotorch.data_structures import *
from ptychointerim.ptychotorch.io_handles import PtychographyDataset
from ptychointerim.forward_models import Ptychography2DForwardModel
from ptychointerim.ptychotorch.utils import (get_suggested_object_size, set_default_complex_dtype, get_default_complex_dtype, 
                                             rescale_probe, add_additional_opr_probe_modes_to_probe, generate_initial_opr_mode_weights)
from ptychointerim.ptychotorch.reconstructors import *
from ptychointerim.metrics import MSELossOfSqrt

import test_utils as tutils


def test_2d_ptycho_autodiff(generate_gold=False, debug=False):
    name = 'test_2d_ptycho_autodiff'
    
    tutils.setup(name, cpu_only=False, gpu_indices=[0])
    
    dataset, probe, pixel_size_m, positions_px = tutils.load_tungsten_data(additional_opr_modes=0)
    
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
        tutils.save_gold_data(name, recon)
    else:
        tutils.run_comparison(name, recon)
        
        
def test_2d_ptycho_autodiff_opr(generate_gold=False, debug=False):
    name = 'test_2d_ptycho_autodiff_opr'
    
    tutils.setup(name, cpu_only=False, gpu_indices=[0])
    
    dataset, probe, pixel_size_m, positions_px = tutils.load_tungsten_data(additional_opr_modes=3)
    
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
        data=generate_initial_opr_mode_weights(len(positions_px), probe.shape[0]),
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
        tutils.save_gold_data(name, recon)
    else:
        tutils.run_comparison(name, recon)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate-gold', action='store_true')
    args = parser.parse_args()

    test_2d_ptycho_autodiff(generate_gold=args.generate_gold, debug=True)
    test_2d_ptycho_autodiff_opr(generate_gold=args.generate_gold, debug=True)
