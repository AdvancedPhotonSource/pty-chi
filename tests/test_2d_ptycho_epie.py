import argparse
import os

import torch
import numpy as np

from ptychointerim.ptychotorch.data_structures import *
from ptychointerim.ptychotorch.io_handles import PtychographyDataset
from ptychointerim.forward_models import Ptychography2DForwardModel
from ptychointerim.ptychotorch.utils import (get_suggested_object_size, set_default_complex_dtype, get_default_complex_dtype, 
                            rescale_probe)
from ptychointerim.ptychotorch.reconstructors import *
from ptychointerim.ptychotorch.metrics import MSELossOfSqrt

import test_utils as tutils


def test_2d_ptycho_epie(generate_gold=False, debug=False):
    gold_dir = os.path.join('gold_data', 'test_2d_ptycho_epie')
    
    tutils.setup(gold_dir, cpu_only=True)
    
    dataset, probe, pixel_size_m, positions_px = tutils.load_tungsten_data(additional_opr_modes=0)
    probe = probe[:, [0], :, :]
    
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

    reconstructor = EPIEReconstructor(
        variable_group=Ptychography2DVariableGroup(object=object, probe=probe, probe_positions=probe_positions),
        dataset=dataset,
        batch_size=96,
        n_epochs=32,
    )
    reconstructor.build()
    reconstructor.run()

    recon = reconstructor.variable_group.object.tensor.complex().detach().cpu().numpy()
    
    if debug:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(np.abs(recon))
        ax[1].imshow(np.angle(recon))
        plt.show()
    if generate_gold:
        np.save(os.path.join(gold_dir, 'recon.npy'), recon)
    else:
        recon_gold = np.load(os.path.join(gold_dir, 'recon.npy'))
        assert np.allclose(recon, recon_gold)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate-gold', action='store_true')
    args = parser.parse_args()

    test_2d_ptycho_epie(generate_gold=args.generate_gold, debug=True)
    