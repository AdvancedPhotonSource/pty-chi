import datetime
import json

import torch
import h5py
import numpy as np
import matplotlib.pyplot as plt
import tifffile

from ptytorch.data_structures import *
from ptytorch.io_handles import PtychographyDataset
from ptytorch.forward_models import Ptychography2DForwardModel
from ptytorch.utils import (get_suggested_object_size, set_default_complex_dtype, get_default_complex_dtype, 
                            rescale_probe)
from ptytorch.reconstructors import *
from ptytorch.metrics import MSELossOfSqrt


torch.set_default_device('cuda')
torch.set_default_dtype(torch.float32)
set_default_complex_dtype(torch.complex64)

patterns = h5py.File('data/dp_250.hdf5', 'r')['dp'][...]
dataset = PtychographyDataset(patterns)

f_meta = h5py.File('data/metadata_250_nominalPos.hdf5', 'r')
probe = f_meta['probe'][...]

probe = rescale_probe(probe, patterns)
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
    optimizable=True,
    optimizer_class=torch.optim.Adam,
    optimizer_params={'lr': 1e-1}
)

reconstructor = AutodiffReconstructor(
    variable_group=Ptychography2DVariableGroup(object=object, probe=probe, probe_positions=probe_positions),
    dataset=dataset,
    forward_model_class=Ptychography2DForwardModel,
    batch_size=96,
    loss_function=MSELossOfSqrt(),
    n_epochs=64
)
# reconstructor = EPIEReconstructor(
#     variable_group=Ptychography2DVariableGroup(object=object, probe=probe, probe_positions=probe_positions),
#     dataset=dataset,
#     batch_size=96,
#     n_epochs=64,
# )
reconstructor.build()
reconstructor.run()

recon = reconstructor.variable_group.object.tensor.complex().detach().cpu().numpy()
fig, ax = plt.subplots(1, 2)
ax[0].imshow(np.angle(recon)); ax[0].set_title('Phase')
ax[1].imshow(np.abs(recon)); ax[1].set_title('Magnitude')
timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
plt.savefig('outputs/recon_{}.png'.format(timestamp))
tifffile.imwrite('outputs/recon_phase_{}.tif'.format(timestamp), np.angle(recon))
tifffile.imwrite('outputs/recon_mag_{}.tif'.format(timestamp), np.abs(recon))
json.dump(reconstructor.get_config_dict(), open('outputs/recon_{}.json'.format(timestamp), 'w'))

pos = reconstructor.variable_group.probe_positions.tensor.detach().cpu().numpy()
f_meta = h5py.File('data/metadata_250_truePos.hdf5', 'r')
positions = np.stack([f_meta['probe_position_y_m'][...], f_meta['probe_position_x_m'][...]], axis=1)
pos_true = positions / pixel_size_m
plt.figure()
plt.plot(pos[:, 1], pos[:, 0], marker='.', label='corrected')
plt.plot(pos_true[:, 1], pos_true[:, 0], marker='.', label='true')
plt.legend()
plt.show()

