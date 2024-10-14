import argparse

import torch

import ptychointerim.api as api
from ptychointerim.api.task import PtychographyTask
from ptychointerim.ptychotorch.utils import get_suggested_object_size, get_default_complex_dtype

import test_utils as tutils


def test_2d_ptycho_epie_position_correction(pytestconfig, generate_gold=False, debug=False, high_tol=False):
    if pytestconfig is not None:
        high_tol = pytestconfig.getoption("high_tol")

    name = 'test_2d_ptycho_epie_position_correction'
    
    tutils.setup(name, cpu_only=False, gpu_indices=[0])
    
    data, probe, pixel_size_m, positions_px = tutils.load_tungsten_data(additional_opr_modes=0)
    probe = probe[:, [0], :, :]
    
    options = api.EPIEOptions()
    
    options.data_options.data = data
    
    options.object_options.initial_guess = torch.ones(get_suggested_object_size(positions_px, probe.shape[-2:], extra=100), dtype=get_default_complex_dtype())
    options.object_options.pixel_size_m = pixel_size_m
    options.object_options.optimizable = True
    options.object_options.optimizer = api.Optimizers.SGD
    options.object_options.step_size = 0.1
    options.object_options.alpha = 1
    
    options.probe_options.initial_guess = probe
    options.probe_options.optimizable = True
    options.probe_options.optimizer = api.Optimizers.SGD
    options.probe_options.step_size = 0.1
    options.probe_options.alpha = 1

    options.probe_position_options.position_x_m = positions_px[:, 1]
    options.probe_position_options.position_y_m = positions_px[:, 0]
    options.probe_position_options.pixel_size_m = 1
    options.probe_position_options.optimizable = True
    options.probe_position_options.optimizer = api.Optimizers.SGD
    options.probe_position_options.step_size = 1000
    options.probe_position_options.correction_options.correction_type = api.PositionCorrectionTypes.CROSS_CORRELATION
    
    options.reconstructor_options.batch_size = 96
    options.reconstructor_options.num_epochs = 32
    
    task = PtychographyTask(options)
    task.run()

    recon = task.get_data_to_cpu('object', as_numpy=True)
    
    if debug and not generate_gold:
        tutils.plot_complex_image(recon)
    if generate_gold:
        tutils.save_gold_data(name, recon)
    else:
        tutils.run_comparison(name, recon, high_tol=high_tol)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate-gold', action='store_true')
    parser.add_argument('--high-tol', action='store_true')
    args = parser.parse_args()

    test_2d_ptycho_epie_position_correction(None, generate_gold=args.generate_gold, debug=True, high_tol=args.high_tol)
    