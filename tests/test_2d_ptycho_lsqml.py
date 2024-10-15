import argparse

import torch

import ptychointerim.api as api
from ptychointerim.api.task import PtychographyTask
from ptychointerim.ptychotorch.utils import get_suggested_object_size, get_default_complex_dtype, generate_initial_opr_mode_weights

import test_utils as tutils


def test_2d_ptycho_lsqml(pytestconfig, generate_gold=False, debug=False, high_tol=False):
    if pytestconfig is not None:
        high_tol = pytestconfig.getoption("high_tol")
        
    name = 'test_2d_ptycho_lsqml'
    
    tutils.setup(name, cpu_only=False, gpu_indices=[0])

    data, probe, pixel_size_m, positions_px = tutils.load_tungsten_data(pos_type='true')
    
    options = api.LSQMLOptions()
    options.data_options.data = data
    
    options.object_options.initial_guess = torch.ones(get_suggested_object_size(positions_px, probe.shape[-2:], extra=100), dtype=get_default_complex_dtype())
    options.object_options.pixel_size_m = pixel_size_m
    options.object_options.optimizable = True
    options.object_options.optimizer = api.Optimizers.SGD
    options.object_options.step_size = 1
    
    options.probe_options.initial_guess = probe
    options.probe_options.optimizable = True
    options.probe_options.optimizer = api.Optimizers.SGD
    options.probe_options.step_size = 1

    options.probe_position_options.position_x_m = positions_px[:, 1]
    options.probe_position_options.position_y_m = positions_px[:, 0]
    options.probe_position_options.pixel_size_m = 1
    options.probe_position_options.optimizable = False
    
    options.reconstructor_options.batch_size = 96
    options.reconstructor_options.noise_model = api.NoiseModels.GAUSSIAN
    options.reconstructor_options.num_epochs = 8
    
    task = PtychographyTask(options)
    task.run()
    
    recon = task.get_data_to_cpu('object', as_numpy=True)

    if generate_gold:
        tutils.save_gold_data(name, recon)
    else:
        tutils.run_comparison(name, recon, high_tol=high_tol)
    
    
def test_2d_ptycho_lsqml_poscorr(pytestconfig, generate_gold=False, debug=False, high_tol=False):
    if pytestconfig is not None:
        high_tol = pytestconfig.getoption("high_tol")
        
    name = 'test_2d_ptycho_lsqml_poscorr'
    
    tutils.setup(name, cpu_only=False, gpu_indices=[0])

    data, probe, pixel_size_m, positions_px = tutils.load_tungsten_data(pos_type='nominal')
    
    options = api.LSQMLOptions()
    options.data_options.data = data
    
    options.object_options.initial_guess = torch.ones(get_suggested_object_size(positions_px, probe.shape[-2:], extra=100), dtype=get_default_complex_dtype())
    options.object_options.pixel_size_m = pixel_size_m
    options.object_options.optimizable = True
    options.object_options.optimizer = api.Optimizers.SGD
    options.object_options.step_size = 1
    
    options.probe_options.initial_guess = probe
    options.probe_options.optimizable = True
    options.probe_options.optimizer = api.Optimizers.SGD
    options.probe_options.step_size = 1

    options.probe_position_options.position_x_m = positions_px[:, 1]
    options.probe_position_options.position_y_m = positions_px[:, 0]
    options.probe_position_options.pixel_size_m = 1
    options.probe_position_options.optimizable = True
    options.probe_position_options.optimizer = api.Optimizers.SGD
    options.probe_position_options.step_size = 0.1
    options.probe_position_options.update_magnitude_limit = 2.0
    options.probe_position_options.correction_options.correction_type = api.PositionCorrectionTypes.GRADIENT

    options.reconstructor_options.batch_size = 96
    options.reconstructor_options.noise_model = api.NoiseModels.GAUSSIAN
    options.reconstructor_options.num_epochs = 8
        
    task = PtychographyTask(options)
    task.run()
    
    recon = task.get_data_to_cpu('object', as_numpy=True)
    
    if debug and not generate_gold:
        tutils.plot_complex_image(recon)
    if generate_gold:
        tutils.save_gold_data(name, recon)
    else:
        tutils.run_comparison(name, recon, high_tol=high_tol)
    

def test_2d_ptycho_lsqml_opr(pytestconfig, generate_gold=False, debug=False, high_tol=False):
    if pytestconfig is not None:
        high_tol = pytestconfig.getoption("high_tol")
        
    name = 'test_2d_ptycho_lsqml_opr'
    
    tutils.setup(name, cpu_only=False, gpu_indices=[0])

    data, probe, pixel_size_m, positions_px = tutils.load_tungsten_data(pos_type='true', additional_opr_modes=3)
    
    options = api.LSQMLOptions()
    options.data_options.data = data
    
    options.object_options.initial_guess = torch.ones(get_suggested_object_size(positions_px, probe.shape[-2:], extra=100), dtype=get_default_complex_dtype())
    options.object_options.pixel_size_m = pixel_size_m
    options.object_options.optimizable = True
    options.object_options.optimizer = api.Optimizers.SGD
    options.object_options.step_size = 1
    
    options.probe_options.initial_guess = probe
    options.probe_options.optimizable = True
    options.probe_options.optimizer = api.Optimizers.SGD
    options.probe_options.step_size = 1

    options.probe_position_options.position_x_m = positions_px[:, 1]
    options.probe_position_options.position_y_m = positions_px[:, 0]
    options.probe_position_options.pixel_size_m = 1
    options.probe_position_options.optimizable = False
    
    options.opr_mode_weight_options.initial_weights = generate_initial_opr_mode_weights(len(positions_px), probe.shape[0], eigenmode_weight=0.1)
    options.opr_mode_weight_options.optimizable = True
    options.opr_mode_weight_options.update_relaxation = 0.1
    
    options.reconstructor_options.batch_size = 44
    options.reconstructor_options.noise_model = api.NoiseModels.GAUSSIAN
    options.reconstructor_options.num_epochs = 8
    
    task = PtychographyTask(options)
    task.run()
    
    recon = task.get_data_to_cpu('object', as_numpy=True)

    if generate_gold:
        tutils.save_gold_data(name, recon)
    else:
        tutils.run_comparison(name, recon, high_tol=high_tol)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate-gold', action='store_true')
    parser.add_argument('--high-tol', action='store_true')
    args = parser.parse_args()

    # test_2d_ptycho_lsqml(None, generate_gold=args.generate_gold, debug=True, high_tol=args.high_tol)
    test_2d_ptycho_lsqml_poscorr(None, generate_gold=args.generate_gold, debug=True, high_tol=args.high_tol)
    # test_2d_ptycho_lsqml_opr(None, generate_gold=args.generate_gold, debug=True, high_tol=args.high_tol)
