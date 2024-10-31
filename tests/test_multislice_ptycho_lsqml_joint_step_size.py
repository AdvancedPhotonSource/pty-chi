import argparse

import torch

import ptychi.api as api
from ptychi.api.task import PtychographyTask
from ptychi.ptychotorch.utils import get_suggested_object_size, get_default_complex_dtype, generate_initial_opr_mode_weights

import test_utils as tutils


def test_multislice_ptycho_lsqml_joint_step_size(pytestconfig, generate_gold=False, debug=False, high_tol=False):
    if pytestconfig is not None:
        high_tol = pytestconfig.getoption("high_tol")
        
    name = 'test_multislice_ptycho_lsqml_joint_step_size'
    
    tutils.setup(name, cpu_only=False, gpu_indices=[0])

    data, probe, pixel_size_m, positions_px = tutils.load_tungsten_data(pos_type='true', additional_opr_modes=3)
    
    options = api.LSQMLOptions()
    options.data_options.data = data
    
    options.object_options.initial_guess = torch.ones([1, *get_suggested_object_size(positions_px, probe.shape[-2:], extra=100)], dtype=get_default_complex_dtype())
    options.object_options.pixel_size_m = pixel_size_m
    options.object_options.optimizable = True
    options.object_options.optimizer = api.Optimizers.SGD
    options.object_options.step_size = 1
    
    options.probe_options.initial_guess = probe
    options.probe_options.optimizable = True
    options.probe_options.optimizer = api.Optimizers.SGD
    options.probe_options.step_size = 1

    options.probe_position_options.position_x_px = positions_px[:, 1]
    options.probe_position_options.position_y_px = positions_px[:, 0]
    options.probe_position_options.optimizable = False
    
    options.opr_mode_weight_options.initial_weights = generate_initial_opr_mode_weights(len(positions_px), probe.shape[0], eigenmode_weight=0.1)
    options.opr_mode_weight_options.optimizable = True
    options.opr_mode_weight_options.update_relaxation = 0.1
    
    options.reconstructor_options.batch_size = 44
    options.reconstructor_options.noise_model = api.NoiseModels.GAUSSIAN
    options.reconstructor_options.num_epochs = 8
    options.reconstructor_options.solve_obj_prb_step_size_jointly_for_first_slice_in_multislice = True

    task = PtychographyTask(options)
    task.run()
    
    recon = task.get_data_to_cpu('object', as_numpy=True)[0]

    if generate_gold:
        tutils.save_gold_data(name, recon)
    else:
        tutils.run_comparison(name, recon, high_tol=high_tol)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate-gold', action='store_true')
    parser.add_argument('--high-tol', action='store_true')
    args = parser.parse_args()

    test_multislice_ptycho_lsqml_joint_step_size(None, generate_gold=args.generate_gold, debug=True, high_tol=args.high_tol)
