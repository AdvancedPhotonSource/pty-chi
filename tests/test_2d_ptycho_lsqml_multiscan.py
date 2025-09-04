import argparse
import logging

import torch

import ptychi.api as api
from ptychi.api.task import PtychographyTask
from ptychi.utils import get_suggested_object_size, get_default_complex_dtype, generate_initial_opr_mode_weights

import test_utils as tutils


class Test2dPtychoLsqmlMultiscan(tutils.TungstenDataTester):
    
    @tutils.TungstenDataTester.wrap_recon_tester(name='test_2d_ptycho_lsqml_multiscan')
    def test_2d_ptycho_lsqml_multiscan(self):        
        self.setup_ptychi(cpu_only=False)

        data, probe, pixel_size_m, positions_px = self.load_tungsten_data(pos_type='true')
        
        # Split data and positions
        data1 = data[:500]
        data2 = data[500:]
        positions_px_1 = positions_px[:500]
        positions_px_2 = positions_px[500:]
        
        # Create task 1
        options_1 = api.LSQMLOptions()
        options_1.data_options.data = data1
        
        options_1.object_options.initial_guess = torch.ones([1, *get_suggested_object_size(positions_px, probe.shape[-2:], extra=100)], dtype=get_default_complex_dtype())
        options_1.object_options.pixel_size_m = pixel_size_m
        options_1.object_options.optimizable = True
        options_1.object_options.optimizer = api.Optimizers.SGD
        options_1.object_options.step_size = 1
        options_1.object_options.build_preconditioner_with_all_modes = True
        
        options_1.probe_options.initial_guess = probe
        options_1.probe_options.optimizable = True
        options_1.probe_options.optimizer = api.Optimizers.SGD
        options_1.probe_options.step_size = 1

        options_1.probe_position_options.position_x_px = positions_px_1[:, 1]
        options_1.probe_position_options.position_y_px = positions_px_1[:, 0]
        options_1.probe_position_options.optimizable = False
        
        options_1.reconstructor_options.batch_size = 96
        options_1.reconstructor_options.noise_model = api.NoiseModels.GAUSSIAN
        options_1.reconstructor_options.num_epochs = 8
        options_1.reconstructor_options.allow_nondeterministic_algorithms = False
        
        task_1 = PtychographyTask(options_1)
        
        # Create task 2
        options_2 = api.LSQMLOptions()
        options_2.data_options.data = data2
        
        options_2.object_options.initial_guess = torch.ones([1, *get_suggested_object_size(positions_px, probe.shape[-2:], extra=100)], dtype=get_default_complex_dtype())
        options_2.object_options.pixel_size_m = pixel_size_m
        options_2.object_options.optimizable = True
        options_2.object_options.optimizer = api.Optimizers.SGD
        options_2.object_options.step_size = 1
        options_2.object_options.build_preconditioner_with_all_modes = True
        
        options_2.probe_options.initial_guess = probe
        options_2.probe_options.optimizable = True
        options_2.probe_options.optimizer = api.Optimizers.SGD
        options_2.probe_options.step_size = 1

        options_2.probe_position_options.position_x_px = positions_px_2[:, 1]
        options_2.probe_position_options.position_y_px = positions_px_2[:, 0]
        options_2.probe_position_options.optimizable = False
        
        options_2.reconstructor_options.batch_size = 96
        options_2.reconstructor_options.noise_model = api.NoiseModels.GAUSSIAN
        options_2.reconstructor_options.num_epochs = 8
        options_2.reconstructor_options.allow_nondeterministic_algorithms = False
        
        task_2 = PtychographyTask(options_2)
        
        # Disable progress bar for task 2
        task_2.reconstructor.pbar.disable = True
        
        # Run tasks each for one epoch each time
        all_tasks = [task_1, task_2]
        for i_epoch in range(options_1.reconstructor_options.num_epochs):
            for i_task, task in enumerate(all_tasks):
                task.run(1)
                # Copy object to next task
                i_next_task = (i_task + 1) % len(all_tasks)
                all_tasks[i_next_task].copy_data_from_task(task, params_to_copy=("object",))
        
        recon = all_tasks[-1].get_data_to_cpu('object', as_numpy=True)[0]
        return recon
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate-gold', action='store_true')
    args = parser.parse_args()

    tester = Test2dPtychoLsqmlMultiscan()
    tester.setup_method(name="", generate_data=False, generate_gold=args.generate_gold, debug=True)
    tester.test_2d_ptycho_lsqml_multiscan()
