import argparse
import logging

import torch

import ptychi.api as api
from ptychi.workflows import MultiscanSharedObjectWorkflow
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
        
        # Configure scan 1
        options_1 = api.LSQMLOptions()
        diffraction_data_1 = data1
        
        object_data_1 = torch.ones([1, *get_suggested_object_size(positions_px, probe.shape[-2:], extra=100)], dtype=get_default_complex_dtype())
        options_1.object_options.pixel_size_m = pixel_size_m
        options_1.object_options.optimizable = True
        options_1.object_options.optimizer = api.Optimizers.SGD
        options_1.object_options.step_size = 1
        options_1.object_options.build_preconditioner_with_all_modes = True
        
        probe_data_1 = probe
        options_1.probe_options.optimizable = True
        options_1.probe_options.optimizer = api.Optimizers.SGD
        options_1.probe_options.step_size = 1

        probe_position_x_px_1 = positions_px_1[:, 1]
        probe_position_y_px_1 = positions_px_1[:, 0]
        options_1.probe_position_options.optimizable = False
        
        options_1.reconstructor_options.batch_size = 96
        options_1.reconstructor_options.noise_model = api.NoiseModels.GAUSSIAN
        options_1.reconstructor_options.num_epochs = 8
        options_1.reconstructor_options.allow_nondeterministic_algorithms = False
        
        # Configure scan 2
        options_2 = api.LSQMLOptions()
        diffraction_data_2 = data2
        
        object_data_2 = torch.ones([1, *get_suggested_object_size(positions_px, probe.shape[-2:], extra=100)], dtype=get_default_complex_dtype())
        options_2.object_options.pixel_size_m = pixel_size_m
        options_2.object_options.optimizable = True
        options_2.object_options.optimizer = api.Optimizers.SGD
        options_2.object_options.step_size = 1
        options_2.object_options.build_preconditioner_with_all_modes = True
        
        probe_data_2 = probe
        options_2.probe_options.optimizable = True
        options_2.probe_options.optimizer = api.Optimizers.SGD
        options_2.probe_options.step_size = 1

        probe_position_x_px_2 = positions_px_2[:, 1]
        probe_position_y_px_2 = positions_px_2[:, 0]
        options_2.probe_position_options.optimizable = False
        
        options_2.reconstructor_options.batch_size = 96
        options_2.reconstructor_options.noise_model = api.NoiseModels.GAUSSIAN
        options_2.reconstructor_options.num_epochs = 8
        options_2.reconstructor_options.allow_nondeterministic_algorithms = False
        
        workflow = MultiscanSharedObjectWorkflow(
            [options_1, options_2],
            diffraction_data=[diffraction_data_1, diffraction_data_2],
            object_data=[object_data_1, object_data_2],
            probe_data=[probe_data_1, probe_data_2],
            probe_position_x_px=[probe_position_x_px_1, probe_position_x_px_2],
            probe_position_y_px=[probe_position_y_px_1, probe_position_y_px_2],
            workflow_options=api.MultiscanSharedObjectWorkflowOptions(
                num_outer_epochs=options_1.reconstructor_options.num_epochs,
                num_inner_epochs=1,
            ),
        )
        workflow.run()

        recon = workflow.tasks[-1].get_data_to_cpu('object', as_numpy=True)[0]
        return recon
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate-gold', action='store_true')
    args = parser.parse_args()

    tester = Test2dPtychoLsqmlMultiscan()
    tester.setup_method(name="", generate_data=False, generate_gold=args.generate_gold, debug=True)
    tester.test_2d_ptycho_lsqml_multiscan()
