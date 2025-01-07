import argparse
import logging

import torch
import numpy as np

import ptychi.api as api
from ptychi.api.task import PtychographyTask
from ptychi.api import LSQMLOptions, AutodiffPtychographyOptions
import ptychi.utils as utils

import test_utils as tutils


class Tester2DPtychoProbePowerConstraint(tutils.TungstenDataTester):

    @tutils.TungstenDataTester.wrap_recon_tester(name='test_2d_ptycho_probe_power_constraint_lsqml')
    def test_2d_ptycho_probe_power_constraint_lsqml(self):
        self.setup_ptychi(cpu_only=False)

        data, probe, pixel_size_m, positions_px = self.load_tungsten_data(pos_type='true', additional_opr_modes=3)
        object_init = torch.ones(
            [1, *utils.get_suggested_object_size(positions_px, probe.shape[-2:], extra=100)], 
            dtype=utils.get_default_complex_dtype()
        )
        
        options = LSQMLOptions()
        options.data_options.data = data
        
        options.object_options.initial_guess = object_init
        options.object_options.pixel_size_m = pixel_size_m
        options.object_options.optimizable = True
        options.object_options.optimizer = api.Optimizers.SGD
        options.object_options.step_size = 1
        
        options.probe_options.initial_guess = probe
        options.probe_options.optimizable = True
        options.probe_options.optimizer = 'sgd'
        options.probe_options.step_size = 1
        options.probe_options.power_constraint.probe_power = data[0].sum()
        options.probe_options.power_constraint.enabled = True
        
        options.probe_position_options.position_x_px = positions_px[:, 1]
        options.probe_position_options.position_y_px = positions_px[:, 0]
        options.probe_position_options.update_magnitude_limit = 1.0
        options.probe_position_options.optimizable = True
        options.probe_position_options.optimizer = api.Optimizers.ADAM
        options.probe_position_options.step_size = 1e-1
        
        options.opr_mode_weight_options.initial_weights = np.array([1, 1e-6, 1e-6, 1e-6])
        options.opr_mode_weight_options.optimize_intensity_variation.enabled = True
        options.opr_mode_weight_options.optimizable = True
        
        options.reconstructor_options.num_epochs = 4
        options.reconstructor_options.batch_size = 40
        options.reconstructor_options.default_device = api.Devices.GPU
        options.reconstructor_options.displayed_loss_function = api.LossFunctions.MSE_SQRT
        
        with PtychographyTask(options) as task:
            task.run()
            # This should be equivalent to:
            # for _ in range(64):
            #     task.run(1)
            
            recon = task.get_data_to_cpu(name='object', as_numpy=True)[0]
        return recon
        
    @tutils.BaseTester.wrap_recon_tester(name='test_2d_ptycho_probe_power_constraint_ad')
    def test_2d_ptycho_probe_power_constraint_ad(self):
        self.setup_ptychi(cpu_only=False)

        data, probe, pixel_size_m, positions_px = self.load_tungsten_data(pos_type='true', additional_opr_modes=3)
        object_init = torch.ones(
            [1, *utils.get_suggested_object_size(positions_px, probe.shape[-2:], extra=100)], 
            dtype=utils.get_default_complex_dtype()
        )
        
        options = AutodiffPtychographyOptions()
        options.data_options.data = data
        
        options.object_options.initial_guess = object_init
        options.object_options.pixel_size_m = pixel_size_m
        options.object_options.optimizable = True
        options.object_options.optimizer = api.Optimizers.SGD
        options.object_options.step_size = 1e-1
        
        options.probe_options.initial_guess = probe
        options.probe_options.optimizable = True
        options.probe_options.optimizer = api.Optimizers.SGD
        options.probe_options.step_size = 1e-1
        options.probe_options.power_constraint.probe_power = data[0].sum()
        options.probe_options.power_constraint.enabled = True
        
        options.probe_position_options.position_x_px = positions_px[:, 1]
        options.probe_position_options.position_y_px = positions_px[:, 0]
        options.probe_position_options.update_magnitude_limit = 1.0
        options.probe_position_options.optimizable = True
        options.probe_position_options.optimizer = api.Optimizers.SGD
        options.probe_position_options.step_size = 1e-1
        
        options.opr_mode_weight_options.initial_weights = np.array([1, 1e-6, 1e-6, 1e-6])
        options.opr_mode_weight_options.optimize_intensity_variation.enabled = True
        options.opr_mode_weight_options.optimizable = True
        options.opr_mode_weight_options.optimizer = api.Optimizers.SGD
        options.opr_mode_weight_options.step_size = 1e-3
        
        options.reconstructor_options.num_epochs = 4
        options.reconstructor_options.batch_size = 40
        options.reconstructor_options.default_device = api.Devices.GPU
        options.reconstructor_options.displayed_loss_function = api.LossFunctions.MSE_SQRT
        
        with PtychographyTask(options) as task:
            task.run()
            # This should be equivalent to:
            # for _ in range(64):
            #     task.run(1)
            
            recon = task.get_data_to_cpu(name='object', as_numpy=True)[0]
            
        return recon
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate-gold', action='store_true')
    args = parser.parse_args()

    tester = Tester2DPtychoProbePowerConstraint()
    tester.setup_method(name="", generate_data=False, generate_gold=args.generate_gold, debug=True)
    tester.test_2d_ptycho_probe_power_constraint_ad()
    tester.test_2d_ptycho_probe_power_constraint_lsqml()
    