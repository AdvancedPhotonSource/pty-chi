import argparse

import torch
import numpy as np

import ptychi.api as api
from ptychi.api.task import PtychographyTask
from ptychi.api import LSQMLOptions
import ptychi.utils as utils

import test_utils as tutils


class Test2dPtychoOptPlan(tutils.TungstenDataTester):
    @tutils.TungstenDataTester.wrap_recon_tester(name='test_2d_ptycho_opt_plan')
    def test_lsqml_e2e(self):
        self.setup_ptychi(cpu_only=False)

        data, probe, pixel_size_m, positions_px = self.load_tungsten_data(pos_type='true', additional_opr_modes=3)
        object_init = torch.ones(
            [1, *utils.get_suggested_object_size(positions_px, probe.shape[-2:], extra=100)], 
            dtype=utils.get_default_complex_dtype()
        )
        
        options = LSQMLOptions()
        diffraction_data = data
        
        object_data = object_init
        options.object_options.pixel_size_m = pixel_size_m
        options.object_options.optimizable = True
        options.object_options.optimizer = api.Optimizers.SGD
        options.object_options.step_size = 1
        
        probe_data = probe
        options.probe_options.optimizable = True
        options.probe_options.optimization_plan.start = 2
        options.probe_options.optimization_plan.stop = None
        options.probe_options.optimizer = api.Optimizers.SGD
        options.probe_options.step_size = 1
        
        probe_position_x_px = positions_px[:, 1]
        probe_position_y_px = positions_px[:, 0]
        options.probe_position_options.correction_options.update_magnitude_limit = 1.0
        options.probe_position_options.optimizable = True
        options.probe_position_options.optimization_plan.start = 1
        options.probe_position_options.optimization_plan.stop = 3
        options.probe_position_options.optimization_plan.stride = 2
        options.probe_position_options.optimizer = api.Optimizers.ADAM
        options.probe_position_options.step_size = 1e-1
        
        opr_mode_weights_data = np.array([1, 1e-6, 1e-6, 1e-6])
        options.opr_mode_weight_options.optimize_intensity_variation = True
        options.opr_mode_weight_options.optimizable = True
        
        options.reconstructor_options.num_epochs = 4
        options.reconstructor_options.batch_size = 40
        options.reconstructor_options.default_device = api.Devices.GPU
        options.reconstructor_options.displayed_loss_function = api.LossFunctions.MSE_SQRT
        options.reconstructor_options.allow_nondeterministic_algorithms = False
        
        task = PtychographyTask(
            options,
            diffraction_data=diffraction_data,
            object_data=object_data,
            probe_data=probe_data,
            probe_position_x_px=probe_position_x_px,
            probe_position_y_px=probe_position_y_px,
            opr_mode_weights_data=opr_mode_weights_data,
        )
        task.run()
            
        recon = task.get_data_to_cpu(name='object', as_numpy=True)[0]
        return recon



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate-gold", action="store_true")
    args = parser.parse_args()
    tester = Test2dPtychoOptPlan()
    tester.setup_method(name="", generate_data=False, generate_gold=args.generate_gold, debug=False)
    tester.test_lsqml_e2e()
