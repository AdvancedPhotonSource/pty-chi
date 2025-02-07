import argparse

import torch

import ptychi.api as api
from ptychi.api.task import PtychographyTask
from ptychi.utils import get_suggested_object_size, get_default_complex_dtype, generate_initial_opr_mode_weights

import test_utils as tutils


class Test2dPtychoLsqml(tutils.TungstenDataTester):
    
    @tutils.TungstenDataTester.wrap_recon_tester(name='test_2d_ptycho_lsqml')
    def test_2d_ptycho_lsqml(self):        
        self.setup_ptychi(cpu_only=False)

        data, probe, pixel_size_m, positions_px = self.load_tungsten_data(pos_type='true')
        
        options = api.LSQMLOptions()
        options.data_options.data = data
        
        options.object_options.initial_guess = torch.ones([1, *get_suggested_object_size(positions_px, probe.shape[-2:], extra=100)], dtype=get_default_complex_dtype())
        options.object_options.pixel_size_m = pixel_size_m
        options.object_options.optimizable = True
        options.object_options.optimizer = api.Optimizers.SGD
        options.object_options.step_size = 1
        options.object_options.build_preconditioner_with_all_modes = True
        
        options.probe_options.initial_guess = probe
        options.probe_options.optimizable = True
        options.probe_options.optimizer = api.Optimizers.SGD
        options.probe_options.step_size = 1

        options.probe_position_options.position_x_px = positions_px[:, 1]
        options.probe_position_options.position_y_px = positions_px[:, 0]
        options.probe_position_options.optimizable = False
        
        options.reconstructor_options.batch_size = 96
        options.reconstructor_options.noise_model = api.NoiseModels.GAUSSIAN
        options.reconstructor_options.num_epochs = 8
        
        task = PtychographyTask(options)
        task.run()
        
        recon = task.get_data_to_cpu('object', as_numpy=True)[0]
        return recon
    
    
    @tutils.TungstenDataTester.wrap_recon_tester(name='test_2d_ptycho_lsqml_poscorr')
    def test_2d_ptycho_lsqml_poscorr(self):        
        self.setup_ptychi(cpu_only=False)

        data, probe, pixel_size_m, positions_px = self.load_tungsten_data(pos_type='nominal')
        
        options = api.LSQMLOptions()
        options.data_options.data = data
        
        options.object_options.initial_guess = torch.ones([1, *get_suggested_object_size(positions_px, probe.shape[-2:], extra=100)], dtype=get_default_complex_dtype())
        options.object_options.pixel_size_m = pixel_size_m
        options.object_options.optimizable = True
        options.object_options.optimizer = api.Optimizers.SGD
        options.object_options.step_size = 1
        options.object_options.build_preconditioner_with_all_modes = True
        
        options.probe_options.initial_guess = probe
        options.probe_options.optimizable = True
        options.probe_options.optimizer = api.Optimizers.SGD
        options.probe_options.step_size = 1

        options.probe_position_options.position_x_px = positions_px[:, 1]
        options.probe_position_options.position_y_px = positions_px[:, 0]
        options.probe_position_options.optimizable = True
        options.probe_position_options.optimizer = api.Optimizers.SGD
        options.probe_position_options.magnitude_limit.enabled = True
        options.probe_position_options.magnitude_limit.limit = 5.0
        options.probe_position_options.correction_options.correction_type = api.PositionCorrectionTypes.GRADIENT

        options.reconstructor_options.batch_size = 96
        options.reconstructor_options.noise_model = api.NoiseModels.GAUSSIAN
        options.reconstructor_options.num_epochs = 32
            
        task = PtychographyTask(options)
        task.run()
        
        recon = task.get_data_to_cpu('object', as_numpy=True)[0]
        if self.debug and not self.generate_gold:
            pos = task.get_data_to_cpu('probe_positions', as_numpy=True)
            _, _, _, pos_true = self.load_tungsten_data(pos_type='true')
            tutils.plot_probe_positions(pos, pos_true)
        return recon
    
    @tutils.TungstenDataTester.wrap_recon_tester(name='test_2d_ptycho_lsqml_opr')
    def test_2d_ptycho_lsqml_opr(self):        
        self.setup_ptychi(cpu_only=False)

        data, probe, pixel_size_m, positions_px = self.load_tungsten_data(pos_type='true', additional_opr_modes=3)
        
        options = api.LSQMLOptions()
        options.data_options.data = data
        
        options.object_options.initial_guess = torch.ones([1, *get_suggested_object_size(positions_px, probe.shape[-2:], extra=100)], dtype=get_default_complex_dtype())
        options.object_options.pixel_size_m = pixel_size_m
        options.object_options.optimizable = True
        options.object_options.optimizer = api.Optimizers.SGD
        options.object_options.step_size = 1
        options.object_options.build_preconditioner_with_all_modes = True
        
        options.probe_options.initial_guess = probe
        options.probe_options.optimizable = True
        options.probe_options.optimizer = api.Optimizers.SGD
        options.probe_options.step_size = 1

        options.probe_position_options.position_x_px = positions_px[:, 1]
        options.probe_position_options.position_y_px = positions_px[:, 0]
        options.probe_position_options.optimizable = False
        
        options.opr_mode_weight_options.initial_weights = generate_initial_opr_mode_weights(len(positions_px), probe.shape[0])
        options.opr_mode_weight_options.optimizable = True
        
        options.reconstructor_options.batch_size = 44
        options.reconstructor_options.noise_model = api.NoiseModels.GAUSSIAN
        options.reconstructor_options.num_epochs = 32
        
        task = PtychographyTask(options)
        task.run()
        
        recon = task.get_data_to_cpu('object', as_numpy=True)[0]
        return recon
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate-gold', action='store_true')
    args = parser.parse_args()

    tester = Test2dPtychoLsqml()
    tester.setup_method(name="", generate_data=False, generate_gold=args.generate_gold, debug=True)
    tester.test_2d_ptycho_lsqml()
    tester.test_2d_ptycho_lsqml_poscorr()
    tester.test_2d_ptycho_lsqml_opr()
