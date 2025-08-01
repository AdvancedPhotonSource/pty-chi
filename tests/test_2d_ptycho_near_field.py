import argparse

import torch

import ptychi.api as api
from ptychi.api.task import PtychographyTask
from ptychi.global_settings import get_default_complex_dtype
from ptychi.utils import get_suggested_object_size

import test_utils as tutils


class Test2dPtychoNearField(tutils.BaseTester):
    
    @tutils.BaseTester.wrap_recon_tester(name='test_2d_ptycho_near_field_lsqml')
    def test_2d_ptycho_near_field_lsqml(self):        
        self.setup_ptychi(cpu_only=False)

        data, probe, pixel_size_m, positions_px = self.load_data_ptychodus(
            *self.get_default_input_data_file_paths('2d_nf_ptycho'),
            subtract_position_mean=False,
            scale_probe_magnitude=False
        )
        
        options = api.LSQMLOptions()
        options.data_options.data = data
        options.data_options.wavelength_m = 1240 / 33.35 * 1e-12
        options.data_options.free_space_propagation_distance_m = 0.0043
        options.data_options.fft_shift = False
        
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
        
        options.reconstructor_options.batch_size = 16
        options.reconstructor_options.noise_model = api.NoiseModels.GAUSSIAN
        options.reconstructor_options.num_epochs = 32
        options.reconstructor_options.allow_nondeterministic_algorithms = False
        options.reconstructor_options.forward_model_options.pad_for_shift = 50
        
        task = PtychographyTask(options)
        task.run()
        
        recon = task.get_data_to_cpu('object', as_numpy=True)[0]
        return recon
    
    @tutils.BaseTester.wrap_recon_tester(name='test_2d_ptycho_near_field_ad')
    def test_2d_ptycho_near_field_ad(self):        
        self.setup_ptychi(cpu_only=False)

        data, probe, pixel_size_m, positions_px = self.load_data_ptychodus(
            *self.get_default_input_data_file_paths('2d_nf_ptycho'),
            subtract_position_mean=False,
            scale_probe_magnitude=False
        )
        
        options = api.AutodiffPtychographyOptions()
        options.data_options.data = data
        options.data_options.wavelength_m = 1240 / 33.35 * 1e-12
        options.data_options.free_space_propagation_distance_m = 0.0043
        options.data_options.fft_shift = False
        
        options.object_options.initial_guess = torch.ones([1, *get_suggested_object_size(positions_px, probe.shape[-2:], extra=100)], dtype=get_default_complex_dtype())
        options.object_options.pixel_size_m = pixel_size_m
        options.object_options.optimizable = True
        options.object_options.optimizer = api.Optimizers.ADAM
        options.object_options.step_size = 1e-2
        
        options.probe_options.initial_guess = probe
        options.probe_options.optimizable = True
        options.probe_options.optimizer = api.Optimizers.ADAM
        options.probe_options.step_size = 1e-2

        options.probe_position_options.position_x_px = positions_px[:, 1]
        options.probe_position_options.position_y_px = positions_px[:, 0]
        options.probe_position_options.optimizable = False
        
        options.reconstructor_options.batch_size = 16
        options.reconstructor_options.num_epochs = 32
        options.reconstructor_options.allow_nondeterministic_algorithms = False
        options.reconstructor_options.forward_model_options.pad_for_shift = 50
        
        task = PtychographyTask(options)
        task.run()
        
        recon = task.get_data_to_cpu('object', as_numpy=True)[0]
        return recon


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate-gold', action='store_true')
    args = parser.parse_args()

    tester = Test2dPtychoNearField()
    tester.setup_method(name="", generate_data=False, generate_gold=args.generate_gold, debug=True)
    tester.test_2d_ptycho_near_field_lsqml()
    tester.test_2d_ptycho_near_field_ad()
