import argparse

import torch
import numpy as np

import ptychi.api as api
from ptychi.api.task import PtychographyTask
from ptychi.global_settings import get_default_complex_dtype
from ptychi.utils import get_suggested_object_size

import test_utils as tutils


class Test2DPtychoLSQMLCompact(tutils.TungstenDataTester):
    
    @tutils.TungstenDataTester.wrap_recon_tester(name='test_2d_ptycho_lsqml_compact')
    def test_2d_ptycho_lsqml_compact(self):                    
        self.setup_ptychi(cpu_only=False)

        data, probe, pixel_size_m, positions_px = self.load_tungsten_data(pos_type='true')
        
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
        options.probe_position_options.step_size = 0.1
        options.probe_position_options.optimizable = True
        
        options.reconstructor_options.batch_size = 96
        options.reconstructor_options.noise_model = api.NoiseModels.GAUSSIAN
        options.reconstructor_options.num_epochs = 8
        options.reconstructor_options.batching_mode = api.BatchingModes.COMPACT
        options.reconstructor_options.compact_mode_update_clustering = True
        options.reconstructor_options.allow_nondeterministic_algorithms = False
        
        task = PtychographyTask(options)
        task.run()
        
        recon = task.get_data_to_cpu('object', as_numpy=True)[0]
        return recon
    
    @tutils.TungstenDataTester.wrap_recon_tester(name='test_2d_ptycho_lsqml_compact_no_epoch0_rescaling')
    def test_2d_ptycho_lsqml_compact_no_epoch0_rescaling(self):                    
        self.setup_ptychi(cpu_only=False)

        data, probe, pixel_size_m, positions_px = self.load_tungsten_data(pos_type='true')
        
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
        options.probe_position_options.step_size = 0.1
        options.probe_position_options.optimizable = True
        
        options.reconstructor_options.batch_size = 96
        options.reconstructor_options.noise_model = api.NoiseModels.GAUSSIAN
        options.reconstructor_options.num_epochs = 4
        options.reconstructor_options.batching_mode = api.BatchingModes.COMPACT
        options.reconstructor_options.compact_mode_update_clustering = True
        options.reconstructor_options.allow_nondeterministic_algorithms = False
        options.reconstructor_options.rescale_probe_intensity_in_first_epoch = False
        
        task = PtychographyTask(options)
        task.run()
        
        recon = task.get_data_to_cpu('object', as_numpy=True)[0]
        return recon

    @tutils.TungstenDataTester.wrap_recon_tester(name='test_2d_ptycho_lsqml_compact_multislice')
    def test_2d_ptycho_lsqml_compact_multislice(self):        
        self.setup_ptychi(cpu_only=False)

        data, probe, pixel_size_m, positions_px = self.load_tungsten_data(pos_type='true')
        
        options = api.LSQMLOptions()
        options.data_options.data = data
        
        options.object_options.initial_guess = torch.ones([2, *get_suggested_object_size(positions_px, probe.shape[-2:], extra=100)], dtype=get_default_complex_dtype())
        options.object_options.pixel_size_m = pixel_size_m
        options.object_options.optimizable = True
        options.object_options.optimizer = api.Optimizers.SGD
        options.object_options.step_size = 0.5
        options.object_options.slice_spacings_m = np.array([2e-7])
        
        options.probe_options.initial_guess = probe
        options.probe_options.optimizable = True
        options.probe_options.optimizer = api.Optimizers.SGD
        options.probe_options.step_size = 0.5

        options.probe_position_options.position_x_px = positions_px[:, 1]
        options.probe_position_options.position_y_px = positions_px[:, 0]
        options.probe_position_options.step_size = 0.1
        options.probe_position_options.optimizable = True
        
        options.reconstructor_options.batch_size = 96
        options.reconstructor_options.noise_model = api.NoiseModels.GAUSSIAN
        options.reconstructor_options.num_epochs = 8
        options.reconstructor_options.batching_mode = api.BatchingModes.COMPACT
        options.reconstructor_options.compact_mode_update_clustering = True
        options.reconstructor_options.allow_nondeterministic_algorithms = False
        task = PtychographyTask(options)
        task.run()
        
        recon = task.get_data_to_cpu('object', as_numpy=True)
        return recon
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate-gold', action='store_true')
    args = parser.parse_args()

    tester = Test2DPtychoLSQMLCompact()
    tester.setup_method(name="", generate_data=False, generate_gold=args.generate_gold, debug=True)
    tester.test_2d_ptycho_lsqml_compact()
    tester.test_2d_ptycho_lsqml_compact_no_epoch0_rescaling()
    tester.test_2d_ptycho_lsqml_compact_multislice()
