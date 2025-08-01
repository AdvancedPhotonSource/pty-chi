import argparse

import torch
import numpy as np

import ptychi.api as api
from ptychi.api.task import PtychographyTask
from ptychi.global_settings import get_default_complex_dtype
from ptychi.utils import get_suggested_object_size

import test_utils as tutils


class TestPtychoLowMemForwardModel(tutils.TungstenDataTester):
    def test_ptycho_low_mem_forward_model(self):
        name = 'test_ptycho_low_mem_forward_model'
        
        self.setup_ptychi(cpu_only=False)
        
        data, probe, pixel_size_m, positions_px = self.load_tungsten_data(pos_type='true')

        options = api.LSQMLOptions()
        options.data_options.data = data
            
        options.object_options.initial_guess = torch.ones([3, *get_suggested_object_size(positions_px, probe.shape[-2:], extra=100)], dtype=get_default_complex_dtype())
        options.object_options.pixel_size_m = pixel_size_m
        options.object_options.slice_spacings_m = np.array([3e-6, 3e-6])
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
        
        options.reconstructor_options.batch_size = 96
        options.reconstructor_options.noise_model = api.NoiseModels.GAUSSIAN
        options.reconstructor_options.num_epochs = 8
        
        task = PtychographyTask(options)
        fm = task.reconstructor.forward_model
        
        y_normal = fm.forward(torch.tensor([0, 1]).long())
        y_lowmem = fm.forward_low_memory(torch.tensor([0, 1]).long())
        
        assert torch.allclose(y_normal, y_lowmem)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate-gold', action='store_true')
    args = parser.parse_args()

    tester = TestPtychoLowMemForwardModel()
    tester.setup_method(name="", generate_data=False, generate_gold=args.generate_gold, debug=True)
    tester.test_ptycho_low_mem_forward_model()
