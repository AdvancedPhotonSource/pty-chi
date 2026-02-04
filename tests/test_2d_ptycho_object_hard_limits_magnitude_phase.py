import argparse

import torch
import pytest

import ptychi.api as api
from ptychi.api.task import PtychographyTask
from ptychi.utils import get_suggested_object_size, get_default_complex_dtype, generate_initial_opr_mode_weights

import test_utils as tutils

import os
os.environ["PTYCHO_CI_DATA_DIR"] = "/net/s8iddata/export/8-id-ECA/Analysis/atripath/ptychointerim-data/ci_data"

class Tester2DPtychoObjectHardLimitsMagnitudePhase(tutils.TungstenDataTester):
    
    @pytest.mark.local
    @tutils.TungstenDataTester.wrap_recon_tester(name='test_2d_ptycho_object_hard_limits_magnitude_phase')
    def test_2d_ptycho_object_hard_limits_magnitude_phase(self):        
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
        
        options.object_options.hard_limits_magnitude_phase.enabled = True
        options.object_options.hard_limits_magnitude_phase.abs_lim = torch.asarray([0.5, 1.2])
        options.object_options.hard_limits_magnitude_phase.phs_lim = torch.asarray([-0.2 * torch.pi, +0.2 * torch.pi])
        
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
        options.reconstructor_options.allow_nondeterministic_algorithms = False
        
        task = PtychographyTask(options)
        task.run()
        
        recon = task.get_data_to_cpu('object', as_numpy=True)[0]
        return recon

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate-gold', action='store_true')
    args = parser.parse_args()

    tester = Tester2DPtychoObjectHardLimitsMagnitudePhase()
    tester.setup_method(name="", generate_data=False, generate_gold=args.generate_gold, debug=True)
    tester.test_2d_ptycho_object_hard_limits_magnitude_phase()
