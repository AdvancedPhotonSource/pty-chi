import pytest
import argparse

import torch

import ptychi.api as api
from ptychi.api.task import PtychographyTask

import test_utils as tutils


class TestLargeTensorOffload(tutils.BaseTester):

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required for offload test")
    def test_set_large_tensor_device_moves_buffers(self):
        self.setup_ptychi(cpu_only=False)

        n_positions = 4
        detector_shape = (8, 8)
        data = torch.rand((n_positions, *detector_shape), device="cuda")
        object_guess = torch.ones((1, 16, 16), dtype=torch.complex64, device="cuda")
        probe_guess = torch.randn((1, 1, *detector_shape), dtype=torch.complex64, device="cuda")
        positions = torch.linspace(-2, 2, steps=n_positions).cpu()

        options = api.LSQMLOptions()
        options.data_options.data = data
        options.data_options.save_data_on_device = True

        options.object_options.initial_guess = object_guess
        options.object_options.pixel_size_m = 1e-6
        options.object_options.optimizable = False

        options.probe_options.initial_guess = probe_guess
        options.probe_options.optimizable = False

        options.probe_position_options.position_x_px = positions
        options.probe_position_options.position_y_px = positions
        options.probe_position_options.optimizable = False

        options.reconstructor_options.batch_size = 2
        options.reconstructor_options.num_epochs = 1

        task = PtychographyTask(options)
        fm = task.reconstructor.forward_model
        indices = torch.arange(2, device="cuda", dtype=torch.long)
        fm.forward(indices)

        assert task.dataset.patterns.device.type == "cuda"
        assert task.reconstructor.parameter_group.object.tensor.data.device.type == "cuda"
        assert task.reconstructor.parameter_group.probe.tensor.data.device.type == "cuda"
        assert fm.intermediate_variables.obj_patches.device.type == "cuda"

        task.set_large_tensor_device("cpu")

        assert task.dataset.patterns.device.type == "cpu"
        assert not task.dataset.save_data_on_device
        assert task.reconstructor.parameter_group.object.tensor.data.device.type == "cpu"
        assert task.reconstructor.parameter_group.probe.tensor.data.device.type == "cpu"
        assert fm.intermediate_variables.obj_patches.device.type == "cpu"

        task.set_large_tensor_device()

        assert task.dataset.patterns.device.type == "cuda"
        assert task.dataset.save_data_on_device
        assert task.reconstructor.parameter_group.object.tensor.data.device.type == "cuda"
        assert task.reconstructor.parameter_group.probe.tensor.data.device.type == "cuda"
        assert fm.intermediate_variables.obj_patches.device.type == "cuda"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate-gold', action='store_true')
    args = parser.parse_args()

    tester = TestLargeTensorOffload()
    tester.setup_method(name="", generate_data=False, generate_gold=args.generate_gold, debug=True)
    tester.test_set_large_tensor_device_moves_buffers()
