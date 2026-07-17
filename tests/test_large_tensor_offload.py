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
        diffraction_data = data
        options.data_options.save_data_on_device = True

        object_data = object_guess
        options.object_options.pixel_size_m = 1e-6
        options.object_options.optimizable = True

        probe_data = probe_guess
        options.probe_options.optimizable = False

        probe_position_x_px = positions
        probe_position_y_px = positions
        options.probe_position_options.optimizable = False

        options.reconstructor_options.batch_size = 2
        options.reconstructor_options.num_epochs = 1

        task = PtychographyTask(
            options,
            diffraction_data=diffraction_data,
            object_data=object_data,
            probe_data=probe_data,
            probe_position_x_px=probe_position_x_px,
            probe_position_y_px=probe_position_y_px,
        )
        fm = task.reconstructor.forward_model
        indices = torch.arange(2, device="cuda", dtype=torch.long)
        fm.forward(indices)
        task.object.preconditioner = torch.ones(task.object.shape, device="cuda")
        task.probe.update_buffer = torch.ones(task.probe.shape, device="cuda")
        object_optimizer = task.object.optimizer
        object_parameter = object_optimizer.param_groups[0]["params"][0]
        object_optimizer.state[object_parameter]["test_buffer"] = torch.ones_like(
            object_parameter
        )
        reconstructor_buffers = task.reconstructor.reconstructor_buffers

        assert task.dataset.patterns.device.type == "cuda"
        assert task.reconstructor.parameter_group.object.tensor.data.device.type == "cuda"
        assert task.reconstructor.parameter_group.probe.tensor.data.device.type == "cuda"
        assert task.probe_positions.data.device.type == "cuda"
        assert task.opr_mode_weights.data.device.type == "cuda"
        assert task.dataset.valid_pixel_mask.device.type == "cuda"
        assert task.object.preconditioner.device.type == "cuda"
        assert task.probe.update_buffer.device.type == "cuda"
        assert object_optimizer.state[object_parameter]["test_buffer"].device.type == "cuda"
        assert reconstructor_buffers.alpha_probe_all_pos.device.type == "cuda"
        assert fm.intermediate_variables.obj_patches.device.type == "cuda"

        task.set_large_tensor_device("cpu")

        assert task.dataset.patterns.device.type == "cpu"
        assert not task.dataset.save_data_on_device
        assert task.reconstructor.parameter_group.object.tensor.data.device.type == "cpu"
        assert task.reconstructor.parameter_group.probe.tensor.data.device.type == "cpu"
        assert task.probe_positions.data.device.type == "cpu"
        assert task.opr_mode_weights.data.device.type == "cpu"
        assert task.dataset.valid_pixel_mask.device.type == "cpu"
        assert task.object.preconditioner.device.type == "cpu"
        assert task.probe.update_buffer.device.type == "cpu"
        assert object_optimizer.state[object_parameter]["test_buffer"].device.type == "cpu"
        assert reconstructor_buffers.alpha_probe_all_pos.device.type == "cpu"
        assert fm.intermediate_variables.obj_patches.device.type == "cpu"

        task.set_large_tensor_device()

        assert task.dataset.patterns.device.type == "cuda"
        assert task.dataset.save_data_on_device
        assert task.reconstructor.parameter_group.object.tensor.data.device.type == "cuda"
        assert task.reconstructor.parameter_group.probe.tensor.data.device.type == "cuda"
        assert task.probe_positions.data.device.type == "cuda"
        assert task.opr_mode_weights.data.device.type == "cuda"
        assert task.dataset.valid_pixel_mask.device.type == "cuda"
        assert task.object.preconditioner.device.type == "cuda"
        assert task.probe.update_buffer.device.type == "cuda"
        assert object_optimizer.state[object_parameter]["test_buffer"].device.type == "cuda"
        assert reconstructor_buffers.alpha_probe_all_pos.device.type == "cuda"
        assert fm.intermediate_variables.obj_patches.device.type == "cuda"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate-gold', action='store_true')
    args = parser.parse_args()

    tester = TestLargeTensorOffload()
    tester.setup_method(name="", generate_data=False, generate_gold=args.generate_gold, debug=True)
    tester.test_set_large_tensor_device_moves_buffers()
