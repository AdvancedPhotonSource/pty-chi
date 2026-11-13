import pytest
import torch

import ptychi.api as api
from ptychi.api.task import PtychographyTask

pytestmark = pytest.mark.usefixtures("numerical_cpu")


def make_synthetic_data():
    yy, xx = torch.meshgrid(
        torch.arange(10, dtype=torch.float32),
        torch.arange(10, dtype=torch.float32),
        indexing="ij",
    )
    true_object = (1 + 0.15 * torch.sin(xx / 2)) * torch.exp(0.35j * torch.cos(yy / 3))
    probe_yy, probe_xx = torch.meshgrid(
        torch.arange(4, dtype=torch.float32) - 1.5,
        torch.arange(4, dtype=torch.float32) - 1.5,
        indexing="ij",
    )
    true_probe = torch.exp(-(probe_xx**2 + probe_yy**2) / 3).to(torch.complex64)
    positions = torch.tensor([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])

    exit_waves = []
    position_origin = torch.tensor([5.5, 5.5])
    for position in positions:
        center = position + position_origin
        start = (center - 1.5).round().int()
        object_patch = true_object[
            start[0] : start[0] + 4,
            start[1] : start[1] + 4,
        ]
        exit_waves.append(object_patch * true_probe)
    exit_waves = torch.stack(exit_waves)
    diffraction_data = torch.fft.fft2(exit_waves).abs() ** 2
    return diffraction_data, true_probe, positions


def test_raar_e2e():
    previous_device = torch.get_default_device()
    torch.set_default_device("cpu")
    try:
        diffraction_data, true_probe, positions = make_synthetic_data()
        initial_object = torch.ones((1, 10, 10), dtype=torch.complex64)
        shared_probe = true_probe * torch.exp(0.2j * true_probe.real)
        eigenmode = torch.roll(true_probe, 1, 0) - torch.roll(true_probe, -1, 1)
        eigenmode = eigenmode * (1 + 0.3j)
        initial_probe = torch.stack([shared_probe, eigenmode])[:, None]
        initial_weights = torch.tensor([[1.0, 0.04], [1.0, -0.03], [1.0, 0.02], [1.0, -0.01]])

        options = api.RAAROptions()
        options.reconstructor_options.default_device = api.Devices.CPU
        options.reconstructor_options.num_epochs = 1
        options.reconstructor_options.chunk_length = 1
        options.reconstructor_options.allow_nondeterministic_algorithms = False
        options.object_options.remove_object_probe_ambiguity.enabled = False
        options.probe_options.orthogonalize_opr_modes.enabled = False
        options.probe_position_options.optimizable = False
        options.opr_mode_weight_options.optimizable = True
        options.opr_mode_weight_options.update_relaxation = 0.1

        task = PtychographyTask(
            options,
            diffraction_data=diffraction_data,
            object_data=initial_object,
            probe_data=initial_probe,
            probe_position_x_px=positions[:, 1],
            probe_position_y_px=positions[:, 0],
            opr_mode_weights_data=initial_weights,
        )
        object_before = task.object.data.detach().clone()
        shared_probe_before = task.probe.data[0].detach().clone()
        eigenmode_before = task.probe.data[1].detach().clone()
        weights_before = task.opr_mode_weights.data.detach().clone()

        task.run()

        assert not torch.allclose(task.object.data, object_before)
        assert torch.isfinite(task.object.data).all()
        assert torch.isfinite(task.reconstructor.psi).all()
        assert not torch.allclose(task.probe.data[0], shared_probe_before)
        assert not torch.allclose(task.probe.data[1], eigenmode_before)
        assert not torch.allclose(task.opr_mode_weights.data[:, 1], weights_before[:, 1])
        assert torch.isfinite(task.probe.data).all()
        assert torch.isfinite(task.opr_mode_weights.data).all()
    finally:
        torch.set_default_device(previous_device)
