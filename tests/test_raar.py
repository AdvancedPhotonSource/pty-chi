import torch

import ptychi.api as api
import ptychi.maps as maps
from ptychi.api.task import PtychographyTask
from ptychi.api.options.raar import RAARReconstructorOptions
from ptychi.reconstructors import RAARReconstructor


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


def test_raar_options_select_raar_reconstructor():
    options = api.RAAROptions()

    assert options.reconstructor_options.get_reconstructor_type() is api.Reconstructors.RAAR
    assert maps.get_reconstructor_by_enum(api.Reconstructors.RAAR) is RAARReconstructor
    assert options.reconstructor_options.beta == 0.75


def test_raar_exit_wave_update_matches_paper_equation():
    reconstructor = object.__new__(RAARReconstructor)
    reconstructor.options = RAARReconstructorOptions(beta=0.75)
    reconstructor.psi = torch.tensor([1 + 1j, 2 - 1j], dtype=torch.complex64).reshape(2, 1, 1, 1)
    previous_exit_wave = reconstructor.psi.clone()
    pa_exit_wave = previous_exit_wave + (0.5 - 0.25j)
    q_exit_wave = previous_exit_wave * (0.8 + 0.1j)
    q_pa_exit_wave = pa_exit_wave * (0.9 - 0.2j)
    reconstructor.apply_data_projection = lambda exit_wave, y_true: pa_exit_wave
    reconstructor.synthesize_exit_wave_chunk = (
        lambda projected_object, start_pt, end_pt: projected_object[start_pt:end_pt]
    )

    error_squared = reconstructor.apply_raar_exit_wave_update(
        y_true=torch.zeros((2, 1, 1)),
        q_object=q_exit_wave,
        pa_object=q_pa_exit_wave,
        start_pts=[0],
        end_pts=[2],
    )

    beta = reconstructor.options.beta
    expected = (
        2 * beta * q_pa_exit_wave
        + (1 - 2 * beta) * pa_exit_wave
        + beta * (q_exit_wave - previous_exit_wave)
    )
    assert torch.allclose(reconstructor.psi, expected)
    assert torch.allclose(error_squared, ((expected - previous_exit_wave).abs() ** 2).sum())


def test_raar_updates_object_and_probe():
    previous_device = torch.get_default_device()
    torch.set_default_device("cpu")
    try:
        diffraction_data, true_probe, positions = make_synthetic_data()
        initial_object = torch.ones((1, 10, 10), dtype=torch.complex64)
        initial_probe = (true_probe * torch.exp(0.2j * true_probe.real))[None, None]

        options = api.RAAROptions()
        options.reconstructor_options.default_device = api.Devices.CPU
        options.reconstructor_options.num_epochs = 1
        options.reconstructor_options.chunk_length = 2
        options.reconstructor_options.allow_nondeterministic_algorithms = False
        options.object_options.remove_object_probe_ambiguity.enabled = False
        options.probe_position_options.optimizable = False

        task = PtychographyTask(
            options,
            diffraction_data=diffraction_data,
            object_data=initial_object,
            probe_data=initial_probe,
            probe_position_x_px=positions[:, 1],
            probe_position_y_px=positions[:, 0],
        )
        object_before = task.object.data.detach().clone()
        probe_before = task.probe.data.detach().clone()

        task.run()

        assert not torch.allclose(task.object.data, object_before)
        assert not torch.allclose(task.probe.data, probe_before)
        assert torch.isfinite(task.object.data).all()
        assert torch.isfinite(task.probe.data).all()
        assert torch.isfinite(task.reconstructor.psi).all()
    finally:
        torch.set_default_device(previous_device)
