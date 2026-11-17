"""Inline examples for reciprocal projection, PIE weights, and DM updates."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import ptychi.api as api
from ptychi.reconstructors.dm import DMReconstructor
from ptychi.reconstructors.pie import PIEReconstructor, EPIEReconstructor, RPIEReconstructor
from test_utils import make_task

pytestmark = pytest.mark.usefixtures("numerical_cpu")


@pytest.fixture
def projection_reconstructor():
    recon = make_task(api.EPIEOptions()).reconstructor
    recon.dataset = SimpleNamespace(valid_pixel_mask=torch.ones((1, 2), dtype=torch.bool))
    return recon


@pytest.mark.parametrize("masked", [False, True])
def test_reciprocal_magnitude_replacement_preserves_mode_ratio_and_phase(masked, projection_reconstructor):
    # Two incoherent modes have total intensity 25 at each pixel.
    psi = torch.tensor([[[[3j, 4]], [[4, -3j]]]])
    target = torch.tensor([[[100.0, 6.25]]])
    mask = torch.tensor([[[True, False]]]) if masked else None
    projected = projection_reconstructor.replace_propagated_exit_wave_magnitude(
        psi, target, mask
    )
    expected = torch.tensor([[[[6j, 2]], [[8, -1.5j]]]])
    if masked:
        expected[..., 1] = psi[..., 1]
    torch.testing.assert_close(projected, expected, atol=2e-7, rtol=1e-7)
    torch.testing.assert_close(psi, torch.tensor([[[[3j, 4]], [[4, -3j]]]]))


def test_reciprocal_projection_zero_field_remains_finite(projection_reconstructor):
    psi = torch.zeros((2, 2, 1, 2), dtype=torch.complex128)
    actual = projection_reconstructor.replace_propagated_exit_wave_magnitude(
        psi, torch.tensor([[[0.0, 4.0]], [[9.0, 0.0]]])
    )
    torch.testing.assert_close(actual, psi)


@pytest.mark.parametrize(
    "cls,expected",
    [
        (PIEReconstructor, [[1 / 24, -4j / 36], [9 / 56, 0]]),
        (EPIEReconstructor, [[0.05, -0.1j], [0.15, 0]]),
        (RPIEReconstructor, [[1 / 5.5, -2j / 7], [3 / 9.5, 0]]),
    ],
)
def test_pie_object_step_weights_use_total_mode_intensity(cls, expected):
    recon = cls.__new__(cls)
    recon.parameter_group = SimpleNamespace(
        object=SimpleNamespace(options=SimpleNamespace(alpha=0.5))
    )
    probe = torch.tensor([[[[1, 2j]], [[3, 0]]]])
    torch.testing.assert_close(
        recon.calculate_object_step_weight(probe), torch.tensor(expected).reshape(1, 2, 1, 2)
    )


@pytest.mark.parametrize(
    "cls,expected",
    [
        (PIEReconstructor, [[1 / 12, -1j / 6], [1 / 48, -1j / 24]]),
        (EPIEReconstructor, [[1 / 8, -1j / 4], [1 / 16, -1j / 8]]),
        (RPIEReconstructor, [[1 / 2.5, -1j / 2], [1 / 5, -1j / 4]]),
    ],
)
def test_pie_probe_step_weights_normalize_each_scan_independently(cls, expected):
    recon = cls.__new__(cls)
    recon.parameter_group = SimpleNamespace(
        probe=SimpleNamespace(options=SimpleNamespace(alpha=0.5))
    )
    patches = torch.tensor([[[[1, 2j]], [[99, 99]]], [[[2, 4j]], [[99, 99]]]])
    actual = recon.calculate_probe_step_weight(patches)
    shape = (2, 1, 2) if cls is PIEReconstructor else (2, 1, 1, 2)
    torch.testing.assert_close(actual, torch.tensor(expected).reshape(shape))


@pytest.mark.parametrize("momentum,expected", [(False, 1.2), (True, 1.3)])
def test_pie_applies_negative_gradient_and_batch_mean(momentum, expected):
    options = api.RPIEOptions()
    options.object_options.optimizer = api.Optimizers.SGD
    options.object_options.step_size = 0.1
    options.probe_options.optimizer = api.Optimizers.SGD
    options.probe_options.step_size = 0.1
    if momentum:
        options.object_options.optimizer_params = {"momentum": 0.5, "nesterov": True}
    task = make_task(options)
    recon = task.reconstructor
    before_probe = task.probe.data.clone()
    task.probe.initialize_grad()
    with torch.no_grad():
        recon.apply_updates(
            torch.full((1, 6, 6), 2 + 0j),
            torch.tensor([[[[1 + 0j] * 2] * 2], [[[3 + 0j] * 2] * 2]]),
            None,
        )
    torch.testing.assert_close(task.object.data, torch.full((1, 6, 6), expected + 0j))
    torch.testing.assert_close(task.probe.data, before_probe + 0.2)


@pytest.mark.parametrize("relaxation", [0.5, 1.0])
def test_dm_exit_wave_reflection_and_error_accumulation(relaxation, projection_reconstructor, monkeypatch):
    recon = DMReconstructor.__new__(DMReconstructor)
    recon.options = SimpleNamespace(
        exit_wave_update_relaxation=relaxation, exclude_measured_pixels_below=None
    )
    recon.dataset = projection_reconstructor.dataset
    recon.psi = torch.tensor([[[[99 + 0j, 99]]], [[[1 + 0j, 1j]]], [[[77 + 0j, 77]]]])
    new_psi = torch.tensor([[[[2 + 0j, 2j]]]])
    recon.calculate_exit_wave_chunk = Mock(return_value=(new_psi, torch.tensor([5.0])))
    recon.forward_model = projection_reconstructor.forward_model
    for method in ("propagate_forward", "propagate_backward"):
        monkeypatch.setattr(recon.forward_model.free_space_propagator, method, lambda x: x)
    _, error, returned = recon.apply_dm_update_to_exit_wave_chunk(
        1, 2, torch.tensor([[[0.0, 0.0]], [[16.0, 16.0]], [[0.0, 0.0]]]), torch.tensor(7.0)
    )
    expected = torch.tensor(
        [[[[99 + 0j, 99]]], [[[1 + 2 * relaxation, (1 + 2 * relaxation) * 1j]]], [[[77 + 0j, 77]]]]
    )
    torch.testing.assert_close(recon.psi, expected, atol=2e-7, rtol=1e-7)
    torch.testing.assert_close(error, torch.tensor(7 + 8 * relaxation**2), atol=1e-6, rtol=1e-7)
    torch.testing.assert_close(returned, new_psi)


def test_dm_probe_projection_damping_and_inertia():
    recon = DMReconstructor.__new__(DMReconstructor)
    recon.parameter_group = SimpleNamespace(
        probe=SimpleNamespace(
            data=torch.tensor([[[[2 + 2j, 4]]]]), options=SimpleNamespace(inertia=0.25)
        )
    )
    actual = recon.calculate_probe_update(
        torch.tensor([[[[3 + 4j, 8]]]]), torch.tensor([[[[3.0, 4.0]]]])
    )
    expected = torch.tensor(
        [[[[0.5 + 0.5j + 0.75 * (3 + 4j) / (9.04**0.5), 1 + 6 / (16.04**0.5)]]]]
    )
    torch.testing.assert_close(actual, expected)


def test_dm_probe_terms_sum_complex_overlap():
    recon = DMReconstructor.__new__(DMReconstructor)
    recon.psi = torch.tensor([[[[1 + 2j, 3]]], [[[4j, 5]]]])
    recon.adjoint_shift_probe_update_direction = lambda indices, x: x
    numerator = torch.zeros((1, 1, 2), dtype=torch.complex128)
    denominator = torch.zeros((1, 1, 2))
    recon.add_to_probe_update_terms(
        numerator, denominator, torch.tensor([[[[1j, 2]]], [[[2j, 3]]]]), 0, 2
    )
    torch.testing.assert_close(numerator, torch.tensor([[[10 - 1j, 21]]]))
    torch.testing.assert_close(denominator, torch.tensor([[[5.0, 13.0]]]))


@pytest.mark.parametrize(
    "operator,expected",
    [
        (api.ShiftOperators.PATCH, [[2 + 1j, 30], [7, 4]]),
        (api.ShiftOperators.FULL_OBJECT, [[2 + 1j, 1.5], [7, 4]]),
    ],
)
def test_raar_object_projection_illumination_floor_and_unobserved_pixels(operator, expected):
    from ptychi.reconstructors.raar import RAARReconstructor

    recon = RAARReconstructor.__new__(RAARReconstructor)
    recon.forward_model = SimpleNamespace(
        resolved_shift_operator=operator,
        get_probe_grid_object=lambda: torch.tensor([[[9 + 0j, 8], [7, 6]]]),
    )
    actual = recon.finish_object_projection(
        torch.tensor([[8 + 4j, 0.3], [0, 4]]), torch.tensor([[4.0, 0.01], [0.0, 1.0]])
    )
    # FULL_OBJECT floors the positive weight .01 at .05*4=.2.
    torch.testing.assert_close(actual, torch.tensor(expected))
