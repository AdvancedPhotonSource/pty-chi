import dataclasses
from types import SimpleNamespace

import torch

from ptychi.api.options.base import ReconstructorOptions
from ptychi.api.options.data import PtychographyDataOptions
from ptychi.forward_models import PtychographyGaussianNoiseModel, PtychographyPoissonNoiseModel
from ptychi.reconstructors.base import (
    AnalyticalIterativePtychographyReconstructor,
    IterativePtychographyReconstructor,
)
from ptychi.reconstructors.bh import BHReconstructor


class _DummyAnalyticalPtychographyReconstructor:
    get_constrained_pixel_mask = IterativePtychographyReconstructor.get_constrained_pixel_mask
    replace_propagated_exit_wave_magnitude = (
        AnalyticalIterativePtychographyReconstructor.replace_propagated_exit_wave_magnitude
    )

class _IdentityPropagator:
    def propagate_backward(self, tensor):
        return tensor


class _DummyBHReconstructor:
    gradientF = BHReconstructor.gradientF

    def __init__(self, constrained_pixel_mask):
        self.eps = 1e-8
        self.current_constrained_pixel_mask = constrained_pixel_mask[:, None]
        self.forward_model = SimpleNamespace(free_space_propagator=_IdentityPropagator())


def _compute_gradient(
    exclude_measured_pixels_below=None,
    valid_pixel_mask=None,
):
    model = PtychographyGaussianNoiseModel(
        valid_pixel_mask=valid_pixel_mask,
        exclude_measured_pixels_below=exclude_measured_pixels_below,
    )
    y_pred = torch.full((1, 2, 2), 4.0)
    y_true = torch.tensor([[[0.0, 1.0], [2.0, 3.0]]])
    psi_far = torch.ones((1, 1, 2, 2), dtype=torch.complex64)
    return model.backward_to_psi_far(y_pred, y_true, psi_far).squeeze(1)


def _compute_poisson_gradient(
    exclude_measured_pixels_below=None,
    valid_pixel_mask=None,
):
    model = PtychographyPoissonNoiseModel(
        valid_pixel_mask=valid_pixel_mask,
        exclude_measured_pixels_below=exclude_measured_pixels_below,
    )
    y_pred = torch.full((1, 2, 2), 4.0)
    y_true = torch.tensor([[[0.0, 1.0], [2.0, 3.0]]])
    psi_far = torch.ones((1, 1, 2, 2), dtype=torch.complex64)
    return model.backward_to_psi_far(y_pred, y_true, psi_far).squeeze(1)


def test_exclude_measured_pixels_below_lives_on_reconstructor_options():
    reconstructor_fields = {field.name for field in dataclasses.fields(ReconstructorOptions)}
    data_fields = {field.name for field in dataclasses.fields(PtychographyDataOptions)}

    assert "exclude_measured_pixels_below" in reconstructor_fields
    assert "exclude_measured_pixels_below" not in data_fields


def test_exclude_measured_pixels_below_none_only_uses_valid_pixel_mask():
    valid_pixel_mask = torch.tensor([[True, False], [True, True]])

    gradient = _compute_gradient(
        exclude_measured_pixels_below=None,
        valid_pixel_mask=valid_pixel_mask,
    )

    assert gradient[0, 0, 0].abs() > 0
    assert gradient[0, 0, 1].abs() == 0


def test_exclude_measured_pixels_below_zero_masks_only_zero_measurements():
    baseline = _compute_gradient(exclude_measured_pixels_below=None)
    gradient = _compute_gradient(exclude_measured_pixels_below=0.0)

    measured_zero_mask = torch.tensor([[[True, False], [False, False]]])
    assert torch.all(gradient[measured_zero_mask] == 0)
    torch.testing.assert_close(
        gradient[~measured_zero_mask],
        baseline[~measured_zero_mask],
    )


def test_exclude_measured_pixels_below_positive_threshold_masks_low_measurements():
    baseline = _compute_gradient(exclude_measured_pixels_below=None)
    gradient = _compute_gradient(exclude_measured_pixels_below=1.0)

    low_measurement_mask = torch.tensor([[[True, True], [False, False]]])
    assert torch.all(gradient[low_measurement_mask] == 0)
    torch.testing.assert_close(
        gradient[~low_measurement_mask],
        baseline[~low_measurement_mask],
    )


def test_constrained_pixel_mask_combines_valid_mask_and_threshold():
    reconstructor = _DummyAnalyticalPtychographyReconstructor()
    reconstructor.dataset = SimpleNamespace(valid_pixel_mask=torch.tensor([[True, True], [False, True]]))
    reconstructor.options = SimpleNamespace(exclude_measured_pixels_below=1.0)
    y_true = torch.tensor([[[0.0, 2.0], [3.0, 1.0]]])

    constrained_pixel_mask = reconstructor.get_constrained_pixel_mask(y_true)

    expected = torch.tensor([[[False, True], [False, False]]])
    assert torch.equal(constrained_pixel_mask, expected)


def test_projection_helper_leaves_excluded_pixels_unconstrained():
    reconstructor = _DummyAnalyticalPtychographyReconstructor()
    reconstructor.dataset = SimpleNamespace(valid_pixel_mask=torch.tensor([[True, True], [True, True]]))
    reconstructor.options = SimpleNamespace(exclude_measured_pixels_below=1.0)
    psi_far = torch.full((1, 1, 2, 2), 2 + 0j, dtype=torch.complex64)
    y_true = torch.tensor([[[0.0, 4.0], [9.0, 1.0]]])

    constrained_pixel_mask = reconstructor.get_constrained_pixel_mask(y_true)
    constrained_projection = reconstructor.replace_propagated_exit_wave_magnitude(
        psi_far,
        y_true,
        constrained_pixel_mask=constrained_pixel_mask,
    )
    fully_constrained_projection = reconstructor.replace_propagated_exit_wave_magnitude(
        psi_far,
        y_true,
    )

    expanded_mask = constrained_pixel_mask[:, None]
    torch.testing.assert_close(
        constrained_projection[expanded_mask],
        fully_constrained_projection[expanded_mask],
    )
    torch.testing.assert_close(
        constrained_projection[~expanded_mask],
        psi_far[~expanded_mask],
    )


def test_poisson_noise_model_respects_exclude_measured_pixels_below():
    baseline = _compute_poisson_gradient(exclude_measured_pixels_below=None)
    gradient = _compute_poisson_gradient(exclude_measured_pixels_below=1.0)

    low_measurement_mask = torch.tensor([[[True, True], [False, False]]])
    assert torch.all(gradient[low_measurement_mask] == 0)
    torch.testing.assert_close(
        gradient[~low_measurement_mask],
        baseline[~low_measurement_mask],
    )


def test_bh_gradient_masks_excluded_detector_pixels():
    constrained_pixel_mask = torch.tensor([[[True, False], [False, True]]])
    reconstructor = _DummyBHReconstructor(constrained_pixel_mask)
    psi_far = torch.full((1, 1, 2, 2), 2 + 0j, dtype=torch.complex64)
    d = torch.ones((1, 1, 2, 2))

    gradient = reconstructor.gradientF(psi_far, d)

    expected = torch.tensor([[[[2 + 0j, 0 + 0j], [0 + 0j, 2 + 0j]]]], dtype=torch.complex64)
    torch.testing.assert_close(gradient, expected)
