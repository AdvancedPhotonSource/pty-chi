"""Regression checks for amplitude likelihoods and analytical loss selection."""

import pytest
import torch

import ptychi.api as api
from ptychi.forward_models import GaussianNoiseModel, MagnitudeMeanSquaredNoiseModel
from ptychi.metrics import MSELossOfSqrt
from ptychi.utils import get_constrained_pixel_mask
from test_utils import make_task

pytestmark = pytest.mark.usefixtures("numerical_cpu")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_magnitude_nll_preserves_loss_and_gradient(dtype, reduction):
    predicted = torch.tensor([0.0, 1e-12, 0.25, 4.0, 100.0], dtype=dtype, requires_grad=True)
    measured = torch.tensor([4.0, 0.0, 1.0, 9.0, 100.0], dtype=dtype)
    expected = MSELossOfSqrt(reduction=reduction)(predicted, measured)
    actual = MagnitudeMeanSquaredNoiseModel().nll(predicted, measured, reduction=reduction)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    expected_gradient, = torch.autograd.grad(expected.sum(), predicted)
    actual_gradient, = torch.autograd.grad(actual.sum(), predicted)
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)

    gaussian = GaussianNoiseModel(sigma=0.7).nll(predicted, measured, reduction=reduction)
    torch.testing.assert_close(gaussian, expected / 0.7**2 * 0.5, rtol=0, atol=0)


@pytest.mark.parametrize(
    "options_class", [api.EPIEOptions, api.RPIEOptions, api.MAGPIEOptions, api.DMOptions, api.RAAROptions]
)
def test_analytical_magnitude_losses_preserve_masked_mse(options_class):
    options = options_class()
    options.reconstructor_options.exclude_measured_pixels_below = 1.0
    recon = make_task(options).reconstructor
    recon.dataset.valid_pixel_mask[0, 1] = False
    measured = torch.tensor([[[0.0, 4.0], [9.0, 1.0]], [[4.0, 0.0], [1.0, 16.0]]])
    inputs = [torch.arange(2)]
    predicted = recon.forward_model(*inputs)
    mask = get_constrained_pixel_mask(recon.dataset.valid_pixel_mask, 1.0, measured)
    expected = MSELossOfSqrt()(predicted[mask], measured[mask])

    assert type(recon.noise_model) is MagnitudeMeanSquaredNoiseModel
    torch.testing.assert_close(recon.compute_optimizer_loss(inputs, measured), expected, rtol=0, atol=0)
    if options_class in (api.DMOptions, api.RAAROptions):
        # These algorithms retain their exit-wave update diagnostics.
        assert recon.loss_tracker.metric_function is None
    else:
        assert recon.loss_tracker.metric_function == recon.noise_model.nll
        recon.update_batch_loss(predicted, measured)
        assert recon.loss_tracker.epoch_loss == expected.detach().numpy()


def test_explicit_displayed_loss_overrides_magnitude_nll():
    options = api.EPIEOptions()
    options.reconstructor_options.displayed_loss_function = api.LossFunctions.MSE
    recon = make_task(options).reconstructor
    assert isinstance(recon.loss_tracker.metric_function, torch.nn.MSELoss)
    assert recon.loss_tracker.metric_function is recon.displayed_loss_function
