"""Likelihood-domain checks for fitted detector responses."""

import numpy as np
import pytest
import torch

import ptychi.api as api
from ptychi.forward_models import PtychographyGaussianNoiseModel, PtychographyPoissonNoiseModel
from ptychi.metrics import MSELossOfSqrt
from test_detector_response_refinement import make_task, restore_defaults as restore_defaults


@pytest.mark.parametrize("noise", ["gaussian", "poisson"])
@pytest.mark.parametrize("optimizer", [api.Optimizers.SGD, api.Optimizers.LBFGS])
@pytest.mark.parametrize("dtype", [api.Dtypes.FLOAT32, api.Dtypes.FLOAT64])
def test_offset_fit_preserves_positive_predictions(noise, optimizer, dtype):
    task = make_task(noise=noise, dtype=dtype, response_options=api.DetectorResponseOptions(
        optimizable=True, optimizer=optimizer,
    ))
    rec = task.reconstructor
    indices = torch.arange(5)
    with torch.no_grad():
        initial = rec.forward_model(indices).clone()
        task.dataset.patterns.copy_(0.5 * initial)
        unconstrained = initial - 0.5 * initial.mean(0)
        assert (unconstrained < 0).any()
        rec.update_detector_response()
        prediction = rec.forward_model(indices)
        assert (prediction > 0).all()
        assert (task.detector_response.offset < 0).any()
        assert torch.isfinite(rec.noise_model.nll(prediction, task.dataset.patterns))
        gradient = rec.noise_model.backward_to_psi_far(
            prediction, task.dataset.patterns, rec.forward_model.intermediate_variables.psi_far,
            forward_model=rec.forward_model,
        )
        assert torch.isfinite(gradient).all()
    task.run(n_epochs=2)
    assert torch.isfinite(task.get_data("detector_response")).all()


@pytest.mark.parametrize("optimizer", [api.Optimizers.SGD, api.Optimizers.LBFGS])
def test_feasible_negative_offsets_are_preserved(optimizer):
    task = make_task(response_options=api.DetectorResponseOptions(
        optimizable=True, optimizer=optimizer,
    ))
    rec = task.reconstructor
    with torch.no_grad():
        intensity = rec.forward_model(torch.arange(5)).clone()
        offset = -0.25 * intensity.amin(0)
        task.dataset.patterns.copy_(intensity + offset)
        rec.update_detector_response()
    torch.testing.assert_close(task.detector_response.offset, offset, atol=2e-5, rtol=1e-4)


def test_feasibility_uses_final_normalized_gain_and_constrained_observations():
    mask = np.ones((3, 5), dtype=bool)
    mask[0, 0] = False
    task = make_task(mask=mask, response_options=api.DetectorResponseOptions(
        optimizable=True, optimize_gain=True,
    ))
    rec, response = task.reconstructor, task.detector_response
    rec.options.exclude_measured_pixels_below = 0
    with torch.no_grad():
        intensity = rec.forward_model(torch.arange(5)).clone()
        # Gain/probe normalization must preserve this feasible negative offset.
        offset = -1.5 * intensity.amin(0)
        task.dataset.patterns.copy_(2 * intensity + offset)
        task.dataset.patterns[:, 0, 0] = torch.nan
        task.dataset.patterns[:, 0, 1] = -1
        minimum_position = intensity[:, 1, 1].argmin()
        task.dataset.patterns[minimum_position, 1, 1] = -1
        observed = task.dataset.patterns > 0
        observed[:, 0, 0] = False
        rec.update_detector_response()
        prediction = rec.forward_model(torch.arange(5))
        assert (prediction[observed] > 0).all()
        torch.testing.assert_close(response.gain[response.valid_pixel_mask].mean(), torch.tensor(1.0))
        assert response.offset[0, 0] == 0
        assert response.offset[0, 1] == 0
        # The smallest excluded observation must not set the offset bound.
        signal = prediction - response.offset
        minimum = torch.where(observed, signal, torch.inf).amin(0)
        expected = torch.maximum(offset, -minimum)
        torch.testing.assert_close(response.offset[observed.any(0)], expected[observed.any(0)],
                                   atol=1e-12, rtol=1e-10)


def test_infeasible_gain_only_fit_is_rejected_without_changing_response():
    task = make_task(response_options=api.DetectorResponseOptions(
        optimizable=True, optimize_gain=True, optimize_offset=False,
    ))
    rec, response = task.reconstructor, task.detector_response
    with torch.no_grad():
        intensity = rec.forward_model(torch.arange(5)).clone()
        response.gain.fill_(2)
        response.offset.copy_(-1.5 * intensity.amin(0))
        # Positive measurements with little variation fit a gain too small
        # to support the fixed negative offset at the dimmest observation.
        task.dataset.patterns.copy_(0.1 * intensity)
        before = response.data.clone()
        probe_before = task.probe.data.clone()
        with pytest.raises(ValueError, match="fixed offset"):
            rec.update_detector_response()
    torch.testing.assert_close(response.data, before, rtol=0, atol=0)
    torch.testing.assert_close(task.probe.data, probe_before, rtol=0, atol=0)


@pytest.mark.parametrize("noise_class", [PtychographyGaussianNoiseModel, PtychographyPoissonNoiseModel])
@pytest.mark.parametrize("invalid", [-0.1, float("nan"), float("inf")])
def test_likelihood_rejects_invalid_predictions(noise_class, invalid):
    noise = noise_class()
    prediction = torch.tensor([[[1.0, invalid]]])
    measured = torch.ones_like(prediction)
    with pytest.raises(ValueError, match="Predicted intensities"):
        noise.nll(prediction, measured)
    with pytest.raises(ValueError, match="Predicted intensities"):
        noise.backward_to_intensity(prediction, measured)
    noise.valid_pixel_mask = torch.tensor([[True, False]])
    assert torch.isfinite(noise.backward_to_intensity(prediction, measured)).all()
    noise.valid_pixel_mask = None
    noise.exclude_measured_pixels_below = 0
    measured[0, 0, 1] = 0
    assert torch.isfinite(noise.backward_to_intensity(prediction, measured)).all()


@pytest.mark.parametrize("invalid", [-0.1, float("nan"), float("inf")])
def test_amplitude_loss_rejects_invalid_predictions(invalid):
    with pytest.raises(ValueError, match="Predicted intensities"):
        MSELossOfSqrt()(torch.tensor([invalid]), torch.ones(1))


def test_later_wavefield_change_fails_before_likelihood_gradient():
    task = make_task(response_options=api.DetectorResponseOptions(optimizable=True))
    rec = task.reconstructor
    with torch.no_grad():
        intensity = rec.forward_model(torch.arange(5)).clone()
        task.dataset.patterns.copy_(intensity - 0.25 * intensity.amin(0))
        rec.update_detector_response()
        task.probe.set_data(task.probe.data * 0.01)
        prediction = rec.forward_model(torch.arange(5))
        with pytest.raises(ValueError, match="Predicted intensities"):
            rec.run_reciprocal_space_step(prediction, task.dataset.patterns, torch.arange(5))


def test_response_bound_uses_minimum_across_ranks(monkeypatch):
    task = make_task(response_options=api.DetectorResponseOptions(optimizable=True))
    rec = task.reconstructor
    with torch.no_grad():
        intensity = rec.forward_model(torch.arange(5)).clone()
        task.dataset.patterns.copy_(0.5 * intensity)
    remote_minimum = intensity.amin(0) * 0.01
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)

    def reduce(tensor, op=None):
        if op == torch.distributed.ReduceOp.MIN:
            tensor.copy_(torch.minimum(tensor, remote_minimum))

    monkeypatch.setattr(torch.distributed, "all_reduce", reduce)
    rec.update_detector_response()
    remote_prediction = task.detector_response.gain * remote_minimum + task.detector_response.offset
    assert (remote_prediction > 0).all()
    torch.testing.assert_close(task.detector_response.offset, -remote_minimum, atol=1e-12, rtol=1e-10)
