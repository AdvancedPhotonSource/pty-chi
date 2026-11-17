from functools import wraps

import numpy as np
import pytest
import torch

import ptychi.api as api
from ptychi.api.task import PtychographyTask
from ptychi.api.options.base import OptimizationPlan
from ptychi.data_structures.detector_response import DetectorResponse
from ptychi.forward_models import PtychographyGaussianNoiseModel, PtychographyPoissonNoiseModel
from ptychi.utils import get_default_complex_dtype, set_default_complex_dtype


@pytest.fixture(autouse=True)
def restore_defaults():
    dtype, device = torch.get_default_dtype(), torch.get_default_device()
    complex_dtype = get_default_complex_dtype()
    torch.set_default_device("cpu")
    yield
    torch.set_default_dtype(dtype)
    torch.set_default_device(device)
    set_default_complex_dtype(complex_dtype)


def make_task(options_class=api.LSQMLOptions, response_options=None, response=None,
              mask=None, fft_shift=False, blur=None, noise="gaussian", refine_object_probe=False, dtype=api.Dtypes.FLOAT64):
    options = options_class()
    rec = options.reconstructor_options
    rec.default_device = api.Devices.CPU
    rec.default_dtype = dtype
    rec.batch_size = 2
    rec.random_seed = 12
    rec.forward_model_options.diffraction_pattern_blur_sigma = blur
    if response_options is not None:
        rec.forward_model_options.detector_response_options = response_options
    if options_class is api.LSQMLOptions:
        rec.rescale_probe_intensity_in_first_epoch = False
        rec.noise_model = noise
    options.data_options.fft_shift = fft_shift
    options.object_options.pixel_size_m = 1
    options.object_options.optimizable = refine_object_probe or options_class is api.MAGPIEOptions
    options.probe_options.optimizable = refine_object_probe or options_class is api.MAGPIEOptions
    options.probe_position_options.optimizable = False
    rng = np.random.default_rng(28)
    probe_shape = (1, 1, 3, 5) if options_class is api.MAGPIEOptions else (1, 1, 6, 6)
    return PtychographyTask(
        options,
        diffraction_data=np.ones((5, 3, 5), dtype=np.float32),
        object_data=(0.4 + rng.random((1, 12, 12)) * 0.5).astype(np.complex64),
        probe_data=(rng.random(probe_shape) + 1j * rng.random(probe_shape)).astype(np.complex64),
        probe_position_x_px=np.array([-2, -1, 0, 1, 2]),
        probe_position_y_px=np.array([0, -1, 1, -2, 2]),
        detector_response_map=response,
        valid_pixel_mask=mask,
    )


def test_default_identity_and_nested_options_roundtrip():
    task = make_task()
    response = task.detector_response
    assert response is task.reconstructor.parameter_group.detector_response
    assert response is task.reconstructor.forward_model.detector_response
    assert response.is_identity and not response.optimizable
    assert response.optimizer is None
    assert not response.tensor.requires_grad
    assert task.get_data("detector_response").shape == (2, 3, 5)
    torch.testing.assert_close(response.gain, torch.ones((3, 5)))
    torch.testing.assert_close(response.offset, torch.zeros((3, 5)))
    restored = api.LSQMLOptions().load_from_dict(task.get_options_as_dict(), strict=True)
    assert isinstance(restored.reconstructor_options.forward_model_options.detector_response_options,
                      api.DetectorResponseOptions)
    assert not restored.reconstructor_options.forward_model_options.detector_response_options.optimizable


def test_response_coordinates_copy_and_offload():
    gain = np.arange(15).reshape(3, 5) + 1.0
    offset = -np.arange(15).reshape(3, 5) / 10
    task = make_task(response=(gain, offset), fft_shift=True)
    expected = torch.fft.fftshift(torch.tensor(np.stack((gain, offset)), dtype=torch.float64), dim=(-2, -1))
    torch.testing.assert_close(task.get_data("detector_response"), expected)
    destination = make_task()
    destination.copy_data_from_task(task, params_to_copy=("detector_response",))
    torch.testing.assert_close(destination.get_data("detector_response"), expected)
    assert not destination.detector_response.is_identity
    task.set_large_tensor_device("cpu")
    assert task.detector_response is task.reconstructor.forward_model.detector_response
    torch.testing.assert_close(task.get_data_to_cpu("detector_response"), expected)


def test_response_copy_preserves_destination_pixel_mask():
    source_mask = np.ones((3, 5), dtype=bool)
    source_mask[0, 0] = False
    destination_mask = np.ones((3, 5), dtype=bool)
    destination_mask[1, 1] = False
    source = make_task(
        response=(np.linspace(0.8, 1.2, 15).reshape(3, 5), np.full((3, 5), 0.05)),
        mask=source_mask,
    )
    destination = make_task(
        mask=destination_mask,
        response_options=api.DetectorResponseOptions(optimizable=True, optimize_gain=True),
    )
    destination.copy_data_from_task(source, params_to_copy=("detector_response",))
    before = destination.get_data("detector_response")
    rec = destination.reconstructor
    with torch.no_grad():
        destination.dataset.patterns.copy_(rec.forward_model(torch.arange(5)) + 0.2)
        rec.update_detector_response()
    for mask in (destination.dataset.valid_pixel_mask, destination.detector_response.valid_pixel_mask,
                 rec.noise_model.valid_pixel_mask):
        torch.testing.assert_close(mask, torch.from_numpy(destination_mask))
    torch.testing.assert_close(destination.detector_response.offset[1, 1], before[1, 1, 1])
    assert (destination.detector_response.gain > 0).all()


@pytest.mark.parametrize("optimize_gain,optimize_offset", [(False, True), (True, False), (True, True)])
@pytest.mark.parametrize("chunk_size", [1, 3, 7])
def test_least_squares_fit_matches_independent_regression(optimize_gain, optimize_offset, chunk_size):
    rng = np.random.default_rng(15)
    intensity = rng.uniform(0.3, 4, (7, 2, 3))
    measured = 1.2 * intensity - 0.1 + rng.normal(0, 0.2, intensity.shape)
    mask = rng.uniform(size=intensity.shape) > 0.2
    mask[:, 0, 0] = False
    response = DetectorResponse(
        (2, 3), options=api.DetectorResponseOptions(
            optimizable=True, optimize_gain=optimize_gain, optimize_offset=optimize_offset
        )
    )
    response.reset_least_squares_statistics()
    for start in range(0, 7, chunk_size):
        end = start + chunk_size
        response.accumulate_least_squares_statistics(
            torch.tensor(intensity[start:end]), torch.tensor(measured[start:end]),
            torch.tensor(mask[start:end])
        )
    target = response.data.detach().numpy().copy()
    for y in range(2):
        for x in range(3):
            i, d = intensity[mask[:, y, x], y, x], measured[mask[:, y, x], y, x]
            if len(i) == 0:
                continue
            if optimize_gain and optimize_offset:
                target[:, y, x] = np.linalg.lstsq(np.stack((i, np.ones_like(i)), axis=1), d, rcond=None)[0]
            elif optimize_gain:
                target[0, y, x] = np.linalg.lstsq(i[:, None], d, rcond=None)[0][0]
            else:
                target[1, y, x] = np.mean(d - i)
    torch.testing.assert_close(response.data + response.calculate_least_squares_update(),
                               torch.tensor(target), rtol=1e-5, atol=1e-6)


def test_unresolved_variance_preserves_gain_and_fits_offset():
    response = DetectorResponse((2, 3), options=api.DetectorResponseOptions(optimizable=True, optimize_gain=True))
    intensity = torch.ones((4, 2, 3)) * 1000
    intensity[:, 0, 1] += torch.tensor([0, 1e-4, 0, -1e-4])
    data = intensity - 0.2
    mask = torch.ones_like(intensity, dtype=torch.bool)
    mask[:, 1, 2] = False
    data[:, 1, 2] = torch.nan
    response.reset_least_squares_statistics()
    response.accumulate_least_squares_statistics(intensity, data, mask)
    update = response.calculate_least_squares_update()
    torch.testing.assert_close(update[0], torch.zeros((2, 3)))
    torch.testing.assert_close(update[1, mask.any(0)], torch.full((5,), -0.2), atol=1e-4, rtol=0)
    assert update[:, 1, 2].eq(0).all()
    assert torch.isfinite(update).all()


@pytest.mark.parametrize("options_class", [api.LSQMLOptions, api.EPIEOptions, api.RPIEOptions,
                                         api.DMOptions, api.RAAROptions, api.MAGPIEOptions, api.BHOptions])
@pytest.mark.parametrize("optimize_gain", [False, True])
@pytest.mark.parametrize("optimizer", [api.Optimizers.SGD, api.Optimizers.LBFGS])
def test_all_analytical_solvers_apply_epoch_fit_through_optimizer(options_class, optimize_gain, optimizer, monkeypatch):
    response_options = api.DetectorResponseOptions(
        optimizable=True, optimize_gain=optimize_gain, optimizer=optimizer,
        optimizer_params={"line_search_fn": "strong_wolfe"} if optimizer == api.Optimizers.LBFGS else {},
    )
    task = make_task(options_class, response_options=response_options)
    rec, response = task.reconstructor, task.detector_response
    with torch.no_grad():
        intensity = rec.forward_model(torch.arange(5))
        gain = torch.linspace(0.8, 1.2, 15).reshape(3, 5) if optimize_gain else torch.ones((3, 5))
        offset = torch.linspace(-0.01, 0.2, 15).reshape(3, 5)
        task.dataset.patterns.copy_(gain * intensity + offset)
    captured = []
    original_step = response.optimizer.step

    @wraps(original_step)
    def record_step(*args, **kwargs):
        captured.append(response.tensor.grad.clone())
        return original_step(*args, **kwargs)

    monkeypatch.setattr(response.optimizer, "step", record_step)
    rec.update_detector_response()
    assert len(captured) == 1
    torch.testing.assert_close(captured[0][0], 1 - gain, atol=2e-5, rtol=1e-4)
    torch.testing.assert_close(captured[0][1], -offset, atol=2e-5, rtol=1e-4)
    torch.testing.assert_close(response.gain, gain, atol=2e-5, rtol=1e-4)
    torch.testing.assert_close(response.offset, offset, atol=2e-5, rtol=1e-4)


@pytest.mark.parametrize("noise", ["gaussian", "poisson"])
def test_analytical_fit_independent_of_noise_and_respects_masks_schedule_and_step(noise):
    mask = np.ones((3, 5), dtype=bool)
    mask[0, 0] = False
    response_options = api.DetectorResponseOptions(
        optimizable=True, step_size=0.25, optimization_plan=OptimizationPlan(start=1, stop=3, stride=2)
    )
    task = make_task(response_options=response_options, mask=mask, noise=noise, blur=0.2)
    rec, response = task.reconstructor, task.detector_response
    rec.options.exclude_measured_pixels_below = 0
    with torch.no_grad():
        intensity = rec.forward_model(torch.arange(5))
        task.dataset.patterns.copy_(intensity + 0.4)
        task.dataset.patterns[:, 0, 0] = 1e8
        task.dataset.patterns[0, 1, 1] = -1e8
        for epoch in (0, 1, 2, 3):
            rec.current_epoch = epoch
            rec.update_detector_response()
    torch.testing.assert_close(response.offset[torch.from_numpy(mask)], torch.full((14,), 0.1), atol=1e-6, rtol=1e-5)
    assert response.offset[0, 0] == 0
    torch.testing.assert_close(response.gain, torch.ones((3, 5)))


@pytest.mark.parametrize("noise_model_class", [PtychographyGaussianNoiseModel, PtychographyPoissonNoiseModel])
def test_autodiff_gain_offset_gradients_and_optimizer_step(noise_model_class):
    options = api.DetectorResponseOptions(optimizable=True, optimize_gain=True, step_size=1e-4)
    task = make_task(api.AutodiffPtychographyOptions, response_options=options)
    rec, response = task.reconstructor, task.detector_response
    model = rec.forward_model
    indices = torch.arange(5)
    with torch.no_grad():
        intensity = model(indices).clone()
        data = 1.1 * intensity + 0.2
    noise_model = noise_model_class()
    loss = noise_model.nll(model(indices), data)
    gradient = torch.autograd.grad(loss, response.tensor)[0]
    # Differentiate the scalar likelihood w.r.t. y independently of the forward model.
    y = intensity.detach().requires_grad_()
    derivative = torch.autograd.grad(noise_model.nll(y, data), y)[0]
    expected = torch.stack(((derivative * intensity).sum(0), derivative.sum(0)))
    torch.testing.assert_close(gradient, expected)
    rec.loss_function = noise_model.nll
    rec.run_minibatch([indices], data)
    assert noise_model.nll(model(indices), data) < loss
    assert response.offset.abs().max() > 0
    assert response.gain.min() > 0
    torch.testing.assert_close(response.gain.mean(), torch.tensor(1.0))


@pytest.mark.parametrize("offset_nonnegative", [False, True])
def test_optimizer_constraints_preserve_masked_pixels_and_fixed_components(offset_nonnegative):
    gain = torch.ones((2, 3))
    mask = torch.ones((2, 3), dtype=torch.bool)
    mask[0, 0] = False
    options = api.DetectorResponseOptions(
        optimizable=True, optimize_gain=True, offset_nonnegative=offset_nonnegative,
        optimizer=api.Optimizers.ADAM, optimizer_params={"weight_decay": 0.1}, step_size=2,
    )
    response = DetectorResponse(
        (2, 3), data=(gain, torch.full((2, 3), -0.2)), options=options, valid_pixel_mask=mask
    )
    response.set_grad(torch.ones_like(response.tensor))
    response.step_optimizer()
    assert response.gain[0, 0] == 1
    assert response.offset[0, 0].item() == pytest.approx(-0.2)
    assert (response.gain[response.valid_pixel_mask] > 0).all()
    torch.testing.assert_close(response.gain[response.valid_pixel_mask],
                               torch.full((5,), options.gain_floor))
    if offset_nonnegative:
        assert (response.offset[response.valid_pixel_mask] >= 0).all()
    else:
        assert (response.offset[response.valid_pixel_mask] < 0).all()
    options.optimize_gain = False
    previous = response.gain.clone()
    response.set_grad(torch.ones_like(response.tensor))
    response.step_optimizer()
    torch.testing.assert_close(response.gain, previous, rtol=0, atol=0)


@pytest.mark.parametrize("options_class", [api.LSQMLOptions, api.EPIEOptions, api.RPIEOptions,
                                         api.DMOptions, api.RAAROptions, api.MAGPIEOptions, api.BHOptions])
@pytest.mark.parametrize("optimizer", [api.Optimizers.SGD, api.Optimizers.LBFGS])
@pytest.mark.parametrize("optimize_gain", [False, True])
def test_response_fit_runs_once_per_epoch_in_complete_reconstruction(
    options_class, optimizer, optimize_gain, monkeypatch,
):
    task = make_task(options_class, response_options=api.DetectorResponseOptions(
        optimizable=True, step_size=0.1, offset_nonnegative=True, optimizer=optimizer,
        optimize_gain=optimize_gain,
    ), refine_object_probe=True)
    response, rec = task.detector_response, task.reconstructor
    with torch.no_grad():
        task.dataset.patterns.copy_(rec.forward_model(torch.arange(5)) + 0.3)
    calls = []
    original_step = response.optimizer.step
    monkeypatch.setattr(rec, "compute_optimizer_loss", lambda *args: pytest.fail(
        "Response epoch step evaluated a likelihood closure instead of the fitted-target closure"
    ))

    @wraps(original_step)
    def record_step(*args, **kwargs):
        calls.append(response.tensor.grad.clone())
        return original_step(*args, **kwargs)

    monkeypatch.setattr(response.optimizer, "step", record_step)
    task.run(n_epochs=2)
    assert len(calls) == 2
    assert response._automatic_optimizer_closure is None
    assert all(torch.isfinite(gradient).all() for gradient in calls)
    assert torch.isfinite(task.get_data("detector_response")).all()
    assert response.offset.abs().max() > 0


@pytest.mark.parametrize("optimize_gain,optimize_offset", [(False, True), (True, False), (True, True)])
@pytest.mark.parametrize("offset_nonnegative", [False, True])
def test_lbfgs_response_fit_preserves_masks_and_fixed_components(
    optimize_gain, optimize_offset, offset_nonnegative, monkeypatch,
):
    mask = np.ones((3, 5), dtype=bool)
    mask[0, 0] = False
    gain = np.ones((3, 5))
    mask[0, 1] = False
    task = make_task(response=(gain, np.full((3, 5), 0.05)), mask=mask,
                     response_options=api.DetectorResponseOptions(
                         optimizable=True, optimize_gain=optimize_gain, optimize_offset=optimize_offset,
                         offset_nonnegative=offset_nonnegative, optimizer=api.Optimizers.LBFGS,
                         optimizer_params={"line_search_fn": "strong_wolfe", "tolerance_grad": 1e-12},
                     ))
    rec, response = task.reconstructor, task.detector_response
    rec.options.exclude_measured_pixels_below = 0
    before = response.data.detach().clone()
    target = before.clone()
    if optimize_gain:
        target[0, 1] = torch.linspace(0.8, 1.2, 5)
        target[0, 2] = torch.linspace(1.2, 0.8, 5)
    if optimize_offset:
        target[1, 1:] = torch.linspace(-0.01, 0.3, 10).reshape(2, 5)
    with torch.no_grad():
        rec.forward_model(torch.arange(5))
        intensity = rec.forward_model.intermediate_variables.detector_intensity
        task.dataset.patterns.copy_(target[0] * intensity + target[1])
        task.dataset.patterns[:, 0, 0] = torch.nan
        task.dataset.patterns[:, 0, 1] = 1e6
        task.dataset.patterns[:, 0, 2] = -1e6
        task.dataset.patterns[0, 1, 1] = -1e6
    gradients = []
    original_process = response.process_optimizer_gradients

    def record_gradients():
        original_process()
        gradients.append(response.tensor.grad.clone())

    monkeypatch.setattr(response, "process_optimizer_gradients", record_gradients)
    rec.update_detector_response()
    expected = target.clone()
    if optimize_offset and offset_nonnegative:
        expected[1].clamp_min_(0)
    torch.testing.assert_close(response.data, expected, atol=2e-5, rtol=1e-4)
    assert len(gradients) > 2  # Initial update and repeated line-search evaluations.
    for gradient in gradients:
        assert gradient[:, 0, :3].eq(0).all()
        if not optimize_gain:
            assert gradient[0].eq(0).all()
        if not optimize_offset:
            assert gradient[1].eq(0).all()


@pytest.mark.parametrize("mode", [api.OptimizerClosureModes.LOSS, api.OptimizerClosureModes.LOSS_AND_GRADIENTS])
def test_response_fit_closure_matches_supplied_update(mode):
    class CheckClosureOptimizer(torch.optim.SGD):
        closure_mode = mode

        def step(self, closure):
            parameter = self.param_groups[0]["params"][0]
            gradient = parameter.grad.clone()
            loss = closure()
            torch.testing.assert_close(parameter.grad, gradient)
            torch.testing.assert_close(loss, 0.5 * gradient.square().sum())
            super().step()
            next_loss = closure()
            torch.testing.assert_close(next_loss, loss.detach() * 0.75 ** 2)
            expected_grad = gradient if mode == api.OptimizerClosureModes.LOSS else 0.75 * gradient
            torch.testing.assert_close(parameter.grad, expected_grad)
            return next_loss

    task = make_task(response_options=api.DetectorResponseOptions(optimizable=True))
    response, rec = task.detector_response, task.reconstructor
    response.optimizer = CheckClosureOptimizer([response.tensor], lr=0.25)
    with torch.no_grad():
        task.dataset.patterns.copy_(rec.forward_model(torch.arange(5)) + 0.4)
    rec.update_detector_response()
    torch.testing.assert_close(response.offset, torch.full((3, 5), 0.1))


def test_response_fit_honors_custom_optimizer_closure(monkeypatch):
    task = make_task(response_options=api.DetectorResponseOptions(
        optimizable=True, optimizer=api.Optimizers.LBFGS,
    ))
    response = task.detector_response
    response.optimizer_closure = response.make_optimizer_closure(
        lambda: 0.5 * (response.offset - 0.7).square().sum()
    )
    monkeypatch.setattr(response, "make_optimizer_closure", lambda *args, **kwargs: pytest.fail(
        "Automatic closure replaced the custom closure"
    ))
    task.reconstructor.update_detector_response()
    torch.testing.assert_close(response.offset, torch.full((3, 5), 0.7))


def test_epoch_response_closure_overrides_unused_distributed_batch_closure(monkeypatch):
    task = make_task(response_options=api.DetectorResponseOptions(
        optimizable=True, optimizer=api.Optimizers.LBFGS,
    ))
    response, rec = task.detector_response, task.reconstructor
    with torch.no_grad():
        task.dataset.patterns.copy_(rec.forward_model(torch.arange(5)) + 0.2)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    reductions = []
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda tensor, op=None: reductions.append((tensor.clone(), op)))
    monkeypatch.setattr(rec, "compute_optimizer_loss", lambda *args: pytest.fail(
        "Response epoch step evaluated the unused distributed batch objective"
    ))
    rec.prepare_optimizer_closures([torch.arange(2)], task.dataset.patterns[:2])
    assert response._automatic_optimizer_closure is not None
    rec.update_detector_response()
    assert len(reductions) == 2
    assert reductions[0][0].shape == (5, 3, 5)
    assert reductions[1][1] == torch.distributed.ReduceOp.MIN
    torch.testing.assert_close(response.offset, torch.full((3, 5), 0.2))


def test_compact_object_closure_with_epoch_response_refinement():
    task = make_task(response_options=api.DetectorResponseOptions(
        optimizable=True, optimizer=api.Optimizers.LBFGS, offset_nonnegative=True,
    ), refine_object_probe=True)
    rec = task.reconstructor
    with torch.no_grad():
        task.dataset.patterns.copy_(rec.forward_model(torch.arange(5)) + 0.2)
    rec.options.batching_mode = api.BatchingModes.COMPACT
    task.object.optimizer_class = torch.optim.LBFGS
    task.object.optimizer_params.update(max_iter=3, line_search_fn="strong_wolfe")
    task.object.build_optimizer()
    calls = []
    original_loss = rec.compute_optimizer_loss

    def record(input_data, y_true):
        calls.append(input_data[0].clone())
        return original_loss(input_data, y_true)

    rec.compute_optimizer_loss = record
    task.run(n_epochs=2)
    assert calls
    for indices in calls:
        torch.testing.assert_close(indices, torch.arange(5))
    assert torch.isfinite(task.get_data("object")).all()
    assert torch.isfinite(task.get_data("detector_response")).all()


def test_autodiff_lbfgs_response_preserves_fixed_gain_during_trials(monkeypatch):
    task = make_task(api.AutodiffPtychographyOptions, response_options=api.DetectorResponseOptions(
        optimizable=True, optimizer=api.Optimizers.LBFGS,
        optimizer_params={"line_search_fn": "strong_wolfe", "max_iter": 30, "tolerance_grad": 1e-12},
    ))
    response, rec = task.detector_response, task.reconstructor
    indices = torch.arange(5)
    with torch.no_grad():
        target = rec.forward_model(indices) + 0.2
    original_forward = rec.forward_model.forward
    trials = []

    def check_fixed_gain(*args, **kwargs):
        torch.testing.assert_close(response.gain, torch.ones_like(response.gain), rtol=0, atol=0)
        trials.append(response.offset.detach().clone())
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(rec.forward_model, "forward", check_fixed_gain)
    rec.run_minibatch([indices], target)
    assert len(trials) > 2
    torch.testing.assert_close(response.offset, torch.full((3, 5), 0.2), atol=2e-4, rtol=1e-3)


@pytest.mark.parametrize("options_class", [api.LSQMLOptions, api.AutodiffPtychographyOptions])
@pytest.mark.parametrize("blur", [None, 0.4])
def test_gain_normalization_preserves_predictions_and_runs_after_update(options_class, blur, monkeypatch):
    mask = np.ones((3, 5), dtype=bool)
    mask[0, 0] = False
    gain = np.linspace(1.5, 2.5, 15).reshape(3, 5)
    gain[0, 0] = 100
    task = make_task(options_class, mask=mask, blur=blur,
                     response=(gain, np.full((3, 5), 0.2)),
                     response_options=api.DetectorResponseOptions(optimizable=True, optimize_gain=True))
    rec, response = task.reconstructor, task.detector_response
    indices = torch.arange(5)
    # Construction preserves supplied normalization.
    torch.testing.assert_close(response.gain, torch.as_tensor(gain))
    calls = []
    normalize = response.normalize_gain

    @torch.no_grad()
    def check_normalization(probe):
        assert probe is task.probe
        before = rec.forward_model(indices).clone()
        probe_before = task.probe.data.clone()
        gain_before, offset_before = response.gain.clone(), response.offset.clone()
        scale = gain_before[response.valid_pixel_mask].mean()
        normalize(probe)
        torch.testing.assert_close(rec.forward_model(indices), before, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(task.probe.data, probe_before * scale.sqrt())
        torch.testing.assert_close(response.gain, gain_before / scale)
        torch.testing.assert_close(response.offset, offset_before, rtol=0, atol=0)
        calls.append(scale)

    monkeypatch.setattr(response, "normalize_gain", check_normalization)
    with torch.no_grad():
        task.dataset.patterns.copy_(rec.forward_model(indices) + 0.1)
        if options_class is api.LSQMLOptions:
            rec.update_detector_response()
            torch.testing.assert_close(rec.forward_model(indices)[:, mask],
                                       task.dataset.patterns[:, mask], rtol=1e-10, atol=1e-10)
        else:
            response.set_grad(torch.full_like(response.tensor, -0.1))
            rec.step_all_optimizers([indices], task.dataset.patterns)
    assert len(calls) == 1
    assert calls[0] > 1.5


@pytest.mark.parametrize("optimizable", [False, True])
@pytest.mark.parametrize("supplied", [False, True])
def test_identity_status_tracks_set_data_and_optimization(optimizable, supplied):
    identity = torch.stack((torch.ones((2, 3)), torch.zeros((2, 3))))
    response = DetectorResponse((2, 3), data=identity if supplied else None,
                                options=api.DetectorResponseOptions(optimizable=optimizable))
    assert response.is_identity is (not optimizable)
    with torch.no_grad():
        response.set_data(identity)
        assert response.is_identity is (not optimizable)
        response.set_data(torch.full((2, 3), 2.0), slicer=0)
        assert not response.is_identity
        response.set_data(identity)
        assert response.is_identity is (not optimizable)
        increment = torch.zeros_like(identity)
        increment[1, 0, 0] = torch.finfo(identity.dtype).tiny
        response.set_data(increment, op="add")
        assert not response.is_identity  # Exact zero, not an allclose tolerance.
        response.set_data(identity)
    response.set_optimizable(True)
    assert not response.is_identity
    response.set_optimizable(False)
    assert response.is_identity


def test_identity_status_requires_exact_unit_gain():
    gain = torch.ones((2, 3))
    gain[0, 0] = torch.nextafter(gain[0, 0], torch.tensor(2.0))
    response = DetectorResponse((2, 3), data=(gain, torch.zeros_like(gain)))
    assert not response.is_identity


def test_identity_status_survives_copy_and_state_loading():
    source = make_task()
    destination = make_task(response=(np.full((3, 5), 2.0), np.full((3, 5), 0.1)))
    destination.copy_data_from_task(source, params_to_copy=("detector_response",))
    assert destination.detector_response.is_identity
    response = destination.detector_response
    nonidentity = DetectorResponse((3, 5), data=(torch.full((3, 5), 2.0), torch.zeros((3, 5))))
    response.load_state_dict(nonidentity.state_dict())
    assert not response.is_identity
    response.load_state_dict(source.detector_response.state_dict())
    assert response.is_identity


def test_identity_cache_refreshes_after_dtype_conversion():
    torch.set_default_dtype(torch.float64)
    gain = torch.ones((2, 3)) + 1e-10
    response = DetectorResponse((2, 3), data=(gain, torch.zeros((2, 3))))
    assert not response.is_identity
    response.float()
    assert response.is_identity


def test_identity_cache_is_reused_during_forward(monkeypatch):
    task = make_task()
    response = task.detector_response
    monkeypatch.setattr(response, "_update_identity_status", lambda: pytest.fail(
        "Unchanged detector response should reuse its cached identity status"
    ))
    with torch.no_grad():
        task.reconstructor.forward_model(torch.arange(5))
        task.reconstructor.forward_model(torch.arange(5))
    assert response.is_identity


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_response_fit_buffers_reset_move_and_stay_out_of_checkpoints(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    response = DetectorResponse((2, 2), options=api.DetectorResponseOptions(optimizable=True))
    checkpoint_keys = set(response.state_dict())
    assert response.least_squares_statistics is None
    assert response.minimum_intensity is None
    response.reset_least_squares_statistics()
    intensity = torch.tensor([[[1., 4.], [3., 8.]], [[2., 6.], [5., 9.]]], requires_grad=True)
    mask = torch.tensor([[[True, False], [True, False]], [[True, True], [True, False]]])
    response.accumulate_least_squares_statistics(intensity, intensity + 1, mask)
    torch.testing.assert_close(response.minimum_intensity, torch.tensor([[1., 6.], [3., torch.inf]]))
    torch.testing.assert_close(response.least_squares_statistics[0],
                               torch.tensor([[2., 1.], [2., 0.]], dtype=torch.float64))
    assert not response.least_squares_statistics.requires_grad
    assert not response.minimum_intensity.requires_grad
    statistics, minimum = response.least_squares_statistics.clone(), response.minimum_intensity.clone()
    response.to(device)
    assert response.least_squares_statistics.device == response.tensor.device
    assert response.minimum_intensity.device == response.tensor.device
    torch.testing.assert_close(response.least_squares_statistics.cpu(), statistics)
    torch.testing.assert_close(response.minimum_intensity.cpu(), minimum)
    assert set(response.state_dict()) == checkpoint_keys
    response.reset_least_squares_statistics()
    assert response.least_squares_statistics.dtype == torch.float64
    assert response.least_squares_statistics.eq(0).all()
    assert response.minimum_intensity.isposinf().all()
    assert response.least_squares_statistics.device == response.tensor.device
    assert response.minimum_intensity.device == response.tensor.device


def test_each_response_fit_uses_only_current_observations():
    task = make_task(response_options=api.DetectorResponseOptions(optimizable=True))
    rec, response = task.reconstructor, task.detector_response
    with torch.no_grad():
        intensity = rec.forward_model(torch.arange(5)).clone()
        for offset in (0.2, 0.7):
            task.dataset.patterns.copy_(intensity + offset)
            rec.update_detector_response()
            torch.testing.assert_close(response.offset, torch.full((3, 5), offset))
            torch.testing.assert_close(response.least_squares_statistics[0],
                                       torch.full((3, 5), 5., dtype=torch.float64))


def test_response_fit_with_no_global_observations_is_noop(monkeypatch):
    task = make_task(response=(np.full((3, 5), 2.), np.full((3, 5), 0.1)),
                     response_options=api.DetectorResponseOptions(optimizable=True, optimize_gain=True))
    response = task.detector_response
    before, probe_before = response.data, task.probe.data
    response.reset_least_squares_statistics()
    intensity = torch.ones((2, 3, 5))
    response.accumulate_least_squares_statistics(
        intensity, torch.full_like(intensity, torch.nan), torch.zeros_like(intensity, dtype=torch.bool)
    )
    monkeypatch.setattr(response.optimizer, "step", lambda *args, **kwargs: pytest.fail(
        "An empty fit should not step the optimizer"
    ))
    response.apply_least_squares_update(task.probe)
    torch.testing.assert_close(response.data, before, rtol=0, atol=0)
    torch.testing.assert_close(task.probe.data, probe_before, rtol=0, atol=0)


def test_response_fit_uses_remote_observations_on_an_empty_rank(monkeypatch):
    task = make_task(response_options=api.DetectorResponseOptions(optimizable=True))
    response = task.detector_response
    intensity = torch.arange(1, 46).reshape(3, 3, 5).to(torch.float64)
    measured = intensity + 0.3
    remote_statistics = torch.stack((
        torch.full((3, 5), 3., dtype=torch.float64), intensity.sum(0), measured.sum(0),
        intensity.square().sum(0), (intensity * measured).sum(0),
    ))
    response.reset_least_squares_statistics()
    reductions = []

    def reduce(tensor, op=None):
        reductions.append(op)
        if op == torch.distributed.ReduceOp.MIN:
            assert tensor.isposinf().all()
            tensor.copy_(intensity.amin(0))
        else:
            assert tensor.eq(0).all()
            tensor.copy_(remote_statistics)

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "all_reduce", reduce)
    response.apply_least_squares_update(task.probe)
    assert reductions == [None, torch.distributed.ReduceOp.MIN]
    torch.testing.assert_close(response.offset, torch.full((3, 5), 0.3))
