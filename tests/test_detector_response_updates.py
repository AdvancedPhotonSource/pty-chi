"""Numerical contracts for detector crop and blur adjoints."""
import numpy as np
import pytest
import torch

import ptychi.api as api
from ptychi.api.task import PtychographyTask
from ptychi.utils import get_default_complex_dtype, set_default_complex_dtype


@pytest.fixture(autouse=True)
def restore_defaults():
    old = torch.get_default_dtype(), torch.get_default_device(), get_default_complex_dtype()
    torch.set_default_device("cpu")
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old[0])
    torch.set_default_device(old[1])
    set_default_complex_dtype(old[2])


def make_task(options_class=api.LSQMLOptions, *, shape=(3, 5),
              wave_shape=(6, 6), modes=2, blur=None, noise="gaussian", levels=1):
    options = options_class()
    options.reconstructor_options.default_device = api.Devices.CPU
    options.reconstructor_options.default_dtype = api.Dtypes.FLOAT64
    options.reconstructor_options.batch_size = 2
    options.reconstructor_options.num_epochs = 1
    options.reconstructor_options.allow_nondeterministic_algorithms = False
    options.reconstructor_options.forward_model_options.diffraction_pattern_blur_sigma = blur
    if options_class is api.LSQMLOptions:
        options.reconstructor_options.noise_model = noise
        options.reconstructor_options.rescale_probe_intensity_in_first_epoch = False
    if options_class is api.BHOptions:
        options.reconstructor_options.method = "CG"
    if options_class is api.MAGPIEOptions:
        options.reconstructor_options.multigrid_levels = levels
    options.data_options.fft_shift = False
    options.object_options.pixel_size_m = 1.0
    options.probe_position_options.optimizable = False
    rng = np.random.default_rng(419)
    return PtychographyTask(
        options,
        diffraction_data=np.ones((2, *shape)),
        object_data=(1 + rng.random((1, 16, 16)) + 0.3j * rng.random((1, 16, 16))),
        probe_data=(0.5 + rng.random((1, modes, *wave_shape))
                    + 0.3j * rng.random((1, modes, *wave_shape))),
        probe_position_x_px=np.array([-1.0, 1.0]),
        probe_position_y_px=np.array([-1.0, 1.0]),
    )


@pytest.mark.parametrize("shape,wave_shape", [((3, 5), (6, 6)), ((4, 4), (5, 7)), ((5, 7), (5, 7))])
@pytest.mark.parametrize("blur", [None, 0.4])
def test_detector_intensity_adjoint(shape, wave_shape, blur):
    model = make_task(shape=shape, wave_shape=wave_shape, blur=blur).reconstructor.forward_model
    generator = torch.Generator().manual_seed(11)
    intensity = torch.randn((2, *wave_shape), generator=generator, requires_grad=True)
    detector_vector = torch.randn((2, *shape), generator=generator)
    mapped = model.map_intensity_to_detector(intensity)
    adjoint = model.intensity_to_detector_adjoint(detector_vector, wave_shape)
    torch.testing.assert_close(
        (mapped * detector_vector).sum(), (intensity * adjoint).sum(), rtol=1e-12, atol=1e-12
    )
    expected = torch.autograd.grad((mapped * detector_vector).sum(), intensity)[0]
    torch.testing.assert_close(adjoint, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("noise", ["gaussian", "poisson"])
@pytest.mark.parametrize("blur", [None, 0.4])
def test_noise_model_wavefield_gradient_matches_autograd(noise, blur):
    task = make_task(noise=noise, blur=blur)
    rec, model = task.reconstructor, task.reconstructor.forward_model
    rec.noise_model.eps = 0
    rec.noise_model.loss_function.eps = 0
    rec.noise_model.valid_pixel_mask[1, 2] = False
    task.dataset.valid_pixel_mask[1, 2] = False
    psi = torch.randn((2, 2, 6, 6), dtype=torch.complex128, requires_grad=True)
    y = model.map_intensity_to_detector(psi.abs().square().sum(1))
    data = torch.linspace(0.5, 2.5, y.numel()).reshape(y.shape)
    mask = rec.get_constrained_pixel_mask(data)
    loss = torch.where(mask, rec.noise_model.nll(y, data, reduction="none"), 0).sum()
    expected = torch.autograd.grad(loss, psi)[0] / 2
    actual = rec.noise_model.backward_to_psi_far(y, data, psi, forward_model=model)
    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("noise", ["gaussian", "poisson"])
def test_blur_only_reciprocal_step_uses_full_detector_model(noise):
    task = make_task(noise=noise, blur=0.4)
    rec, model = task.reconstructor, task.reconstructor.forward_model
    indices = torch.arange(2)
    with torch.no_grad():
        prediction = model(indices)
        psi = model.intermediate_variables.psi_far.clone()
        data = prediction * 0.2
        gradient = rec.noise_model.backward_to_psi_far(
            prediction, data, psi, forward_model=model
        )
        cached_steps = rec.alpha_psi_far_all_pos.clone()
        rng_state = torch.get_rng_state()
        alpha = rec.get_psi_far_step_size(prediction, data, indices)
        rec.alpha_psi_far_all_pos.copy_(cached_steps)
        torch.set_rng_state(rng_state)
        before = rec.noise_model.nll(prediction, data)
        updated = rec.run_reciprocal_space_step(prediction, data, indices)
        actual_wave = model.free_space_propagator.propagate_forward(updated)
        expected_wave = psi - alpha[:, None, None, None] * gradient
        torch.testing.assert_close(actual_wave, expected_wave, rtol=1e-11, atol=1e-11)
        after = rec.noise_model.nll(
            model.map_intensity_to_detector(actual_wave.abs().square().sum(1)), data
        )
    assert after < before


@pytest.mark.parametrize("noise", ["gaussian", "poisson"])
def test_wavefield_backward_requires_geometry_for_cropped_detector(noise):
    task = make_task(noise=noise)
    rec = task.reconstructor
    y = rec.forward_model(torch.arange(2))
    psi = rec.forward_model.intermediate_variables.psi_far
    with pytest.raises(ValueError, match="Pass forward_model"):
        rec.noise_model.backward_to_psi_far(y, task.dataset.patterns, psi)


@pytest.mark.parametrize("noise", ["gaussian", "poisson"])
def test_identity_detector_backward_matches_legacy_call(noise):
    task = make_task(noise=noise, shape=(4, 4), wave_shape=(4, 4))
    rec = task.reconstructor
    y = rec.forward_model(torch.arange(2))
    psi = rec.forward_model.intermediate_variables.psi_far
    expected = rec.noise_model.backward_to_psi_far(y, task.dataset.patterns, psi)
    actual = rec.noise_model.backward_to_psi_far(
        y, task.dataset.patterns, psi, forward_model=rec.forward_model
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("blur", [None, 0.4])
def test_detector_intensity_polynomial_matches_full_prediction(blur):
    model = make_task(blur=blur).reconstructor.forward_model
    generator = torch.Generator().manual_seed(631)
    psi = torch.randn((2, 2, 6, 6), dtype=torch.complex128, generator=generator)
    direction = torch.randn(psi.shape, dtype=psi.dtype, generator=generator)
    y = model.map_intensity_to_detector(psi.abs().square().sum(1))
    s = model.map_intensity_to_detector((psi.conj() * direction).real.sum(1))
    t = model.map_intensity_to_detector(direction.abs().square().sum(1))
    for alpha in (-0.3, 0.0, 0.7):
        expected = model.map_intensity_to_detector((psi - alpha * direction).abs().square().sum(1))
        torch.testing.assert_close(y - 2 * alpha * s + alpha**2 * t, expected,
                                   rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("noise", ["gaussian", "poisson"])
@pytest.mark.parametrize("shape", [(3, 5), (6, 6)])
def test_detector_fixed_point_reduces_to_ideal_step(noise, shape):
    task = make_task(noise=noise, shape=shape)
    rec, model = task.reconstructor, task.reconstructor.forward_model
    rec.noise_model.eps = 0
    rec.noise_model.valid_pixel_mask[1, 2] = False
    task.dataset.valid_pixel_mask[1, 2] = False
    indices = torch.arange(2)
    with torch.no_grad(), torch.random.fork_rng(devices=[]):
        y = model(indices)
        data = y * torch.linspace(0.3, 1.7, y.numel()).reshape_as(y)
        gradient = rec.noise_model.backward_to_psi_far(
            y, data, model.intermediate_variables.psi_far, forward_model=model
        )
        cached_steps = rec.alpha_psi_far_all_pos.clone()
        torch.manual_seed(72)
        expected = rec.get_psi_far_step_size(y, data, indices, eps=0)
        rec.alpha_psi_far_all_pos.copy_(cached_steps)
        torch.manual_seed(72)
        # Call the general solver explicitly because the public method preserves
        # the original implementation for an ideal detector.
        actual = rec._get_detector_model_step_size(y, data, indices, gradient, eps=0)
    torch.testing.assert_close(actual, expected.expand_as(actual), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("noise", ["gaussian", "poisson"])
@pytest.mark.parametrize("all_masked", [False, True])
def test_detector_fixed_point_zero_direction(noise, all_masked):
    task = make_task(noise=noise, blur=0.4)
    rec, model = task.reconstructor, task.reconstructor.forward_model
    indices = torch.arange(2)
    with torch.no_grad():
        y = model(indices)
        psi = model.intermediate_variables.psi_far
        if all_masked:
            rec.noise_model.valid_pixel_mask.fill_(False)
            task.dataset.valid_pixel_mask.fill_(False)
            gradient = rec.noise_model.backward_to_psi_far(y, y, psi, forward_model=model)
        else:
            gradient = torch.zeros_like(psi)
        alpha = rec.get_psi_far_step_size(y, y, indices, gradient=gradient)
    torch.testing.assert_close(alpha, torch.zeros_like(alpha), rtol=0, atol=0)
    torch.testing.assert_close(rec.alpha_psi_far_all_pos, alpha, rtol=0, atol=0)
