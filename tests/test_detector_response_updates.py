"""Numerical contracts for analytical reconstruction with detector response."""
import numpy as np
import pytest
import torch

import ptychi.api as api
from ptychi.api.task import PtychographyTask
from ptychi.utils import get_default_complex_dtype, set_default_complex_dtype, get_constrained_pixel_mask


@pytest.fixture(autouse=True)
def restore_defaults():
    old = torch.get_default_dtype(), torch.get_default_device(), get_default_complex_dtype()
    torch.set_default_device("cpu")
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old[0])
    torch.set_default_device(old[1])
    set_default_complex_dtype(old[2])


def make_task(options_class=api.LSQMLOptions, *, response=True, shape=(3, 5),
              wave_shape=(6, 6), modes=2, blur=None, noise="gaussian", levels=1,
              valid_pixel_mask=None):
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
    gain = np.linspace(0.3, 4.0, np.prod(shape)).reshape(shape)
    offset = np.linspace(0.1, 1.5, np.prod(shape)).reshape(shape)
    if response == "identity":
        gain, offset = np.ones(shape), np.zeros(shape)
    return PtychographyTask(
        options,
        diffraction_data=np.ones((2, *shape)),
        object_data=(1 + rng.random((1, 16, 16)) + 0.3j * rng.random((1, 16, 16))),
        probe_data=(0.5 + rng.random((1, modes, *wave_shape))
                    + 0.3j * rng.random((1, modes, *wave_shape))),
        probe_position_x_px=np.array([-1.0, 1.0]),
        probe_position_y_px=np.array([-1.0, 1.0]),
        detector_response_map=(gain, offset) if response else None,
        valid_pixel_mask=valid_pixel_mask,
    )


@pytest.mark.parametrize("shape,wave_shape", [((3, 5), (6, 6)), ((4, 4), (5, 7)), ((5, 7), (5, 7))])
@pytest.mark.parametrize("blur", [None, 0.4])
def test_detector_intensity_adjoint(shape, wave_shape, blur):
    model = make_task(shape=shape, wave_shape=wave_shape, blur=blur).reconstructor.forward_model
    generator = torch.Generator().manual_seed(11)
    intensity = torch.randn((2, *wave_shape), generator=generator, requires_grad=True)
    detector_vector = torch.randn((2, *shape), generator=generator)
    mapped = model.map_intensity_to_detector(intensity, include_offset=False)
    adjoint = model.intensity_to_detector_adjoint(detector_vector, wave_shape)
    torch.testing.assert_close(
        (mapped * detector_vector).sum(), (intensity * adjoint).sum(), rtol=1e-12, atol=1e-12
    )
    expected = torch.autograd.grad((mapped * detector_vector).sum(), intensity)[0]
    torch.testing.assert_close(adjoint, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("response", [False, "identity", True])
def test_inverse_response_projection_with_crop_and_mask(response):
    task = make_task(api.EPIEOptions, response=response)
    rec, model = task.reconstructor, task.reconstructor.forward_model
    psi = torch.ones((2, 2, 6, 6), dtype=torch.complex128) * (1 + 2j)
    psi[:, :, 0, 1] = 0
    measured = torch.full((2, 3, 5), 11.0)
    mask = torch.ones_like(measured, dtype=torch.bool)
    mask[0, 1, 2] = False
    projected = rec.replace_propagated_exit_wave_magnitude(psi, measured, mask)
    constrained = mask
    prediction = model.map_intensity_to_detector(projected.abs().square().sum(1))
    torch.testing.assert_close(prediction[constrained], measured[constrained])
    full_mask = model.detector_crop_adjoint(constrained, psi.shape[-2:])[:, None].expand_as(psi)
    torch.testing.assert_close(projected[~full_mask], psi[~full_mask], rtol=0, atol=0)
    repeated = rec.replace_propagated_exit_wave_magnitude(projected, measured, mask)
    torch.testing.assert_close(repeated, projected, rtol=1e-12, atol=1e-12)
    below_offset = model.detector_response.offset[None].expand_as(measured) - 1
    zeroed = rec.replace_propagated_exit_wave_magnitude(psi, below_offset, mask)
    torch.testing.assert_close(zeroed[full_mask], torch.zeros_like(zeroed[full_mask]))


@pytest.mark.parametrize("response", [False, "identity", True])
def test_response_projection_rejects_blur(response):
    task = make_task(api.EPIEOptions, blur=0.4, response=response)
    with pytest.raises(ValueError, match="Magnitude projection.*requires"):
        task.reconstructor.replace_propagated_exit_wave_magnitude(
            torch.ones((2, 2, 6, 6), dtype=torch.complex128), task.dataset.patterns
        )


@pytest.mark.parametrize(
    "options_class,levels",
    [(api.EPIEOptions, 1), (api.RPIEOptions, 1), (api.MAGPIEOptions, 1),
     (api.MAGPIEOptions, 2), (api.DMOptions, 1), (api.RAAROptions, 1)],
)
def test_projection_family_update_matches_response_corrected_data(options_class, levels):
    reference = make_task(options_class, response=False, shape=(4, 4), wave_shape=(4, 4),
                          modes=1, levels=levels)
    corrected = make_task(options_class, shape=(4, 4), wave_shape=(4, 4),
                          modes=1, levels=levels)
    indices = torch.arange(2)
    with torch.no_grad():
        signal = reference.reconstructor.forward_model(indices) * 0.7
        reference.dataset.patterns.copy_(signal)
        model = corrected.reconstructor.forward_model
        corrected.dataset.patterns.copy_(model.apply_detector_response(signal))
    reference.run()
    corrected.run()
    for name in ("object", "probe"):
        torch.testing.assert_close(corrected.get_data(name), reference.get_data(name),
                                   rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("noise", ["gaussian", "poisson"])
@pytest.mark.parametrize("blur", [None, 0.4])
@pytest.mark.parametrize("response", [False, True])
def test_noise_model_wavefield_gradient_matches_autograd(noise, blur, response):
    task = make_task(noise=noise, blur=blur, response=response)
    rec, model = task.reconstructor, task.reconstructor.forward_model
    rec.noise_model.eps = 0
    rec.noise_model.loss_function.eps = 0
    rec.noise_model.valid_pixel_mask[1, 2] = False
    task.dataset.valid_pixel_mask[1, 2] = False
    psi = torch.randn((2, 2, 6, 6), dtype=torch.complex128, requires_grad=True)
    y = model.map_intensity_to_detector(psi.abs().square().sum(1))
    data = torch.linspace(0.5, 2.5, y.numel()).reshape(y.shape)
    mask = get_constrained_pixel_mask(
        valid_pixel_mask=rec.dataset.valid_pixel_mask,
        exclude_measured_pixels_below=rec.options.exclude_measured_pixels_below,
        y_true=data,
    )
    loss = torch.where(mask, rec.noise_model.nll(y, data, reduction="none"), 0).sum()
    expected = torch.autograd.grad(loss, psi)[0] / 2
    actual = rec.noise_model.backward_to_psi_far(y, data, psi, forward_model=model)
    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("noise", ["gaussian", "poisson"])
@pytest.mark.parametrize("blur", [None, 0.4])
@pytest.mark.parametrize("gain_scale", [0.1, 30.0])
def test_lsqml_reciprocal_step_reduces_full_response_loss(noise, blur, gain_scale):
    task = make_task(noise=noise, blur=blur)
    rec, model = task.reconstructor, task.reconstructor.forward_model
    model.detector_response.gain.mul_(gain_scale)
    model.detector_response.offset.add_(10)
    task.dataset.valid_pixel_mask[1, 2] = False
    rec.noise_model.valid_pixel_mask[1, 2] = False
    indices = torch.arange(2)
    with torch.no_grad():
        y = model(indices)
        psi = model.intermediate_variables.psi_far.clone()
        data = model.map_intensity_to_detector(psi.abs().square().sum(1) * 0.2)
        data[:, 1, 2] = 1e6  # Masked measurements must not affect the reciprocal step.
        mask = get_constrained_pixel_mask(
            valid_pixel_mask=rec.dataset.valid_pixel_mask,
            exclude_measured_pixels_below=rec.options.exclude_measured_pixels_below,
            y_true=data,
        )
        before = torch.where(mask, rec.noise_model.nll(y, data, reduction="none"), 0).sum()
        updated = rec.run_reciprocal_space_step(y, data, indices)
        after_y = model.map_intensity_to_detector(
            model.free_space_propagator.propagate_forward(updated).abs().square().sum(1)
        )
        after = torch.where(mask, rec.noise_model.nll(after_y, data, reduction="none"), 0).sum()
    assert torch.isfinite(updated).all()
    assert torch.all(rec.alpha_psi_far > 0)
    assert after < before
    assert not torch.allclose(after_y, y)


def test_lsqml_probe_rescale_subtracts_offset_and_uses_mask():
    task = make_task()
    rec, model = task.reconstructor, task.reconstructor.forward_model
    task.dataset.valid_pixel_mask[0, 0] = False
    task.dataset.valid_pixel_mask[0, 1] = False
    with torch.no_grad():
        predicted = model(torch.arange(2))
        signal = predicted - model.detector_response.offset
        measured = 9 * signal + model.detector_response.offset
        measured[:, 0, :2] = 1e8
        probe = task.probe.data.clone()
        rec.update_accumulated_intensities(measured[:1], predicted[:1])
        rec.update_accumulated_intensities(measured[1:], predicted[1:])
        rec._apply_probe_intensity_scaling_correction()
    torch.testing.assert_close(task.probe.data, 3 * probe, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("excluded_value", [0.0, 1e8])
def test_lsqml_identity_probe_rescale_ignores_excluded_measurements(excluded_value):
    mask = torch.ones((3, 5), dtype=torch.bool)
    mask[0, 0] = False
    task = make_task(response=False, valid_pixel_mask=mask)
    rec = task.reconstructor
    rec.options.exclude_measured_pixels_below = 0
    with torch.no_grad():
        predicted = rec.forward_model(torch.arange(2))
        measured = 9 * predicted
        measured[:, 0, 0] = excluded_value  # Excluded by the supplied pixel mask.
        measured[0, 0, 1] = -excluded_value  # Excluded by the measurement threshold.
        probe = task.probe.data.clone()
        rec.update_accumulated_intensities(measured[:1], predicted[:1])
        rec.update_accumulated_intensities(measured[1:], predicted[1:])
        rec._apply_probe_intensity_scaling_correction()
    torch.testing.assert_close(task.probe.data, 3 * probe, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("response", [False, True])
@pytest.mark.parametrize("exclusion", ["mask", "threshold"])
def test_lsqml_probe_rescale_with_all_pixels_excluded(response, exclusion):
    mask = torch.full((3, 5), exclusion != "mask", dtype=torch.bool)
    task = make_task(response=response, valid_pixel_mask=mask)
    rec = task.reconstructor
    with torch.no_grad():
        predicted = rec.forward_model(torch.arange(2))
        measured = 9 * predicted
        if exclusion == "threshold":
            rec.options.exclude_measured_pixels_below = 0
            measured.zero_()
        probe = task.probe.data.clone()
        rec.update_accumulated_intensities(measured, predicted)
        rec._apply_probe_intensity_scaling_correction()
    torch.testing.assert_close(task.probe.data, probe, rtol=0, atol=0)


def test_lsqml_identity_probe_rescale_clamps_negative_target():
    task = make_task(response=False)
    rec = task.reconstructor
    rec.options.exclude_measured_pixels_below = None
    with torch.no_grad():
        predicted = rec.forward_model(torch.arange(2))
        rec.update_accumulated_intensities(-torch.ones_like(predicted), predicted)
        rec._apply_probe_intensity_scaling_correction()
    torch.testing.assert_close(task.probe.data, torch.zeros_like(task.probe.data), rtol=0, atol=0)


@pytest.mark.parametrize("modes", [1, 3])
@pytest.mark.parametrize("response", [False, "identity", True])
@pytest.mark.parametrize("blur", [None, 0.4])
def test_bh_response_gradient_and_hessian_match_autograd(modes, response, blur):
    task = make_task(api.BHOptions, modes=modes, response=response, blur=blur)
    rec, model = task.reconstructor, task.reconstructor.forward_model
    rec.eps = 0
    mask = get_constrained_pixel_mask(
        valid_pixel_mask=rec.dataset.valid_pixel_mask,
        exclude_measured_pixels_below=rec.options.exclude_measured_pixels_below,
        y_true=task.dataset.patterns,
    ).clone()
    mask[:, 1, 2] = False
    rec.current_constrained_pixel_mask = mask[:, None]
    assert model.diffraction_pattern_blur_sigma == blur
    wave = torch.randn((2, modes, 6, 6), dtype=torch.complex128, requires_grad=True)
    psi = model.free_space_propagator.propagate_forward(wave)
    data = task.dataset.patterns.sqrt()[:, None]

    def loss(field):
        y = model.map_intensity_to_detector(field.abs().square().sum(1))
        return (mask * (y.sqrt() - data[:, 0]).square()).sum()

    expected = torch.autograd.grad(loss(psi), wave)[0]
    torch.testing.assert_close(rec.gradientF(psi.detach(), data), expected, rtol=1e-11, atol=1e-11)
    directions = [torch.randn_like(psi), torch.randn_like(psi)]
    expected_hessian = torch.autograd.functional.hessian(
        lambda steps: loss(psi.detach() + steps[0] * directions[0] + steps[1] * directions[1]),
        torch.zeros(2),
    )
    actual = rec.hessianF(psi.detach(), directions[0], directions[1], data)
    torch.testing.assert_close(actual, expected_hessian[0, 1], rtol=1e-11, atol=1e-11)


def test_bh_response_update_matches_autograd_curvature():
    task = make_task(api.BHOptions)
    rec = task.reconstructor
    rec.eps = 0
    indices = torch.arange(2)
    measured = task.dataset.patterns
    with torch.no_grad():
        updates, _ = rec.compute_updates(indices, measured)
    prediction = rec.forward_model(indices)
    loss = (prediction.sqrt() - measured.sqrt()).square().sum()
    parameters = [task.object.tensor.data, task.probe.tensor.data]
    directions = [
        torch.view_as_real(rec.eta_o)[None],
        torch.view_as_real(task.probe.options.rho * rec.eta_p)[None],
    ]
    gradients = torch.autograd.grad(loss, parameters, create_graph=True)
    slope = sum((gradient * direction).sum() for gradient, direction in zip(gradients, directions))
    hessian_products = torch.autograd.grad(slope, parameters)
    curvature = sum(
        (product * direction).sum() for product, direction in zip(hessian_products, directions)
    )
    for update, direction in zip(updates[:2], directions):
        expected = (-slope.detach() / curvature) * torch.view_as_complex(direction)
        torch.testing.assert_close(update, expected, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("noise", ["gaussian", "poisson"])
def test_lsqml_parameter_update_reduces_response_loss(noise):
    task = make_task(noise=noise)
    rec, model = task.reconstructor, task.reconstructor.forward_model
    indices = torch.arange(2)
    with torch.no_grad():
        prediction = model(indices)
        data = 0.8 * (prediction - model.detector_response.offset) + model.detector_response.offset
        before = rec.noise_model.nll(prediction, data)
        task.dataset.patterns.copy_(data)
        task.run()
        after = rec.noise_model.nll(model(indices), data)
    assert after < before


@pytest.mark.parametrize("noise", ["gaussian", "poisson"])
def test_blur_only_reciprocal_step_uses_full_detector_model(noise):
    task = make_task(noise=noise, blur=0.4, response=False)
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
    task = make_task(noise=noise, response=False, shape=(4, 4), wave_shape=(4, 4))
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
    s = model.map_intensity_to_detector((psi.conj() * direction).real.sum(1), include_offset=False)
    t = model.map_intensity_to_detector(direction.abs().square().sum(1), include_offset=False)
    for alpha in (-0.3, 0.0, 0.7):
        expected = model.map_intensity_to_detector((psi - alpha * direction).abs().square().sum(1))
        torch.testing.assert_close(y - 2 * alpha * s + alpha**2 * t, expected,
                                   rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("noise", ["gaussian", "poisson"])
@pytest.mark.parametrize("shape", [(3, 5), (6, 6)])
def test_detector_fixed_point_reduces_to_ideal_step(noise, shape):
    task = make_task(noise=noise, response=False, shape=shape)
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
def test_detector_fixed_point_matches_likelihood_stationarity(noise):
    task = make_task(noise=noise, blur=0.4)
    rec, model = task.reconstructor, task.reconstructor.forward_model
    rec.noise_model.eps = 0
    rec.noise_model.loss_function.eps = 0
    rec.noise_model.valid_pixel_mask[1, 2] = False
    task.dataset.valid_pixel_mask[1, 2] = False
    indices = torch.arange(2)
    with torch.no_grad():
        y = model(indices)
        psi = model.intermediate_variables.psi_far.clone()
        data = 0.6 * (y - model.detector_response.offset) + model.detector_response.offset
        gradient = rec.noise_model.backward_to_psi_far(y, data, psi, forward_model=model)
        denominator = torch.where(
            get_constrained_pixel_mask(
                valid_pixel_mask=rec.dataset.valid_pixel_mask,
                exclude_measured_pixels_below=rec.options.exclude_measured_pixels_below,
                y_true=data,
            ),
            model.map_intensity_to_detector(gradient.abs().square().sum(1), include_offset=False),
            0,
        ).sum((-2, -1))
    gain = model.detector_response.gain.abs().amax().clamp_min(1)
    alpha = rec.alpha_psi_far_all_pos[indices].clamp(0, 1 / gain)
    coefficient = 1 if noise == "poisson" else 1 / (2 * rec.noise_model.sigma**2)
    for _ in range(2):
        previous = alpha.mean().expand_as(alpha).detach().requires_grad_()
        trial = model.map_intensity_to_detector(
            (psi - previous[:, None, None, None] * gradient).abs().square().sum(1)
        )
        loss = torch.where(
            get_constrained_pixel_mask(
                valid_pixel_mask=rec.dataset.valid_pixel_mask,
                exclude_measured_pixels_below=rec.options.exclude_measured_pixels_below,
                y_true=data,
            ),
            rec.noise_model.nll(trial, data, reduction="none"), 0,
        ).sum()
        derivative = torch.autograd.grad(loss, previous)[0]
        # Rearrange the stationarity condition using an independently
        # differentiated full forward model, including the additive offset.
        alpha = previous.detach() - derivative / (2 * coefficient * denominator)
        if noise == "poisson":
            alpha = (0.5 * previous.detach() + 0.5 * alpha).clamp(0, 1 / gain)
    with torch.no_grad(), torch.random.fork_rng(devices=[]):
        torch.manual_seed(73)
        if noise == "poisson":
            alpha += torch.randn(alpha.shape) * (1e-2 / gain)
        torch.manual_seed(73)
        actual = rec.get_psi_far_step_size(y, data, indices, gradient=gradient, eps=0)
    torch.testing.assert_close(actual, alpha, rtol=1e-12, atol=1e-12)


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


@pytest.mark.parametrize("options_class,noise", [
    (api.EPIEOptions, "gaussian"), (api.RPIEOptions, "gaussian"), (api.DMOptions, "gaussian"),
    (api.RAAROptions, "gaussian"), (api.BHOptions, "gaussian"),
    (api.LSQMLOptions, "gaussian"), (api.LSQMLOptions, "poisson"),
])
@pytest.mark.parametrize("shape,wave_shape", [((3, 5), (6, 6)), ((4, 4), (5, 7)), ((6, 6), (6, 6))])
def test_omitted_and_explicit_identity_response_reconstructions_match(options_class, noise, shape, wave_shape):
    mask = np.ones(shape, dtype=bool)
    mask[0, 1] = False
    tasks = [make_task(options_class, response=response, shape=shape, wave_shape=wave_shape,
                       valid_pixel_mask=mask, noise=noise) for response in (False, "identity")]
    with torch.no_grad():
        data = tasks[0].reconstructor.forward_model(torch.arange(2)) * 0.7
        data[:, 0, 1] = 1e6
        for task in tasks:
            task.dataset.patterns.copy_(data)
    for task in tasks:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(427)
            task.run(n_epochs=2)
        assert torch.isfinite(task.get_data("object")).all()
        assert torch.isfinite(task.get_data("probe")).all()
    tolerance = 1e-11
    for name in ("object", "probe"):
        torch.testing.assert_close(tasks[0].get_data(name), tasks[1].get_data(name),
                                   rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("shape,wave_shape", [((3, 5), (6, 6)), ((4, 4), (5, 7))])
def test_direct_parameter_group_identity_response_with_cropped_detector(shape, wave_shape):
    from ptychi.data_structures.parameter_group import PlanarPtychographyParameterGroup
    from ptychi.reconstructors.lsqml import LSQMLReconstructor

    task = make_task(response=False, shape=shape, wave_shape=wave_shape)
    group = PlanarPtychographyParameterGroup(
        object=task.object, probe=task.probe, probe_positions=task.probe_positions,
        opr_mode_weights=task.opr_mode_weights,
    )
    assert group.detector_response.is_identity
    assert group.detector_response.offset.shape == wave_shape
    rec = LSQMLReconstructor(parameter_group=group, dataset=task.dataset,
                             options=task.reconstructor.options)
    rec.build()
    with torch.no_grad():
        prediction = rec.forward_model(torch.arange(2))
        before = group.probe.data.clone()
        rec.update_accumulated_intensities(9 * prediction, prediction)
        rec._apply_probe_intensity_scaling_correction()
    torch.testing.assert_close(group.probe.data, 3 * before, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("noise", ["gaussian", "poisson"])
@pytest.mark.parametrize("exclusion", ["mask", "threshold"])
def test_lsqml_step_bound_ignores_unconstrained_high_gain(noise, exclusion):
    task = make_task(noise=noise)
    rec, model = task.reconstructor, task.reconstructor.forward_model
    if exclusion == "mask":
        task.dataset.valid_pixel_mask[0, 0] = False
        rec.noise_model.valid_pixel_mask[0, 0] = False
    else:
        rec.options.exclude_measured_pixels_below = 0
        rec.noise_model.exclude_measured_pixels_below = 0
    # The high-gain pixel is excluded in this batch, regardless of other batches.
    indices = torch.tensor([0])
    with torch.no_grad(), torch.random.fork_rng(devices=[]):
        data = model(indices).clone() * 0.6
        data[:, 0, 0] = 0
        steps = []
        for gain in (1.0, 1e4):
            response_data = model.detector_response.data
            response_data[0, 0, 0] = gain
            model.detector_response.set_data(response_data)
            prediction = model(indices)
            gradient = rec.noise_model.backward_to_psi_far(
                prediction, data, model.intermediate_variables.psi_far, forward_model=model
            )
            rec.alpha_psi_far_all_pos.fill_(0.5)
            torch.manual_seed(42)
            steps.append(rec.get_psi_far_step_size(prediction, data, indices, gradient=gradient))
    torch.testing.assert_close(steps[1], steps[0], rtol=1e-12, atol=1e-12)
    assert steps[1].min() > 1e-3


@pytest.mark.parametrize("response", [False, "identity", True])
def test_preconditioner_uses_response_corrected_power(response):
    mask = torch.ones((3, 5), dtype=torch.bool)
    mask[0, 0] = False
    task = make_task(response=response, valid_pixel_mask=mask)
    rec = task.reconstructor
    model = rec.forward_model
    calibration = task.detector_response
    signal = torch.arange(30).reshape(2, 3, 5).to(torch.get_default_dtype())
    if response is True:
        signal -= 2
    with torch.no_grad():
        task.dataset.patterns.copy_(signal * calibration.gain + calibration.offset)
        task.dataset.patterns[:, 0, 0] = 1e8
        corrected = signal.clamp_min(0) if response is True else signal
        task.object.update_preconditioner(
            task.probe, task.probe_positions, corrected,
            sampling=model.get_object_sampling_parameters(),
            propagator=model.free_space_propagator, valid_pixel_mask=task.dataset.valid_pixel_mask,
        )
        expected = task.object.preconditioner.clone()
        task.object.preconditioner = None
        rec.update_preconditioners()
    torch.testing.assert_close(task.object.preconditioner, expected, rtol=1e-12, atol=1e-12)
