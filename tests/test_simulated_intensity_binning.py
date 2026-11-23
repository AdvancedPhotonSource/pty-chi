"""Unit tests for simulated_intensity_binning_factor: binning wave-grid intensity."""
import numpy as np
import pytest
import torch

import ptychi.api as api
import ptychi.image_proc as ip
from ptychi.api.options.magpie import MAGPIECapabilityError
from ptychi.api.task import PtychographyTask
from ptychi.utils import get_constrained_pixel_mask

pytestmark = pytest.mark.usefixtures("numerical_cpu")


def make_task(options_class=api.LSQMLOptions, *, factor=2, shape=(3, 5), wave_shape=(8, 12),
              modes=2, blur=None, response=False, noise="gaussian", distance_m=np.inf,
              low_memory=False, response_options=None):
    options = options_class()
    options.reconstructor_options.default_device = api.Devices.CPU
    options.reconstructor_options.default_dtype = api.Dtypes.FLOAT64
    options.reconstructor_options.batch_size = 2
    options.reconstructor_options.allow_nondeterministic_algorithms = False
    forward_model_options = options.reconstructor_options.forward_model_options
    forward_model_options.diffraction_pattern_blur_sigma = blur
    forward_model_options.simulated_intensity_binning_factor = factor
    forward_model_options.low_memory_mode = low_memory
    if response_options is not None:
        forward_model_options.detector_response_options = response_options
    if options_class is api.LSQMLOptions:
        options.reconstructor_options.noise_model = noise
    if options_class is api.MAGPIEOptions:
        options.reconstructor_options.multigrid_levels = 2
    options.data_options.fft_shift = False
    options.data_options.wavelength_m = 1e-10
    options.data_options.free_space_propagation_distance_m = distance_m
    options.object_options.pixel_size_m = 1e-8
    options.probe_position_options.optimizable = False
    rng = np.random.default_rng(419)
    gain = np.linspace(0.3, 4.0, np.prod(shape)).reshape(shape)
    offset = np.linspace(0.1, 1.5, np.prod(shape)).reshape(shape)
    return PtychographyTask(
        options,
        diffraction_data=np.ones((2, *shape)),
        object_data=(1 + rng.random((1, 24, 24)) + 0.3j * rng.random((1, 24, 24))),
        probe_data=(0.5 + rng.random((1, modes, *wave_shape))
                    + 0.3j * rng.random((1, modes, *wave_shape))),
        probe_position_x_px=np.array([-1.0, 1.0]),
        probe_position_y_px=np.array([-1.0, 1.0]),
        detector_response_map=(gain, offset) if response else None,
    )


def constrained_mask(rec, data):
    return get_constrained_pixel_mask(
        valid_pixel_mask=rec.dataset.valid_pixel_mask,
        exclude_measured_pixels_below=rec.options.exclude_measured_pixels_below,
        y_true=data,
    )


def test_sum_bin_2d():
    image = torch.arange(16.0).reshape(1, 4, 4)
    torch.testing.assert_close(
        ip.sum_bin_2d(image, 2), torch.tensor([[[10.0, 18.0], [42.0, 50.0]]])
    )
    torch.testing.assert_close(
        ip.sum_bin_2d(torch.tensor([[1.0, 2.0]]), 2, adjoint=True),
        torch.tensor([[1.0, 1.0, 2.0, 2.0], [1.0, 1.0, 2.0, 2.0]]),
    )
    assert ip.sum_bin_2d(image, 1) is image
    with pytest.raises(ValueError, match="divisible"):
        ip.sum_bin_2d(torch.zeros((1, 6, 5)), 2)
    # The adjoint is the exact transpose, also for complex input and an odd factor.
    generator = torch.Generator().manual_seed(3)
    fine = torch.randn((2, 9, 12), generator=generator, dtype=torch.complex128)
    coarse = torch.randn((2, 3, 4), generator=generator, dtype=torch.complex128)
    torch.testing.assert_close(
        torch.vdot(ip.sum_bin_2d(fine, 3).flatten(), coarse.flatten()),
        torch.vdot(fine.flatten(), ip.sum_bin_2d(coarse, 3, adjoint=True).flatten()),
    )


@pytest.mark.parametrize(
    "options_class,distance_m,low_memory",
    [
        (api.LSQMLOptions, np.inf, False),
        (api.LSQMLOptions, np.inf, True),
        (api.LSQMLOptions, 1e-4, False),
        (api.EPIEOptions, np.inf, False),
        (api.DMOptions, np.inf, False),
        (api.RAAROptions, np.inf, False),
        (api.MAGPIEOptions, np.inf, False),
        (api.BHOptions, np.inf, False),
        (api.AutodiffPtychographyOptions, np.inf, False),
    ],
)
@torch.no_grad()
def test_forward_model_bins_then_crops(options_class, distance_m, low_memory):
    # MAGPIE rejects detector cropping, so its detector is the binned wave grid.
    magpie = options_class is api.MAGPIEOptions
    shape, modes = ((4, 6), 1) if magpie else ((3, 5), 2)
    kwargs = dict(modes=modes, distance_m=distance_m, low_memory=low_memory)
    indices = torch.arange(2)
    unbinned = make_task(options_class, factor=1, shape=(8, 12), **kwargs)
    binned = ip.sum_bin_2d(unbinned.reconstructor.forward_model(indices), 2)
    if np.isinf(distance_m):
        # Far-field intensities keep the zero frequency at the array corner.
        expected = torch.fft.ifftshift(
            ip.central_crop(torch.fft.fftshift(binned, dim=(-2, -1)), shape), dim=(-2, -1)
        )
    else:
        # Near-field intensities are in spatial order.
        expected = ip.central_crop(binned, shape)
    model = make_task(options_class, shape=shape, **kwargs).reconstructor.forward_model
    torch.testing.assert_close(model(indices), expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    "options_class,factor,wave_shape,error,message",
    [
        (api.LSQMLOptions, 0, (8, 12), ValueError, "greater than or equal to 1"),
        (api.LSQMLOptions, 2, (9, 12), ValueError, "divisible"),
        (api.LSQMLOptions, 3, (12, 9), ValueError, "smaller than the detector"),
        (api.MAGPIEOptions, 2, (12, 12), MAGPIECapabilityError, "matching detector and probe"),
    ],
)
def test_incompatible_settings_are_rejected(options_class, factor, wave_shape, error, message):
    with pytest.raises(error, match=message):
        make_task(options_class, factor=factor, shape=(4, 4), wave_shape=wave_shape, modes=1)


@pytest.mark.parametrize(
    "noise,factor,shape,wave_shape,blur,response",
    [
        ("gaussian", 2, (3, 5), (8, 12), None, False),
        ("gaussian", 2, (3, 5), (8, 12), 0.4, True),
        ("poisson", 3, (2, 3), (6, 12), 0.4, True),
    ],
)
def test_noise_model_wavefield_gradient_matches_autograd(noise, factor, shape, wave_shape,
                                                         blur, response):
    task = make_task(noise=noise, factor=factor, shape=shape, wave_shape=wave_shape, blur=blur,
                     response=response)
    rec, model = task.reconstructor, task.reconstructor.forward_model
    rec.noise_model.eps = 0
    rec.noise_model.loss_function.eps = 0
    rec.noise_model.valid_pixel_mask[1, 1] = False
    task.dataset.valid_pixel_mask[1, 1] = False
    psi = torch.randn((2, 2, *wave_shape), dtype=torch.complex128, requires_grad=True)
    y = model.map_intensity_to_detector(psi.abs().square().sum(1))
    data = torch.linspace(0.5, 2.5, y.numel()).reshape(y.shape)
    mask = constrained_mask(rec, data)
    loss = torch.where(mask, rec.noise_model.nll(y, data, reduction="none"), 0).sum()
    expected = torch.autograd.grad(loss, psi)[0] / 2
    actual = rec.noise_model.backward_to_psi_far(
        y.detach(), data, psi.detach(), forward_model=model
    )
    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("factor,response", [(2, False), (3, True)])
def test_projection_scales_each_block_to_the_measured_intensity(factor, response):
    shape, wave_shape = (3, 5), (4 * factor, 6 * factor)
    task = make_task(api.EPIEOptions, factor=factor, shape=shape, wave_shape=wave_shape,
                     response=response)
    rec, model = task.reconstructor, task.reconstructor.forward_model
    psi = torch.randn((2, 2, *wave_shape), dtype=torch.complex128,
                      generator=torch.Generator().manual_seed(5))
    psi[0, :, :factor, :factor] = 0  # The first pattern's zero-frequency block vanishes.
    measured = torch.full((2, *shape), 11.0)
    mask = torch.ones_like(measured, dtype=torch.bool)
    mask[0, 1, 2] = False
    projected = rec.noise_model.project_to_propagated_magnitude_constraint(
        psi, measured, mask, forward_model=model
    )
    prediction = model.map_intensity_to_detector(projected.abs().square().sum(1))
    torch.testing.assert_close(prediction[mask], measured[mask])
    wave_mask = model.detector_crop_adjoint(mask, model.binned_wave_shape(wave_shape))
    wave_mask = ip.sum_bin_2d(wave_mask, factor, adjoint=True)[:, None].expand_as(psi)
    torch.testing.assert_close(projected[~wave_mask], psi[~wave_mask], rtol=0, atol=0)
    # One real positive factor per block, shared by all modes, keeps phases and mode ratios.
    ratio = projected[1] / psi[1]
    block_mean = ip.sum_bin_2d(ratio, factor) / factor**2
    torch.testing.assert_close(
        ratio, ip.sum_bin_2d(block_mean, factor, adjoint=True).expand_as(ratio)
    )
    torch.testing.assert_close(ratio[0], ratio[1])
    assert torch.all(ratio.real > 0)
    torch.testing.assert_close(ratio.imag, torch.zeros_like(ratio.imag), rtol=0, atol=1e-12)
    # The vanishing block receives the target evenly in the first mode.
    target = measured[0, 0, 0]
    if response:
        detector_response = model.detector_response
        target = (target - detector_response.offset[0, 0]) / detector_response.gain[0, 0]
    expected = (target / factor**2).sqrt().expand(factor, factor).to(projected.dtype)
    torch.testing.assert_close(projected[0, 0, :factor, :factor], expected)
    torch.testing.assert_close(projected[0, 1, :factor, :factor], torch.zeros_like(expected))
    repeated = rec.noise_model.project_to_propagated_magnitude_constraint(
        projected, measured, mask, forward_model=model
    )
    torch.testing.assert_close(repeated, projected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    "noise,blur,response", [("gaussian", None, False), ("gaussian", 0.4, True),
                            ("poisson", 0.4, True)]
)
@torch.no_grad()
def test_lsqml_reciprocal_step(noise, blur, response):
    task = make_task(noise=noise, blur=blur, response=response)
    rec, model = task.reconstructor, task.reconstructor.forward_model
    task.dataset.valid_pixel_mask[1, 2] = False
    rec.noise_model.valid_pixel_mask[1, 2] = False
    indices = torch.arange(2)
    predicted = model(indices)
    data = predicted * torch.linspace(0.3, 2.0, predicted.numel()).reshape(predicted.shape)
    data[:, 1, 2] = 1e6  # Masked measurements must not affect the reciprocal step.
    mask = constrained_mask(rec, data)
    updated = rec.run_reciprocal_space_step(predicted, data, indices)
    after = model.map_intensity_to_detector(
        model.free_space_propagator.propagate_forward(updated).abs().square().sum(1)
    )
    assert torch.isfinite(updated).all()
    if noise == "gaussian" and blur is None and not response:
        # The identity-detector Gaussian step lands exactly on the binned data.
        torch.testing.assert_close(after[mask], data[mask], rtol=1e-8, atol=1e-8)
    nll = rec.noise_model.nll
    assert (torch.where(mask, nll(after, data, reduction="none"), 0).sum()
            < torch.where(mask, nll(predicted, data, reduction="none"), 0).sum())


@pytest.mark.parametrize("modes,response,blur", [(1, False, None), (3, True, 0.4)])
def test_bh_gradient_and_hessian_match_autograd(modes, response, blur):
    task = make_task(api.BHOptions, modes=modes, response=response, blur=blur)
    rec, model = task.reconstructor, task.reconstructor.forward_model
    rec.eps = 0
    mask = constrained_mask(rec, task.dataset.patterns).clone()
    mask[:, 1, 2] = False
    rec.current_constrained_pixel_mask = mask[:, None]
    wave = torch.randn((2, modes, 8, 12), dtype=torch.complex128, requires_grad=True)
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


def test_epoch_response_fit_uses_binned_intensity():
    task = make_task(response_options=api.DetectorResponseOptions(optimizable=True,
                                                                  optimize_gain=True))
    rec, response = task.reconstructor, task.detector_response
    # Unit mean, so the gain normalization that follows a gain fit leaves it unchanged.
    gain = torch.linspace(0.8, 1.2, 15).reshape(3, 5)
    with torch.no_grad():
        task.dataset.patterns.copy_(gain * rec.forward_model(torch.arange(2)) + 0.4)
        rec.current_epoch = 0
        rec.update_detector_response()
    torch.testing.assert_close(response.offset, torch.full_like(response.offset, 0.4))
    torch.testing.assert_close(response.gain, gain)
