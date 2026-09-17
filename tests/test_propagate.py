import math

import pytest
import torch

import ptychi.maths as pmath
import ptychi.utils as utils
from ptychi.propagate import (
    AngularSpectrumPropagator,
    FraunhoferPropagator,
    FourierPropagator,
    FresnelTransformPropagator,
    WavefieldPropagatorParameters,
)

# Hard-x-ray defaults: pixels far larger than the wavelength, so no bin is evanescent.
_WAVELENGTH_M = 1.24e-10

# All three classes take a parameters object; FourierPropagator does not, so it is
# exercised separately where it applies.
_PARAMETERIZED_PROPAGATORS = (
    AngularSpectrumPropagator,
    FresnelTransformPropagator,
    FraunhoferPropagator,
)
_SINGLE_FFT_PROPAGATORS = (FresnelTransformPropagator, FraunhoferPropagator)


@pytest.fixture(autouse=True)
def _double_precision_on_cpu():
    """Pin device and precision, and restore them afterwards.

    `test_utils.BaseTester.setup_ptychi` sets the default device to cuda and the default
    dtypes to float32/complex64 without restoring them, and `PtychographyTask` overrides
    the FFT precision flag. `get_spatial_coordinates` and `get_frequency_coordinates` build
    tensors on the default device, so a gold test running earlier in the same session would
    otherwise drag this module onto the GPU at reduced precision.
    """
    previous_device = torch.get_default_device()
    previous_dtype = torch.get_default_dtype()
    previous_complex_dtype = utils.get_default_complex_dtype()
    previous_fft_precision = pmath.get_use_double_precision_for_fft()
    torch.set_default_device("cpu")
    torch.set_default_dtype(torch.float64)
    utils.set_default_complex_dtype(torch.complex128)
    pmath.set_use_double_precision_for_fft(True)
    try:
        yield
    finally:
        pmath.set_use_double_precision_for_fft(previous_fft_precision)
        utils.set_default_complex_dtype(previous_complex_dtype)
        torch.set_default_dtype(previous_dtype)
        torch.set_default_device(previous_device)


def _params(
    propagation_distance_m,
    wavelength_m=_WAVELENGTH_M,
    width_px=64,
    height_px=64,
    pixel_width_m=50e-6,
    pixel_height_m=None,
):
    return WavefieldPropagatorParameters.create_simple(
        wavelength_m=wavelength_m,
        width_px=width_px,
        height_px=height_px,
        pixel_width_m=pixel_width_m,
        pixel_height_m=pixel_width_m if pixel_height_m is None else pixel_height_m,
        propagation_distance_m=propagation_distance_m,
    )


def _random_wavefield(shape, seed=0):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    real = torch.randn(*shape, generator=generator, dtype=torch.float64, device="cpu")
    imag = torch.randn(*shape, generator=generator, dtype=torch.float64, device="cpu")
    return (real + 1j * imag).to(torch.complex128)


def _relative_error(actual, expected):
    return float(torch.linalg.norm(actual - expected) / torch.linalg.norm(expected))


def _centered_grid(num_px, spacing=1.0):
    return (torch.arange(num_px, dtype=torch.float64, device="cpu") - num_px // 2) * spacing


def _evanescent_params(propagation_distance_m=5e-9):
    """A geometry whose pixels are smaller than the wavelength, so high spatial
    frequencies fall outside the propagating cone."""
    return _params(
        propagation_distance_m,
        wavelength_m=1e-9,
        width_px=64,
        height_px=64,
        pixel_width_m=0.4e-9,
    )


def _evanescent_mask(parameters):
    frequency_y, frequency_x = parameters.get_frequency_coordinates()
    ratio = (frequency_x.double() ** 2 + frequency_y.double() ** 2) / (
        parameters.pixel_width_wlu**2
    )
    return ratio >= 1


def _transfer_function(propagator):
    return propagator._transfer_function_real + 1j * propagator._transfer_function_imag


# --------------------------------------------------------------------------------------
# Round trip
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("propagator_class", _PARAMETERIZED_PROPAGATORS)
@pytest.mark.parametrize("propagation_distance_m", [1.0, -1.0])
@pytest.mark.parametrize(
    ("width_px", "height_px", "pixel_width_m", "pixel_height_m"),
    [(64, 64, 50e-6, 50e-6), (64, 48, 50e-6, 40e-6), (63, 65, 50e-6, 50e-6)],
)
def test_backward_inverts_forward(
    propagator_class, propagation_distance_m, width_px, height_px, pixel_width_m, pixel_height_m
):
    parameters = _params(
        propagation_distance_m,
        width_px=width_px,
        height_px=height_px,
        pixel_width_m=pixel_width_m,
        pixel_height_m=pixel_height_m,
    )
    propagator = propagator_class(parameters)
    wavefield = _random_wavefield((2, 3, height_px, width_px))

    recovered = propagator.propagate_backward(propagator.propagate_forward(wavefield))

    torch.testing.assert_close(recovered, wavefield, rtol=1e-10, atol=1e-10)


def test_fourier_propagator_backward_inverts_forward():
    propagator = FourierPropagator()
    wavefield = _random_wavefield((2, 3, 64, 64))

    recovered = propagator.propagate_backward(propagator.propagate_forward(wavefield))

    torch.testing.assert_close(recovered, wavefield, rtol=1e-12, atol=1e-14)


def test_angular_spectrum_backward_equals_forward_at_negated_distance():
    """Only true for the angular spectrum: its transfer function satisfies
    tf(-z) == conj(tf(z)). The single-FFT propagators swap which plane the stored pixel
    pitch describes when z is negated, so they do not have this property."""
    wavefield = _random_wavefield((2, 64, 64))

    backward = AngularSpectrumPropagator(_params(1.0)).propagate_backward(wavefield)
    forward_at_negated = AngularSpectrumPropagator(_params(-1.0)).propagate_forward(wavefield)

    torch.testing.assert_close(backward, forward_at_negated, rtol=1e-12, atol=1e-14)


# --------------------------------------------------------------------------------------
# Which operator identity holds for which class
# --------------------------------------------------------------------------------------


def test_angular_spectrum_backward_is_the_adjoint():
    """The angular spectrum is a partial isometry -- its transfer function is either
    unit-modulus or exactly zero -- so backward is simultaneously adjoint and inverse."""
    parameters = _evanescent_params()
    propagator = AngularSpectrumPropagator(parameters)
    x = _random_wavefield((64, 64), seed=1)
    y = _random_wavefield((64, 64), seed=2)

    forward_inner = torch.vdot(propagator.propagate_forward(x).ravel(), y.ravel())
    backward_inner = torch.vdot(x.ravel(), propagator.propagate_backward(y).ravel())

    torch.testing.assert_close(forward_inner, backward_inner, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("propagator_class", _SINGLE_FFT_PROPAGATORS)
def test_single_fft_backward_is_the_inverse_not_the_adjoint(propagator_class):
    """`A * DFT * B` with an unnormalized DFT and |C0| != 1. Backward is the inverse; the
    adjoint is that same operator scaled by N * M * |C0|**2. Both halves are asserted so
    that "fixing" backward into the adjoint fails here."""
    parameters = _params(1.0, width_px=64, height_px=48, pixel_width_m=50e-6, pixel_height_m=40e-6)
    propagator = propagator_class(parameters)
    x = _random_wavefield((48, 64), seed=1)
    y = _random_wavefield((48, 64), seed=2)

    forward_inner = torch.vdot(propagator.propagate_forward(x).ravel(), y.ravel())
    backward_inner = torch.vdot(x.ravel(), propagator.propagate_backward(y).ravel())
    scale = 64 * 48 * abs(propagator._C0) ** 2

    torch.testing.assert_close(forward_inner, backward_inner * scale, rtol=1e-11, atol=1e-11)
    assert not torch.allclose(forward_inner, backward_inner, rtol=1e-3)


# --------------------------------------------------------------------------------------
# Energy
# --------------------------------------------------------------------------------------


def test_angular_spectrum_is_unitary_without_evanescent_modes():
    propagator = AngularSpectrumPropagator(_params(1.0))
    wavefield = _random_wavefield((64, 64))

    propagated = propagator.propagate_forward(wavefield)

    torch.testing.assert_close(
        torch.linalg.norm(propagated), torch.linalg.norm(wavefield), rtol=1e-12, atol=1e-12
    )


def test_angular_spectrum_is_contractive_with_evanescent_modes():
    parameters = _evanescent_params()
    propagator = AngularSpectrumPropagator(parameters)
    wavefield = _random_wavefield((64, 64))

    assert _evanescent_mask(parameters).any()
    propagated_norm = float(torch.linalg.norm(propagator.propagate_forward(wavefield)))
    source_norm = float(torch.linalg.norm(wavefield))

    assert 0.0 < propagated_norm < 0.99 * source_norm


@pytest.mark.parametrize("propagator_class", _SINGLE_FFT_PROPAGATORS)
def test_single_fft_conserves_physical_energy(propagator_class):
    """The output grid has pitch lambda|z|/(N dx), so energy is conserved once the change
    of pixel area is accounted for -- exactly, not approximately."""
    width_px, height_px = 64, 48
    pixel_width_m, pixel_height_m = 50e-6, 40e-6
    propagation_distance_m = 1.0
    parameters = _params(
        propagation_distance_m,
        width_px=width_px,
        height_px=height_px,
        pixel_width_m=pixel_width_m,
        pixel_height_m=pixel_height_m,
    )
    propagator = propagator_class(parameters)
    wavefield = _random_wavefield((height_px, width_px))

    propagated = propagator.propagate_forward(wavefield)
    numerator_m2 = _WAVELENGTH_M * abs(propagation_distance_m)
    output_pixel_width_m = numerator_m2 / (width_px * pixel_width_m)
    output_pixel_height_m = numerator_m2 / (height_px * pixel_height_m)

    source_energy = float((wavefield.abs() ** 2).sum()) * pixel_width_m * pixel_height_m
    output_energy = (
        float((propagated.abs() ** 2).sum()) * output_pixel_width_m * output_pixel_height_m
    )

    assert output_energy == pytest.approx(source_energy, rel=1e-12)


def test_fresnel_transform_does_not_conserve_the_raw_l2_norm():
    """Pins the C0 amplitude prefactor, which the physical-energy test alone would not."""
    parameters = _params(1.0, width_px=64, height_px=48, pixel_width_m=50e-6, pixel_height_m=40e-6)
    propagator = FresnelTransformPropagator(parameters)
    wavefield = _random_wavefield((48, 64))

    ratio = float(
        torch.linalg.norm(propagator.propagate_forward(wavefield))
        / torch.linalg.norm(wavefield)
    )
    expected = abs(propagator._C0) * math.sqrt(64 * 48)

    assert ratio == pytest.approx(expected, rel=1e-12)
    assert not math.isclose(expected, 1.0, rel_tol=0.1)


# --------------------------------------------------------------------------------------
# Propagator selection
# --------------------------------------------------------------------------------------

# (wavelength_m, width_px, height_px, pixel_width_m, pixel_height_m, distance_m, expected)
_SELECTOR_CASES = [
    # Square pixels: the crossover sits at Fr == 1 / N, and the rule is strict.
    (1e-10, 64, 64, 1e-6, 1e-6, 0.5, False),
    (1e-10, 64, 64, 1e-6, 1e-6, 0.64, False),
    (1e-10, 64, 64, 1e-6, 1e-6, 1.0, True),
    # Anisotropic: either axis alone can force the Fresnel transform.
    (1e-10, 64, 64, 1e-6, 0.5e-6, 0.5, True),
    (1e-10, 64, 64, 1e-6, 2e-6, 2.0, True),
    (1e-10, 64, 64, 1e-6, 2e-6, 0.5, False),
    # Non-square arrays: the threshold is per axis, 1 / N and 1 / M.
    (1e-10, 32, 128, 1e-6, 1e-6, 0.4, True),
    # Near-field gold geometry (tests/test_2d_ptycho_near_field.py).
    (1240 / 33.35 * 1e-12, 1152, 1152, 2e-8, 2e-8, 0.0043, False),
]


@pytest.mark.parametrize(
    ("wavelength_m", "width_px", "height_px", "pixel_width_m", "pixel_height_m",
     "propagation_distance_m", "expected"),
    _SELECTOR_CASES,
)
def test_selector_uses_per_axis_critical_sampling(
    wavelength_m, width_px, height_px, pixel_width_m, pixel_height_m,
    propagation_distance_m, expected,
):
    parameters = _params(
        propagation_distance_m,
        wavelength_m=wavelength_m,
        width_px=width_px,
        height_px=height_px,
        pixel_width_m=pixel_width_m,
        pixel_height_m=pixel_height_m,
    )

    assert parameters.is_fresnel_transform_preferrable() is expected


@pytest.mark.parametrize(
    ("wavelength_m", "width_px", "height_px", "pixel_width_m", "pixel_height_m",
     "propagation_distance_m", "expected"),
    _SELECTOR_CASES,
)
def test_selector_matches_the_far_field_pitch_rule(
    wavelength_m, width_px, height_px, pixel_width_m, pixel_height_m,
    propagation_distance_m, expected,
):
    """Independent oracle, written in meters straight from the sampling condition, so an
    algebra slip in the dimensionless form cannot be mirrored into the test."""
    parameters = _params(
        propagation_distance_m,
        wavelength_m=wavelength_m,
        width_px=width_px,
        height_px=height_px,
        pixel_width_m=pixel_width_m,
        pixel_height_m=pixel_height_m,
    )
    numerator_m2 = wavelength_m * abs(propagation_distance_m)
    angular_spectrum_is_sampled = (
        numerator_m2 / (width_px * pixel_width_m) <= pixel_width_m
        and numerator_m2 / (height_px * pixel_height_m) <= pixel_height_m
    )

    assert parameters.is_fresnel_transform_preferrable() is (not angular_spectrum_is_sampled)


@pytest.mark.parametrize("propagation_distance_m", [0.5, 0.64, 1.0])
def test_selector_ignores_the_propagation_direction(propagation_distance_m):
    """Regression: the selector used to compare a signed Fresnel number against a positive
    threshold, so every negative distance chose the Fresnel transform."""
    forward = _params(propagation_distance_m, width_px=64, height_px=64, pixel_width_m=1e-6)
    backward = _params(-propagation_distance_m, width_px=64, height_px=64, pixel_width_m=1e-6)

    assert (
        forward.is_fresnel_transform_preferrable()
        == backward.is_fresnel_transform_preferrable()
    )


def test_selector_raises_at_zero_propagation_distance():
    """Zero distance is out of contract: the conjugate-plane pitch vanishes."""
    with pytest.raises(ZeroDivisionError):
        _params(0.0).is_fresnel_transform_preferrable()


def test_near_field_gold_geometry_keeps_a_wide_margin_on_angular_spectrum():
    """Documents why tests/test_2d_ptycho_near_field.py is unaffected by the threshold
    change, and fails cheaply here if the threshold is ever widened too far."""
    parameters = _params(
        0.0043,
        wavelength_m=1240 / 33.35 * 1e-12,
        width_px=1152,
        height_px=1152,
        pixel_width_m=2e-8,
    )

    assert parameters.is_fresnel_transform_preferrable() is False
    assert abs(parameters.fresnel_number) > 2 * (1 / 1152)


# --------------------------------------------------------------------------------------
# Evanescent modes
# --------------------------------------------------------------------------------------


def test_transfer_function_blocks_evanescent_modes():
    parameters = _evanescent_params()
    mask = _evanescent_mask(parameters)
    transfer_function = _transfer_function(AngularSpectrumPropagator(parameters))

    assert mask.any()
    assert torch.equal(transfer_function[mask], torch.zeros_like(transfer_function[mask]))
    torch.testing.assert_close(
        transfer_function[~mask].abs(),
        torch.ones_like(transfer_function[~mask].abs()),
        rtol=1e-12,
        atol=1e-12,
    )


def test_transfer_function_has_no_nans():
    """Regression: sqrt(1 - ratio) is NaN above the cutoff and torch.where does not stop
    the value from being formed."""
    transfer_function = _transfer_function(AngularSpectrumPropagator(_evanescent_params()))

    assert torch.isfinite(transfer_function).all()


def test_evanescent_round_trip_is_an_idempotent_projector():
    parameters = _evanescent_params()
    propagator = AngularSpectrumPropagator(parameters)
    wavefield = _random_wavefield((64, 64))

    def project(field):
        return propagator.propagate_backward(propagator.propagate_forward(field))

    projected = project(wavefield)

    torch.testing.assert_close(project(projected), projected, rtol=1e-11, atol=1e-11)
    # Band limiting really does remove content: this is not the identity.
    assert _relative_error(projected, wavefield) > 0.1


def test_round_trip_is_the_identity_without_evanescent_modes():
    """The counterpart of the projector test, and the property every in-tree geometry
    relies on."""
    parameters = _params(1.0)
    propagator = AngularSpectrumPropagator(parameters)
    wavefield = _random_wavefield((64, 64))

    assert not _evanescent_mask(parameters).any()
    recovered = propagator.propagate_backward(propagator.propagate_forward(wavefield))

    torch.testing.assert_close(recovered, wavefield, rtol=1e-12, atol=1e-14)


# --------------------------------------------------------------------------------------
# Gradients through an optimizable propagation distance
# --------------------------------------------------------------------------------------


def _distance_gradient(parameters_factory, use_update=False):
    distance = torch.tensor(5e-9, dtype=torch.float64, device="cpu", requires_grad=True)
    wavefield = _random_wavefield((64, 64))
    if use_update:
        propagator = AngularSpectrumPropagator(parameters_factory(1e-9))
        propagator.update(parameters_factory(distance))
        propagated = propagator.propagate_backward(wavefield)
    else:
        propagator = AngularSpectrumPropagator(parameters_factory(distance))
        propagated = propagator.propagate_forward(wavefield)
    propagated.abs().pow(2).sum().backward()
    return distance.grad


def test_distance_gradient_is_finite_with_evanescent_modes():
    """Regression: the NaN from sqrt of a negative leaked through exp as a value, and
    torch.where propagates that into the gradient even though it discards the output."""
    gradient = _distance_gradient(_evanescent_params)

    assert gradient is not None
    assert torch.isfinite(gradient)


def test_distance_gradient_is_finite_through_update():
    """`update` is the multislice path (forward_models.propagate_to_previous_slice), which
    the plain forward test does not cover."""
    gradient = _distance_gradient(_evanescent_params, use_update=True)

    assert gradient is not None
    assert torch.isfinite(gradient)


def test_distance_gradient_is_nonzero_without_evanescent_modes():
    """Control: the clamp must not flatten the ordinary gradient."""

    def parameters_factory(propagation_distance_m):
        return _params(propagation_distance_m, wavelength_m=1e-10, pixel_width_m=1e-6)

    gradient = _distance_gradient(parameters_factory)

    assert torch.isfinite(gradient)
    assert float(gradient) != 0.0


# --------------------------------------------------------------------------------------
# Analytic references
# --------------------------------------------------------------------------------------

_GAUSSIAN_WAVELENGTH_M = 6e-7
_GAUSSIAN_NUM_PX = 256
_GAUSSIAN_PIXEL_M = 2e-6
_GAUSSIAN_WAIST_M = 12e-6
_GAUSSIAN_RAYLEIGH_M = math.pi * _GAUSSIAN_WAIST_M**2 / _GAUSSIAN_WAVELENGTH_M


def _gaussian_beam_setup():
    coordinates = _centered_grid(_GAUSSIAN_NUM_PX, _GAUSSIAN_PIXEL_M)
    yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
    radius_squared = xx**2 + yy**2
    wavefield = torch.exp(-radius_squared / _GAUSSIAN_WAIST_M**2).to(torch.complex128)
    return wavefield, radius_squared


@pytest.mark.parametrize("z_over_rayleigh", [0.25, 0.5, 1.0, 2.0])
def test_angular_spectrum_matches_the_gaussian_beam_solution(z_over_rayleigh):
    """Waist, on-axis amplitude and Gouy phase against the closed form. The residual is
    the paraxial-versus-exact difference, not numerical error, so these tolerances are
    stable rather than tuned."""
    propagation_distance_m = z_over_rayleigh * _GAUSSIAN_RAYLEIGH_M
    parameters = _params(
        propagation_distance_m,
        wavelength_m=_GAUSSIAN_WAVELENGTH_M,
        width_px=_GAUSSIAN_NUM_PX,
        height_px=_GAUSSIAN_NUM_PX,
        pixel_width_m=_GAUSSIAN_PIXEL_M,
    )
    wavefield, radius_squared = _gaussian_beam_setup()

    propagated = AngularSpectrumPropagator(parameters).propagate_forward(wavefield)

    center = _GAUSSIAN_NUM_PX // 2
    expected_waist_m = _GAUSSIAN_WAIST_M * math.sqrt(1 + z_over_rayleigh**2)
    intensity = propagated.abs() ** 2
    measured_waist_m = float(
        torch.sqrt(2 * (intensity * radius_squared).sum() / intensity.sum())
    )
    measured_amplitude = float(propagated[center, center].abs())
    expected_phase = (
        2 * math.pi * propagation_distance_m / _GAUSSIAN_WAVELENGTH_M
        - math.atan2(propagation_distance_m, _GAUSSIAN_RAYLEIGH_M)
    )
    phase_error = float(torch.angle(propagated[center, center])) - expected_phase
    phase_error = (phase_error + math.pi) % (2 * math.pi) - math.pi

    assert measured_amplitude == pytest.approx(
        _GAUSSIAN_WAIST_M / expected_waist_m, rel=5e-4
    )
    assert measured_waist_m == pytest.approx(expected_waist_m, rel=1e-3)
    assert phase_error == pytest.approx(0.0, abs=2e-3)


def test_fraunhofer_of_a_rect_aperture_matches_the_dirichlet_kernel():
    """Pins both the C0 prefactor and the centered-in/centered-out convention: against a
    corner-order forward this fails outright."""
    num_px = 128
    aperture_px = 9
    parameters = _params(100.0, width_px=num_px, height_px=num_px, pixel_width_m=1e-6)
    center = num_px // 2
    half = aperture_px // 2
    wavefield = torch.zeros(num_px, num_px, dtype=torch.complex128, device="cpu")
    wavefield[center - half : center + half + 1, center - half : center + half + 1] = 1

    propagated = FraunhoferPropagator(parameters).propagate_forward(wavefield)

    modes = _centered_grid(num_px)
    dirichlet = torch.where(
        modes == 0,
        torch.full_like(modes, float(aperture_px)),
        torch.sin(math.pi * aperture_px * modes / num_px) / torch.sin(math.pi * modes / num_px),
    )
    expected = abs(parameters.fresnel_number / parameters.pixel_aspect_ratio) * (
        dirichlet[:, None] * dirichlet[None, :]
    ).abs()

    torch.testing.assert_close(propagated.abs(), expected, rtol=1e-10, atol=1e-14)
    assert torch.unravel_index(propagated.abs().argmax(), propagated.shape) == (
        torch.tensor(center),
        torch.tensor(center),
    )


@pytest.mark.parametrize("propagation_distance_m", [0.64, -0.64])
def test_fresnel_transform_agrees_with_angular_spectrum_at_critical_sampling(
    propagation_distance_m,
):
    """At |Fr| == 1 / N the conjugate-plane pitch equals the source pitch, so the two
    methods live on the same grid and are directly comparable. Cross-validates the shift
    convention, the prefactor and the crossover point at once."""
    num_px = 64
    parameters = _params(propagation_distance_m, wavelength_m=1e-10,
                         width_px=num_px, height_px=num_px,
                         pixel_width_m=1e-6)
    coordinates = _centered_grid(num_px)
    yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
    wavefield = torch.exp(-((xx - 8)**2 + (yy - 4)**2) / 36.0).to(torch.complex128)

    angular_spectrum = AngularSpectrumPropagator(parameters).propagate_forward(wavefield)
    fresnel_transform = FresnelTransformPropagator(parameters).propagate_forward(wavefield)

    assert _relative_error(fresnel_transform, angular_spectrum) < 1e-5


@pytest.mark.parametrize("propagator_class", _SINGLE_FFT_PROPAGATORS)
@pytest.mark.parametrize(("width_px", "height_px"), [(64, 48), (63, 65)])
def test_single_fft_negative_distance_obeys_conjugation_symmetry(
    propagator_class, width_px, height_px,
):
    """Reversing distance conjugates the diffraction kernel on the same oriented grid."""
    positive = propagator_class(_params(
        1.0, width_px=width_px, height_px=height_px,
        pixel_width_m=50e-6, pixel_height_m=40e-6,
    ))
    negative = propagator_class(_params(
        -1.0, width_px=width_px, height_px=height_px,
        pixel_width_m=50e-6, pixel_height_m=40e-6,
    ))
    wavefield = _random_wavefield((2, 3, height_px, width_px))

    torch.testing.assert_close(
        negative.propagate_forward(wavefield.conj()),
        positive.propagate_forward(wavefield).conj(),
        rtol=1e-11, atol=1e-11,
    )


@pytest.mark.parametrize("propagator_class", _SINGLE_FFT_PROPAGATORS)
def test_single_fft_output_is_centered(propagator_class):
    num_px = 64
    parameters = _params(1.0, width_px=num_px, height_px=num_px)
    coordinates = _centered_grid(num_px)
    yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
    wavefield = torch.exp(-(xx**2 + yy**2) / 8.0).to(torch.complex128)

    propagated = propagator_class(parameters).propagate_forward(wavefield)

    peak = torch.unravel_index(propagated.abs().argmax(), propagated.shape)
    assert (int(peak[0]), int(peak[1])) == (num_px // 2, num_px // 2)


# --------------------------------------------------------------------------------------
# Shapes and batching
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("propagator_class", _PARAMETERIZED_PROPAGATORS)
def test_propagators_preserve_batched_shape_and_dtype(propagator_class):
    parameters = _params(1.0, width_px=64, height_px=48, pixel_width_m=1e-6,
                         pixel_height_m=0.8e-6)
    propagator = propagator_class(parameters)
    wavefield = _random_wavefield((5, 7, 48, 64))

    for propagated in (
        propagator.propagate_forward(wavefield),
        propagator.propagate_backward(wavefield),
    ):
        assert propagated.shape == (5, 7, 48, 64)
        assert propagated.dtype == torch.complex128


@pytest.mark.parametrize("propagator_class", _PARAMETERIZED_PROPAGATORS)
def test_propagation_is_independent_across_batch_and_mode(propagator_class):
    """The operator acts per (batch, mode) slice, so an isolated nonzero slice must stay
    isolated and must match propagating that slice alone."""
    parameters = _params(1.0, width_px=64, height_px=48, pixel_width_m=1e-6,
                         pixel_height_m=0.8e-6)
    propagator = propagator_class(parameters)
    wavefield = torch.zeros(2, 2, 48, 64, dtype=torch.complex128, device="cpu")
    wavefield[0, 0] = _random_wavefield((48, 64))

    propagated = propagator.propagate_forward(wavefield)

    assert torch.count_nonzero(propagated[1]) == 0
    assert torch.count_nonzero(propagated[0, 1]) == 0
    torch.testing.assert_close(
        propagated[0:1, 0:1],
        propagator.propagate_forward(wavefield[0:1, 0:1]),
        rtol=1e-12,
        atol=1e-14,
    )
