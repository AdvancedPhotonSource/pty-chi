from types import SimpleNamespace

import pytest
import torch

import ptychi.api  # noqa: F401 -- Initialize API types before importing reconstructors.
from ptychi.propagate import FourierPropagator
from ptychi.reconstructors.bh import BHReconstructor
from ptychi.api.enums import ShiftOperators


def amplitude_loss(field, data, mask):
    amplitude = field.abs().square().sum(dim=1, keepdim=True).sqrt()
    return (mask * (amplitude - data).square()).sum()


@pytest.fixture
def random_complex():
    generator = torch.Generator(device="cpu").manual_seed(419)

    def sample(shape):
        return torch.randn(shape, generator=generator, dtype=torch.complex128, device="cpu")

    return sample


@pytest.fixture(params=[False, True], ids=["unmasked", "masked"])
def reconstructor(request):
    reconstructor = object.__new__(BHReconstructor)
    # Test the unsmoothed loss at nonzero amplitudes.
    reconstructor.eps = 0.0
    mask = torch.ones(2, 1, 3, 5, dtype=torch.bool, device="cpu")
    if request.param:
        mask[0, :, ::2, ::2] = False
        mask[1, :, 1, :] = False
    reconstructor.current_constrained_pixel_mask = mask
    return reconstructor


@pytest.fixture
def data():
    return torch.linspace(0.5, 1.5, 30, dtype=torch.float64, device="cpu").reshape(2, 1, 3, 5)


@pytest.mark.parametrize("n_modes", [1, 4])
@pytest.mark.parametrize("propagation", ["identity", "fourier"])
def test_gradientF_matches_autograd(reconstructor, random_complex, data, n_modes, propagation):
    wavefield = random_complex((2, n_modes, 3, 5)).requires_grad_()
    if propagation == "fourier":
        propagator = FourierPropagator()
        field = torch.fft.fft2(wavefield)
    else:
        propagator = SimpleNamespace(propagate_backward=lambda value: value)
        field = wavefield
    reconstructor.forward_model = SimpleNamespace(free_space_propagator=propagator)

    loss = amplitude_loss(field, data, reconstructor.current_constrained_pixel_mask)
    expected = torch.autograd.grad(loss, wavefield)[0]
    actual = reconstructor.gradientF(field.detach(), data)

    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("n_modes", [1, 4])
def test_hessianF_matches_autograd(reconstructor, random_complex, data, n_modes):
    field = random_complex((2, n_modes, 3, 5))
    directions = [random_complex(field.shape), random_complex(field.shape)]

    def loss_at_step(step):
        perturbed = field + step[0] * directions[0] + step[1] * directions[1]
        return amplitude_loss(perturbed, data, reconstructor.current_constrained_pixel_mask)

    expected = torch.autograd.functional.hessian(
        loss_at_step, torch.zeros(2, dtype=torch.float64, device="cpu")
    )
    # Check both diagonal curvatures and both orders of the mixed bilinear form.
    for i in range(2):
        for j in range(2):
            actual = reconstructor.hessianF(field, directions[i], directions[j], data)
            torch.testing.assert_close(actual, expected[i, j], rtol=1e-11, atol=1e-11)


def test_hessianF_couples_distinct_probe_modes(reconstructor, random_complex, data):
    field = random_complex((2, 4, 3, 5))
    direction1 = torch.zeros_like(field)
    direction2 = torch.zeros_like(field)
    direction1[:, 0] = field[:, 0]
    direction2[:, 1] = field[:, 1]

    def loss_at_step(step):
        perturbed = field + step[0] * direction1 + step[1] * direction2
        return amplitude_loss(perturbed, data, reconstructor.current_constrained_pixel_mask)

    expected = torch.autograd.functional.hessian(
        loss_at_step, torch.zeros(2, dtype=torch.float64, device="cpu")
    )[0, 1]
    # Disjoint mode support removes the diagonal term; only cross-mode coupling remains.
    assert expected > 0.1
    actual = reconstructor.hessianF(field, direction1, direction2, data)
    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)


@pytest.fixture
def patch_reconstructor():
    dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    reconstructor = object.__new__(BHReconstructor)
    reconstructor.forward_model = SimpleNamespace(resolved_shift_operator=ShiftOperators.PATCH)
    yield reconstructor
    torch.set_default_dtype(dtype)


@pytest.mark.parametrize("size", [6, 7])
@pytest.mark.parametrize("second", [None, "same", "mixed"])
def test_patch_derivatives_match_legacy_square_formulas(
    patch_reconstructor, random_complex, size, second
):
    patches = random_complex((2, 2, size, size))
    direction = torch.tensor([[0.3, -0.8], [-0.2, 0.5]], device="cpu")
    other = None if second is None else (
        direction if second == "same" else direction.flip(-1)
    )
    # These are the square-patch Fourier formulas used before the sampling refactor.
    frequency = torch.fft.fftfreq(size, device="cpu")
    xi2, xi1 = torch.meshgrid(frequency, frequency, indexing="xy")
    dy, dx = direction[:, 0, None, None, None], direction[:, 1, None, None, None]
    spectrum = torch.fft.fft2(patches)
    if other is None:
        expected = 2 * torch.pi * 1j * torch.fft.ifft2((xi1 * dy + xi2 * dx) * spectrum)
    else:
        ey, ex = other[:, 0, None, None, None], other[:, 1, None, None, None]
        multiplier = xi1**2 * dy * ey + xi1 * xi2 * (dy * ex + dx * ey) + xi2**2 * dx * ex
        expected = -4 * torch.pi**2 * torch.fft.ifft2(multiplier * spectrum)
    actual = patch_reconstructor.position_directional_derivative(
        patches, torch.zeros_like(direction), direction, second_direction=other
    )
    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("shape", [(7, 10), (8, 11)])
@pytest.mark.parametrize("second", [None, "same", "mixed"])
def test_patch_derivatives_match_analytic_rectangular_waves(
    patch_reconstructor, random_complex, shape, second
):
    height, width = shape
    y = torch.arange(height, device="cpu")[:, None]
    x = torch.arange(width, device="cpu")[None, :]
    direction = torch.tensor([[1., 0.], [0.3, -0.8]], device="cpu")
    other = None if second is None else (
        direction if second == "same" else torch.tensor([[0., 1.], [-0.2, 0.5]], device="cpu")
    )
    coefficients = random_complex((2, 2, 2))
    patches = torch.zeros((2, 2, height, width), dtype=torch.complex128, device="cpu")
    expected = torch.zeros_like(patches)
    # Different signed frequencies on each axis detect swapped (y, x) components
    # and accidental normalization by the width on both axes.
    for i, (ky, kx) in enumerate(((1, -2), (-2, 1))):
        wave = torch.exp(2j * torch.pi * (ky * y / height + kx * x / width))
        term = coefficients[:, :, i, None, None] * wave
        multiplier = 2j * torch.pi * (ky * direction[:, 0] / height + kx * direction[:, 1] / width)
        if other is not None:
            multiplier *= 2j * torch.pi * (ky * other[:, 0] / height + kx * other[:, 1] / width)
        patches += term
        expected += term * multiplier[:, None, None, None]
    actual = patch_reconstructor.position_directional_derivative(
        patches, torch.zeros_like(direction), direction, second_direction=other
    )
    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)
