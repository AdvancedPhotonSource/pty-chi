from types import SimpleNamespace

import pytest
import torch

import ptychi.api  # noqa: F401 -- Initialize API types before importing reconstructors.
from ptychi.propagate import FourierPropagator
from ptychi.reconstructors.bh import BHReconstructor


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
