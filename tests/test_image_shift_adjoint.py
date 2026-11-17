from types import SimpleNamespace

import pytest
import torch

import ptychi.image_proc as ip
from ptychi.forward_models import PlanarPtychographyForwardModel
from ptychi.reconstructors.base import AnalyticalIterativePtychographyReconstructor


@pytest.mark.parametrize("method", ["bilinear", "fourier", "nearest", "cubic"])
@pytest.mark.parametrize("pad", [None, 0, 1, 3])
@pytest.mark.parametrize("shape", [(2, 5, 6), (2, 1, 1)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.complex64, torch.complex128])
def test_image_shift_adjoint(method, pad, shape, dtype):
    generator = torch.Generator(device="cpu").manual_seed(77)
    image = torch.randn(shape, dtype=dtype, device="cpu", generator=generator, requires_grad=True)
    cotangent = torch.randn(shape, dtype=dtype, device="cpu", generator=generator)
    shifts = torch.tensor([[0.3, -0.4], [-1.2, 1.7]], dtype=image.real.dtype, device="cpu")

    shifted = ip.shift_images(image, shifts, method=method, pad=pad)
    adjoint = ip.shift_images(cotangent, shifts, method=method, pad=pad, adjoint=True)
    (expected,) = torch.autograd.grad(shifted, image, cotangent)

    torch.testing.assert_close(adjoint, expected, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(
        (shifted.conj() * cotangent).sum(),
        (image.conj() * adjoint).sum(),
        rtol=2e-6,
        atol=2e-6,
    )


SHIFTS_FOR_STACK = [[0.3, -0.4], [-1.2, 1.7], [0.6, 0.05]]


def shift_flattened_stack(images, shifts, **kwargs):
    """Shift a (N, M, H, W) stack the way it was done before broadcasting: as N * M images."""
    n, m = images.shape[:2]
    return ip.shift_images(
        images.reshape(n * m, *images.shape[2:]), shifts.repeat_interleave(m, dim=0), **kwargs
    ).reshape(images.shape)


@pytest.mark.parametrize("method", ["bilinear", "fourier", "nearest", "cubic"])
@pytest.mark.parametrize("pad", [0, 1])
@pytest.mark.parametrize("adjoint", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64, torch.complex128])
def test_image_shift_shares_each_shift_across_middle_dims(method, pad, adjoint, dtype):
    generator = torch.Generator(device="cpu").manual_seed(79)
    images = torch.randn((3, 4, 6, 5), dtype=dtype, device="cpu", generator=generator)
    original = images.clone()
    shifts = torch.tensor(SHIFTS_FOR_STACK, dtype=images.real.dtype, device="cpu")

    actual = ip.shift_images(images, shifts, method=method, pad=pad, adjoint=adjoint)
    expected = shift_flattened_stack(images, shifts, method=method, pad=pad, adjoint=adjoint)

    # CPU kernels may round the loop remainder of a broadcast multiply differently (~1 ulp).
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    assert torch.equal(images, original)


@pytest.mark.parametrize("method", ["fourier", "cubic"])
def test_image_shift_gradients_with_middle_dims(method):
    generator = torch.Generator(device="cpu").manual_seed(80)
    images = torch.randn((3, 4, 6, 5), dtype=torch.complex128, device="cpu", generator=generator)
    cotangent = torch.randn(images.shape, dtype=torch.complex128, device="cpu", generator=generator)
    shifts = torch.tensor(SHIFTS_FOR_STACK, dtype=torch.float64, device="cpu")

    def gradients(shift):
        x = images.clone().requires_grad_()
        s = shifts.clone().requires_grad_()
        return torch.autograd.grad(shift(x, s, method=method), (x, s), cotangent)

    expected = gradients(shift_flattened_stack)
    for actual, wanted in zip(gradients(ip.shift_images), expected):
        torch.testing.assert_close(actual, wanted, rtol=1e-10, atol=1e-10)


def test_fourier_image_shift_keeps_type_promotion():
    # A float64 default dtype gives a complex128 multiplier; multiplying complex64 images by it
    # in place would round the result to complex64 instead of promoting it.
    images = torch.ones((2, 3, 4, 4), dtype=torch.complex64, device="cpu")
    shifts = torch.tensor(SHIFTS_FOR_STACK[:2], dtype=torch.float64, device="cpu")
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        shifted = ip.shift_images(images, shifts, method="fourier")
    finally:
        torch.set_default_dtype(default_dtype)
    assert shifted.dtype == torch.complex128


@pytest.mark.parametrize("method", ["bilinear", "fourier", "nearest", "cubic"])
@pytest.mark.parametrize("pad", [0, 1, 3])
@pytest.mark.parametrize("first_mode_only", [False, True])
def test_probe_shift_adjoint_uses_forward_padding(method, pad, first_mode_only):
    shifts = torch.tensor([[0.3, -0.4], [0.1, 0.2], [-0.2, 0.4]], device="cpu")
    parameter_group = SimpleNamespace(
        object=SimpleNamespace(options=SimpleNamespace(patch_interpolation_method=method))
    )
    forward_model = SimpleNamespace(
        parameter_group=parameter_group,
        pad_for_shift=pad,
        get_probe_grid_position_data=lambda: (None, None, shifts),
    )
    reconstructor = SimpleNamespace(parameter_group=parameter_group, forward_model=forward_model)
    indices = torch.tensor([2, 0], device="cpu")
    generator = torch.Generator(device="cpu").manual_seed(78)
    shape = (2, 3, 5, 6)
    probe = torch.randn(
        shape, dtype=torch.complex128, device="cpu", generator=generator, requires_grad=True
    )
    cotangent = torch.randn(shape, dtype=torch.complex128, device="cpu", generator=generator)

    shifted = PlanarPtychographyForwardModel.shift_unique_probes(
        forward_model, indices, probe, first_mode_only=first_mode_only
    )
    (expected,) = torch.autograd.grad(shifted, probe, cotangent)
    actual = AnalyticalIterativePtychographyReconstructor.adjoint_shift_probe_update_direction(
        reconstructor, indices, cotangent, first_mode_only=first_mode_only
    )
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
