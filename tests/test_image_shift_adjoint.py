from types import SimpleNamespace

import pytest
import torch

import ptychi.image_proc as ip
from ptychi.forward_models import PlanarPtychographyForwardModel
from ptychi.reconstructors.base import AnalyticalIterativePtychographyReconstructor


@pytest.mark.parametrize("method", ["bilinear", "fourier"])
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


@pytest.mark.parametrize("method", ["bilinear", "fourier"])
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
    shape = (2, 3 if first_mode_only else 1, 5, 6)
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
