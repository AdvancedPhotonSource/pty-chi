"""Independent references and adjoint checks for interpolating cubic B-splines."""

import numpy as np
import pytest
from scipy.ndimage import map_coordinates, spline_filter
import torch

import ptychi.api as api
import ptychi.image_proc as ip
import ptychi.interpolation as interpolation


DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is unavailable"
))]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.complex64, torch.complex128])
@pytest.mark.parametrize("operator", ["patch", "full_object"])
@pytest.mark.parametrize("source,shape", [((9, 10), (5, 6)), ((5, 6), (8, 9)), ((1, 1), (2, 3))])
def test_cubic_matches_scipy(device, dtype, operator, source, shape):
    generator = torch.Generator(device="cpu").manual_seed(173)
    image = torch.randn(source, dtype=dtype, generator=generator).to(device)
    positions = torch.tensor([[4.3, 5.7], [-2.4, 20.1], [2., 2.5]],
                             dtype=image.real.dtype, device=device)
    sampler = ip.ObjectPatchSampler(operator=operator, method=api.PatchInterpolationMethods.CUBIC)
    actual = sampler.extract(image, positions, shape)
    yy, xx = np.indices(shape, dtype=float)
    expected = np.stack([
        map_coordinates(image.cpu().numpy(),
                        [yy + y - (shape[0] - 1) / 2, xx + x - (shape[1] - 1) / 2],
                        order=3, mode="grid-wrap", prefilter=True)
        for y, x in positions.cpu().double().numpy()
    ])
    if operator == "full_object":
        # Full-object sampling pads its fixed crop when the patch is larger.
        mask = np.zeros(shape, dtype=bool)
        overlap = tuple(min(a, b) for a, b in zip(source, shape))
        start = tuple(p // 2 - n // 2 for p, n in zip(shape, overlap))
        mask[start[0]:start[0] + overlap[0], start[1]:start[1] + overlap[1]] = True
        expected[:, ~mask] = 0
    tolerance = 2e-5 if image.real.dtype == torch.float32 else 2e-12
    torch.testing.assert_close(actual, torch.as_tensor(expected, device=device),
                               atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
def test_prefilter_matches_scipy_and_integer_samples(dtype):
    image = torch.randn(7, 8, dtype=dtype, device="cpu")
    coefficients = interpolation.spline_coefficients(image)
    expected = spline_filter(image.numpy(), order=3, mode="grid-wrap", output=image.numpy().dtype)
    torch.testing.assert_close(coefficients, torch.from_numpy(expected))
    center = torch.tensor([[3., 3.5]], dtype=torch.float64, device="cpu")
    for operator in ("patch", "full_object"):
        actual = ip.ObjectPatchSampler(operator=operator, method="cubic").extract(
            image, center, image.shape
        )
        torch.testing.assert_close(actual[0], image, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.complex64, torch.complex128])
def test_local_cubic_adjoint(device, batch, strided, dtype):
    image = torch.randn(7, 8, dtype=dtype, device=device)
    if strided:
        image = image.T
    image.requires_grad_()
    positions = torch.tensor([[-1.2, 12.3], [4.1, 4.2], [4.3, 4.5]],
                             dtype=image.real.dtype, device=device)[:batch]
    sampler = ip.ObjectPatchSampler(method="cubic")
    patches = sampler.extract(image, positions, (5, 6))
    cotangent = torch.randn_like(patches)
    expected = torch.autograd.grad(patches, image, cotangent)[0]
    buffer = image.detach().clone()
    original = buffer.clone()
    placed = sampler.place(buffer, positions, cotangent)
    tolerance = 2e-5 if image.real.dtype == torch.float32 else 2e-12
    torch.testing.assert_close(placed, original + expected, rtol=tolerance, atol=tolerance)
    assert placed.data_ptr() == buffer.data_ptr()


@pytest.mark.parametrize("operator", ["patch", "full_object"])
def test_cubic_gradients_and_second_derivatives(operator):
    image = torch.randn(5, 6, dtype=torch.complex128, device="cpu", requires_grad=True)
    positions = torch.tensor([[2.2, 2.7], [-0.4, 7.3]], dtype=torch.float64,
                             device="cpu", requires_grad=True)
    sampler = ip.ObjectPatchSampler(operator=operator, method="cubic")
    def extract(image, positions):
        return sampler.extract(image, positions, (2, 3))

    assert torch.autograd.gradcheck(extract, (image, positions))
    assert torch.autograd.gradgradcheck(extract, (image, positions))
    patches = torch.randn(2, 2, 3, dtype=image.dtype, device="cpu", requires_grad=True)
    def place(patches, positions):
        return sampler.place(torch.zeros_like(image), positions, patches)

    assert torch.autograd.gradcheck(place, (patches, positions))


def test_cubic_is_twice_differentiable_at_integer_knots():
    image = torch.randn(5, 6, dtype=torch.complex128, device="cpu", requires_grad=True)
    positions = torch.tensor([[2., 2.5]], dtype=torch.float64, device="cpu", requires_grad=True)
    sampler = ip.ObjectPatchSampler(method="cubic")

    def extract(image, positions):
        return sampler.extract(image, positions, (3, 4))

    assert torch.autograd.gradcheck(extract, (image, positions))
    assert torch.autograd.gradgradcheck(extract, (image, positions))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("batch", [1, 3])
def test_local_cubic_mixed_dtype_placement(device, batch):
    image = torch.randn(7, 8, dtype=torch.complex128, device=device, requires_grad=True)
    positions = torch.tensor([[3.2, 4.7]], dtype=torch.float64, device=device).repeat(batch, 1)
    sampler = ip.ObjectPatchSampler(method="cubic")
    sampled = sampler.extract(image, positions, (3, 4))
    patches = torch.randn(batch, 3, 4, dtype=torch.float32, device=device)
    expected = torch.autograd.grad(sampled, image, patches.to(image.dtype))[0]
    actual = sampler.place(torch.zeros_like(image), positions, patches)
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("operator", ["patch", "full_object"])
def test_cubic_prefilter_is_shared_across_batch_and_recomputed(operator, monkeypatch):
    calls = []
    fft, ifft = ip.pmath.fft2_precise, ip.pmath.ifft2_precise
    monkeypatch.setattr(ip.pmath, "fft2_precise", lambda x: (calls.append(("fft", x.shape)), fft(x))[1])
    monkeypatch.setattr(ip.pmath, "ifft2_precise", lambda x: (calls.append(("ifft", x.shape)), ifft(x))[1])
    image = torch.randn(9, 10, dtype=torch.complex128, device="cpu")
    positions = torch.tensor([[3.2, 4.7]], device="cpu").repeat(16, 1)
    sampler = ip.ObjectPatchSampler(operator=operator, method="cubic")
    patches = sampler.extract(image, positions, (5, 6))
    assert [shape for op, shape in calls if op == "fft"] == [image.shape]
    calls.clear()
    sampler.place(torch.zeros_like(image), positions, patches)
    assert [shape for op, shape in calls if op == "ifft"] == [image.shape]
    image.mul_(2)
    torch.testing.assert_close(sampler.extract(image, positions, (5, 6)), 2 * patches)
