import pytest
import torch
import torch.nn.functional as F

import ptychi.image_proc as ip
import ptychi.interpolation as interpolation


DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is unavailable"
))]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64, torch.complex128])
@pytest.mark.parametrize("batch", [1, 5])
@pytest.mark.parametrize("strided", [False, True])
def test_bilinear_sampling_matches_grid_sample(device, dtype, batch, strided):
    generator = torch.Generator(device="cpu").manual_seed(44)
    image = torch.randn(17, 19, dtype=dtype, generator=generator).to(device)
    if strided:
        image = image.T
    image.requires_grad_()
    positions = torch.tensor([[7.3, 8.7]], dtype=image.real.dtype, device=device).repeat(batch, 1)
    positions.requires_grad_()
    shape = (5, 6)
    y, x = torch.meshgrid(
        torch.arange(shape[0], device=device) - (shape[0] - 1) / 2,
        torch.arange(shape[1], device=device) - (shape[1] - 1) / 2, indexing="ij",
    )
    grid = torch.stack((
        2 * (x + positions[:, 1, None, None]) / (image.shape[1] - 1) - 1,
        2 * (y + positions[:, 0, None, None]) / (image.shape[0] - 1) - 1,
    ), -1)
    channels = torch.stack((image.real, image.imag)) if image.is_complex() else image[None]
    sampled = F.grid_sample(channels[None].expand(batch, -1, -1, -1), grid, align_corners=True)
    expected = torch.complex(sampled[:, 0], sampled[:, 1]) if image.is_complex() else sampled[:, 0]
    sampler = ip.ObjectPatchSampler(method="bilinear")
    actual = sampler.extract(image, positions, shape)
    tol = 1e-5 if image.real.dtype == torch.float32 else 1e-10
    torch.testing.assert_close(actual, expected, rtol=tol, atol=tol)

    cotangent = torch.randn(batch, *shape, dtype=dtype, generator=generator).to(device)
    expected_gradients = torch.autograd.grad(expected, (image, positions), cotangent)
    actual_gradients = torch.autograd.grad(actual, (image, positions), cotangent)
    for observed, reference in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(observed, reference, rtol=tol, atol=tol)
    buffer = image.detach().clone()
    original = buffer.clone()
    placed = sampler.place(buffer, positions.detach(), cotangent)
    torch.testing.assert_close(placed, original + expected_gradients[0], rtol=tol, atol=tol)
    assert placed.data_ptr() == buffer.data_ptr()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("method", ["bilinear", "nearest"])
@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("patch_dtype,image_dtype", [
    (torch.float32, torch.complex64),
    (torch.float32, torch.float64),
    (torch.complex64, torch.complex128),
    (torch.complex128, torch.complex64),
])
def test_mixed_dtype_placement(device, method, batch, patch_dtype, image_dtype):
    generator = torch.Generator(device="cpu").manual_seed(45)
    image = torch.randn(17, 19, dtype=image_dtype, generator=generator).to(device)
    patches = torch.randn(batch, 5, 6, dtype=patch_dtype, generator=generator).to(device)
    positions = torch.tensor(
        [[7.3, 8.7], [8.1, 9.2], [7.8, 8.4]], dtype=torch.float64, device=device
    )[:batch]
    centers = positions.round() if method == "nearest" else positions
    expected = image.clone()
    # Reference the original per-patch accumulation into the destination dtype.
    for center, patch in zip(centers, patches):
        start = center - center.new_tensor([2, 2.5])
        y, x = start.int().tolist()
        fy, fx = start - start.int()
        for dy, dx, weight in (
            (0, 0, (1 - fy) * (1 - fx)), (0, 1, (1 - fy) * fx),
            (1, 0, fy * (1 - fx)), (1, 1, fy * fx),
        ):
            expected[y + dy:y + dy + 5, x + dx:x + dx + 6].add_(weight * patch)

    actual = ip.ObjectPatchSampler(method=method).place(image, positions, patches)
    tolerance = 16 * max(torch.finfo(patches.real.dtype).eps, torch.finfo(image.real.dtype).eps)
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    assert actual.data_ptr() == image.data_ptr()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape", [(5, 6), (6, 5)])
@pytest.mark.parametrize("method", ["fourier", "bilinear", "nearest", "cubic"])
def test_shared_patch_matches_expanded_batch(device, shape, method):
    # Include batch == patch height, which must still interpret a 2D input as shared.
    image = torch.zeros(20, 22, dtype=torch.complex64, device=device)
    positions = torch.tensor([[8.3, 9.7]], device=device).repeat(shape[0], 1)
    patch = torch.ones(shape, dtype=image.dtype, device=device)
    sampler = ip.ObjectPatchSampler(method=method, pad=2)
    actual = sampler.place(image.clone(), positions, patch)
    expected = sampler.place(image.clone(), positions, patch[None].expand(shape[0], -1, -1))
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("shape", [(5, 6), (6, 5)])
def test_local_nearest_preserves_center_rounding(shape):
    image = torch.randn(20, 22, device="cpu")
    positions = torch.tensor([[8.3, 9.7], [8.5, 10.5]], device="cpu")
    nearest = ip.ObjectPatchSampler(method="nearest")
    bilinear = ip.ObjectPatchSampler(method="bilinear")
    expected = bilinear.extract(image, positions.round(), shape)
    torch.testing.assert_close(nearest.extract(image, positions, shape), expected)
    torch.testing.assert_close(
        nearest.place(torch.zeros_like(image), positions, expected),
        bilinear.place(torch.zeros_like(image), positions.round(), expected),
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
def test_bilinear_batch_preserves_scalar_weight_precision(dtype):
    image = torch.randn(20, 22, dtype=dtype, device="cpu")
    position = torch.tensor([[8.3, 9.7]], dtype=torch.float64, device="cpu")
    sampler = ip.ObjectPatchSampler(method="bilinear")
    single = sampler.extract(image, position, (5, 6))
    batched = sampler.extract(image, position.repeat(4, 1), (5, 6))
    torch.testing.assert_close(batched, single.expand(4, -1, -1))
    assert batched.dtype == image.dtype


@pytest.mark.parametrize("method", ["fourier", "bilinear", "nearest", "cubic"])
def test_spatial_and_spectral_translation_agree_interior(method):
    image = torch.randn(2, 13, 15, dtype=torch.complex128, device="cpu")
    shifts = torch.tensor([[0.3, -0.4], [-1.2, 1.7]], dtype=torch.float64, device="cpu")
    spatial = interpolation.translate(image, shifts, method)
    # Use the image-shift frequency grid for the periodic Fourier comparison.
    multiplier = interpolation.translation_multiplier(
        shifts, image.shape[-2:], method,
        frequency_dtype=torch.get_default_dtype() if method == "fourier" else None,
    )
    spectral = torch.fft.ifft2(torch.fft.fft2(image) * multiplier)
    torch.testing.assert_close(spatial[:, 3:-3, 3:-3], spectral[:, 3:-3, 3:-3], rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("position_dtype", [torch.float32, torch.float64])
def test_fourier_shift_preserves_dense_phase_precision(device, position_dtype):
    image = torch.randn(2, 11, 12, dtype=torch.complex128, device=device)
    shifts = torch.tensor([[0.3, -0.4], [-1.2, 1.7]], dtype=position_dtype, device=device)
    fy, fx = torch.meshgrid(
        torch.fft.fftfreq(11, device=device), torch.fft.fftfreq(12, device=device), indexing="ij"
    )
    phase = fx[None] * shifts[:, 1, None, None] + fy[None] * shifts[:, 0, None, None]
    expected = ip.pmath.ifft2_precise(ip.pmath.fft2_precise(image) * torch.exp(-2j * torch.pi * phase))
    torch.testing.assert_close(ip.fourier_shift(image, shifts), expected, rtol=1e-12, atol=1e-12)


def test_image_shift_preserves_float32_upper_neighbor_rounding():
    image = torch.tensor([[[1., 2., 3.]]], dtype=torch.float32, device="cpu")
    shifts = torch.tensor([[0.3, 1e-10]], dtype=torch.float32, device="cpu")
    # Existing image shifts evaluate upper neighbors as floor(coordinate + 1).
    # In float32, -1e-10 + 1 rounds to 1 while floor(-1e-10) is -1.
    expected = image.new_tensor([[[3., 2., 3.]]])
    torch.testing.assert_close(ip.shift_images(image, shifts, "bilinear"), expected)
    expected_adjoint = image.new_tensor([[[1., 2., 1.]]])
    torch.testing.assert_close(
        ip.shift_images(torch.ones_like(image), shifts, "bilinear", adjoint=True),
        expected_adjoint,
    )


@pytest.mark.parametrize("method", ["bilinear", "cubic"])
def test_batched_sampling_uses_one_gather_and_accumulation(monkeypatch, method):
    calls = []
    batch_slice, batch_put = ip.batch_slice, ip.batch_put

    def gather(image, sy, sx, *args, **kwargs):
        calls.append(("extract", len(sy)))
        return batch_slice(image, sy, sx, *args, **kwargs)

    def accumulate(image, patches, sy, sx, *args, **kwargs):
        calls.append(("place", len(sy)))
        return batch_put(image, patches, sy, sx, *args, **kwargs)

    monkeypatch.setattr(ip, "batch_slice", gather)
    monkeypatch.setattr(ip, "batch_put", accumulate)
    image = torch.randn(30, 32, device="cpu")
    positions = torch.tensor([[10.2, 11.7]], device="cpu").repeat(16, 1)
    sampler = ip.ObjectPatchSampler(method=method)
    patches = sampler.extract(image, positions, (6, 7))
    sampler.place(torch.zeros_like(image), positions, patches)
    assert calls == [("extract", 16), ("place", 16)]
