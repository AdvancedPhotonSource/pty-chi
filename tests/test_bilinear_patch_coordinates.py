import pytest
import torch

import ptychi.image_proc as ip


@pytest.fixture(autouse=True)
def cpu_device():
    with torch.device("cpu"):
        yield


@pytest.mark.parametrize("shape", [(4, 6), (5, 7), (4, 7), (5, 6)])
@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
def test_bilinear_matches_fourier_at_pixel_aligned_centers(shape, dtype):
    generator = torch.Generator().manual_seed(42)
    image = torch.randn(20, 24, dtype=dtype, generator=generator)
    starts = torch.tensor([[2, 3], [6, 8]])
    positions = starts + (torch.tensor(shape, dtype=torch.float64) - 1) / 2

    actual = ip.extract_patches_bilinear_shift(image, positions, shape)
    expected = ip.extract_patches_fourier_shift(image, positions, shape)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    patches = torch.randn(2, *shape, dtype=dtype, generator=generator)
    actual = ip.place_patches_bilinear_shift(torch.zeros_like(image), positions, patches)
    expected = ip.place_patches_fourier_shift(torch.zeros_like(image), positions, patches)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(4, 6), (5, 7), (4, 7), (5, 6)])
def test_bilinear_ramp_is_centered_at_requested_position(shape):
    y, x = torch.meshgrid(
        torch.arange(20, dtype=torch.float64),
        torch.arange(24, dtype=torch.float64),
        indexing="ij",
    )
    image = 3 * y + 5 * x
    positions = torch.tensor([[8.2, 10.7], [9.6, 12.1]], dtype=torch.float64)

    patches = ip.extract_patches_bilinear_shift(image, positions, shape)
    torch.testing.assert_close(patches.mean((-2, -1)), 3 * positions[:, 0] + 5 * positions[:, 1])
    torch.testing.assert_close(patches.diff(dim=-2), torch.full_like(patches[:, 1:, :], 3))
    torch.testing.assert_close(patches.diff(dim=-1), torch.full_like(patches[:, :, 1:], 5))


@pytest.mark.parametrize("shape", [(4, 6), (5, 7), (4, 7), (5, 6)])
@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
def test_bilinear_placement_is_extraction_adjoint(shape, dtype):
    generator = torch.Generator().manual_seed(43)
    image = torch.randn(20, 24, dtype=dtype, generator=generator, requires_grad=True)
    positions = torch.tensor([[8.2, 10.7], [9.6, 12.1], [8.2, 10.7]], dtype=torch.float64)
    patches = ip.extract_patches_bilinear_shift(image, positions, shape)
    cotangent = torch.randn(patches.shape, dtype=dtype, generator=generator)

    expected = torch.autograd.grad(patches, image, cotangent)[0]
    actual = ip.place_patches_bilinear_shift(torch.zeros_like(image), positions, cotangent)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        (patches.conj() * cotangent).sum(), (image.conj() * actual).sum()
    )
