import pytest
import torch

import ptychi.image_proc as ip


DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is unavailable"
))]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64, torch.int64])
@pytest.mark.parametrize("strided", [False, True])
def test_batch_slice_matches_rectangles(device, dtype, strided):
    image = torch.arange(15 * 19, device=device).reshape(15, 19).to(dtype)
    if dtype.is_complex:
        image = image + 1j * image.flip(0)
    if strided:
        image = image.T
    patch_shape = (4, 7)
    origins = [(0, 0), (3, 2), (3, 2), (image.shape[0] - 4, image.shape[1] - 7)]
    positions = torch.tensor(origins, device=device, dtype=torch.int32)
    actual = ip.batch_slice(image, positions[:, 0], positions[:, 1], patch_shape)
    expected = torch.stack([image[y:y + 4, x:x + 7] for y, x in origins])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
def test_batch_slice_preserves_gradients_without_unfold(device, dtype, monkeypatch):
    image = torch.randn(9, 11, dtype=dtype, device=device, requires_grad=True)
    origins = [(0, 0), (2, 3), (2, 3), (6, 7)]
    positions = torch.tensor(origins, device=device)

    def unexpected_unfold(*args, **kwargs):
        raise AssertionError("Differentiable extraction must avoid the unfolded gradient")

    monkeypatch.setattr(torch.Tensor, "unfold", unexpected_unfold)
    actual = ip.batch_slice(image, positions[:, 0], positions[:, 1], (3, 4))
    actual.abs().square().sum().backward()
    expected_grad = torch.zeros_like(image)
    for y, x in origins:
        expected_grad[y:y + 3, x:x + 4] += 2 * image.detach()[y:y + 3, x:x + 4]
    torch.testing.assert_close(image.grad, expected_grad)


def test_batch_slice_uses_unfold_with_no_grad(monkeypatch):
    image = torch.randn(8, 9, requires_grad=True, device="cpu")
    sy = torch.tensor([0, 3], device="cpu")
    sx = torch.tensor([0, 4], device="cpu")

    def unexpected_indices(*args, **kwargs):
        raise AssertionError("Extraction without gradients should use the window view")

    monkeypatch.setattr(ip, "_patch_linear_indices", unexpected_indices)
    with torch.no_grad():
        actual = ip.batch_slice(image, sy, sx, (3, 4))
    torch.testing.assert_close(actual, torch.stack([image[:3, :4], image[3:6, 4:8]]))


@pytest.mark.parametrize("origin", [(-1, 0), (0, -1), (6, 0), (0, 7)])
def test_batch_slice_checks_bounds(origin):
    image = torch.zeros(8, 9, device="cpu")
    sy = torch.tensor([origin[0]], device="cpu")
    sx = torch.tensor([origin[1]], device="cpu")
    with pytest.raises(ValueError, match="out of bounds"):
        ip.batch_slice(image, sy, sx, (3, 4))
