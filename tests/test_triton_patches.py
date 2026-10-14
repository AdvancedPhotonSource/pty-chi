import os
import subprocess
import sys

import pytest
import torch

import ptychi._triton_patches as backend
import ptychi.image_proc as ip
import ptychi.maths as pmath


CUDA = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.cuda is None,
    reason="NVIDIA CUDA is unavailable",
)


@pytest.fixture(autouse=True)
def reset_backend(monkeypatch):
    monkeypatch.setattr(backend, "_disabled", False)
    monkeypatch.setattr(backend, "_accumulate", None)
    monkeypatch.delenv("PTYCHI_USE_TRITON", raising=False)
    previous = pmath.get_allow_nondeterministic_algorithms()
    yield
    pmath.set_allow_nondeterministic_algorithms(previous)


def _inputs(device, dtype=torch.float32, layout="ordinary"):
    generator = torch.Generator(device="cpu").manual_seed(413)
    image = torch.randn(19, 23, dtype=dtype, generator=generator, device="cpu")
    patches = torch.randn(40, 5, 7, dtype=dtype, generator=generator, device="cpu")
    # Duplicate, overlapping and boundary rectangles; non-square dimensions.
    origins = [(0, 0), (14, 16), (3, 4), (4, 5)] * 10
    if layout == "broadcast":
        patches = patches[:1].expand(40, -1, -1)
    reference = image.to(torch.complex128 if dtype.is_complex else torch.float64).clone()
    for patch, (y, x) in zip(patches, origins):
        reference[y:y + 5, x:x + 7] += patch.to(reference.dtype)
    image = image.to(device)
    patches = patches.to(device)
    if layout == "broadcast":
        patches = patches[:1].expand(40, -1, -1)
    elif layout == "strided":
        patches = patches.transpose(1, 2).contiguous().transpose(1, 2)
    positions = torch.tensor(origins, dtype=torch.int32, device=device)
    return image, patches, positions[:, 0], positions[:, 1], reference.to(dtype).to(device)


@CUDA
@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
@pytest.mark.parametrize("layout", ["ordinary", "strided", "broadcast"])
@pytest.mark.parametrize("deterministic", [False, True])
def test_triton_accumulation_matches_reference(dtype, layout, deterministic, monkeypatch):
    pytest.importorskip("triton")
    image, patches, sy, sx, expected = _inputs("cuda", dtype, layout)
    original = image.clone()
    pmath.set_allow_nondeterministic_algorithms(not deterministic)

    def unexpected_native(*args, **kwargs):
        raise AssertionError("The supported case should execute the Triton kernel")

    monkeypatch.setattr(ip, "_patch_linear_indices", unexpected_native)
    actual = ip.batch_put(image, patches, sy, sx)
    assert actual.data_ptr() == image.data_ptr()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    if deterministic:
        repeated = ip.batch_put(original, patches, sy, sx)
        assert torch.equal(actual, repeated)
    assert not backend._disabled


@CUDA
@pytest.mark.parametrize("failure", ["import", "compile", "execution"])
def test_triton_failure_retries_without_double_accumulation(failure, monkeypatch, caplog):
    image, patches, sy, sx, expected = _inputs("cuda")
    original = image.clone()
    calls = []

    def unavailable(*args, **kwargs):
        calls.append(failure)
        if failure == "import":
            raise ImportError("simulated unavailable Triton")
        # An unsuccessful attempt may have produced a partial private result.
        scratch = image.clone().add_(123)
        assert scratch.data_ptr() != image.data_ptr()
        if failure == "compile":
            raise RuntimeError("simulated compiler failure")
        return scratch

    if failure == "import":
        monkeypatch.setattr(backend.importlib, "import_module", unavailable)
    else:
        monkeypatch.setattr(backend, "_accumulate", unavailable)
    if failure == "execution":
        real_synchronize = torch.cuda.synchronize
        first = True

        def fail_once(*args, **kwargs):
            nonlocal first
            if first:
                first = False
                raise RuntimeError("simulated recoverable execution failure")
            return real_synchronize(*args, **kwargs)

        monkeypatch.setattr(torch.cuda, "synchronize", fail_once)
    pmath.set_allow_nondeterministic_algorithms(False)
    actual = ip.batch_put(image, patches, sy, sx, check_bounds=False)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    assert backend._disabled
    assert "using PyTorch" in caplog.text
    ip.batch_put(original, patches, sy, sx, check_bounds=False)
    assert calls == [failure]


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=CUDA)])
@pytest.mark.parametrize("reason", ["disabled", "dtype", "gradients", "set", "strided_image"])
def test_native_fallbacks(device, reason, monkeypatch):
    dtype = torch.float64 if reason == "dtype" else torch.float32
    image, patches, sy, sx, expected = _inputs(device, dtype)

    def unexpected_triton(*args, **kwargs):
        raise AssertionError("This case should use PyTorch")

    # A backend exception would otherwise be caught and silently fall back.
    monkeypatch.setattr(backend, "_accumulate", unexpected_triton)
    if reason == "disabled":
        monkeypatch.setenv("PTYCHI_USE_TRITON", "0")
    if reason == "gradients":
        patches.requires_grad_()
    if reason == "strided_image":
        image = image.T.contiguous().T
    pmath.set_allow_nondeterministic_algorithms(False)
    op = "set" if reason == "set" else "add"
    if op == "set":
        # One rectangle avoids overlapping-write semantics for set.
        patches, sy, sx = patches[:1], sy[:1], sx[:1]
        expected = image.clone()
        expected[:5, :7] = patches[0]
    actual = ip.batch_put(image, patches, sy, sx, op=op)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    assert not backend._disabled
    if reason == "gradients":
        actual.sum().backward()
        torch.testing.assert_close(patches.grad, torch.ones_like(patches))


@CUDA
def test_global_determinism_overrides_atomic_setting(monkeypatch):
    image, patches, sy, sx, _ = _inputs("cuda")
    modes = []

    def record_mode(image, patches, sy, sx, deterministic):
        modes.append(deterministic)
        return image.clone()

    monkeypatch.setattr(backend, "_accumulate", record_mode)
    pmath.set_allow_nondeterministic_algorithms(True)
    previous = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        ip.batch_put(image, patches, sy, sx)
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=warn_only)
    assert modes == [True]


def test_import_and_cpu_accumulation_without_triton():
    script = """
import sys
import importlib.abc
class NoTriton(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname == 'triton' or fullname.startswith('triton.'):
            raise ImportError('Triton is unavailable')
sys.meta_path.insert(0, NoTriton())
import torch
import ptychi.image_proc as ip
image = torch.zeros(4, 5)
patches = torch.ones(1, 2, 3)
result = ip.batch_put(image, patches, torch.tensor([1]), torch.tensor([1]))
assert result.sum().item() == 6
assert 'ptychi._triton_patch_kernels' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True, env=os.environ.copy())
