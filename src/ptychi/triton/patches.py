# Copyright © 2025 UChicago Argonne, LLC All right reserved
# Full license accessible at https://github.com//AdvancedPhotonSource/pty-chi/blob/main/LICENSE

"""Optional CUDA patch accumulation with a recoverable native fallback."""

import importlib
import logging
import os

import torch


logger = logging.getLogger(__name__)
_accumulate = None
_disabled = False


def try_accumulate_patches(image, patches, sy, sx, deterministic):
    """Return a completed result, or None to use PyTorch. Leave inputs unchanged.

    Triton is loaded only for supported NVIDIA CUDA tensors. A failed import,
    compilation, or execution disables this backend for the rest of the process.
    """
    global _accumulate, _disabled
    if _disabled or os.environ.get("PTYCHI_USE_TRITON", "1") == "0":
        return None
    if (
        not image.is_cuda
        or torch.version.cuda is None
        or image.layout != torch.strided
        or patches.layout != torch.strided
        or image.ndim != 2
        or image.numel() == 0
        or not image.is_contiguous()
        or image.dtype not in (torch.float32, torch.complex64)
        or patches.dtype != image.dtype
        or patches.device != image.device
        or patches.ndim != 3
        or patches.numel() == 0
        or sy.ndim != 1
        or sx.ndim != 1
        or sy.numel() == 0
        or patches.shape[0] != sy.numel()
        or sx.numel() != sy.numel()
        or sy.dtype not in (torch.int32, torch.int64)
        or sx.dtype not in (torch.int32, torch.int64)
        or sy.device != image.device
        or sx.device != image.device
        or image.is_conj()
        or image.is_neg()
        or patches.is_conj()
        or patches.is_neg()
        or (torch.is_grad_enabled() and (image.requires_grad or patches.requires_grad))
    ):
        return None
    components = 2 if image.is_complex() else 1
    patch_span = 1 + sum((size - 1) * stride for size, stride in zip(
        patches.shape, patches.stride()
    ))
    if max(image.numel(), patch_span) * components >= 2**31:
        # The kernels use 32-bit offsets; larger storage uses native indexing.
        return None
    try:
        if _accumulate is None:
            _accumulate = importlib.import_module(
                "ptychi.triton.patch_kernels"
            ).accumulate_patches
        with torch.cuda.device(image.device):
            result = _accumulate(image, patches, sy, sx, deterministic)
            # Complete the scratch result before the caller commits it to image.
            torch.cuda.synchronize(image.device)
        return result
    except Exception as exc:
        _disabled = True
        logger.warning("Triton patch accumulation failed; using PyTorch: %s", exc)
        return None
