"""Compare sampling against an image_proc.py snapshot in the same process.

Example: uv run python tests/benchmark_patch_sampling.py --device cuda \
    --reference /tmp/ptychi_interp_baseline.py

Timings include output allocation, use two CPU threads, and alternate reference
and current implementations. CUDA peaks are additional live tensor bytes after
warm-up. Save a snapshot before editing; no reference implementation is bundled.
"""

import argparse
import importlib.util
import json
import statistics

import torch
from torch.utils.benchmark import Timer

import ptychi.image_proc as ip


def operations(module, operator, method, image, positions, patches):
    if operator == "image":
        shifts = positions - 127.5

        def extract():
            return module.shift_images(patches, shifts, method=method, pad=1)

        def place():
            return module.shift_images(patches, shifts, method=method, pad=1, adjoint=True)
    elif module is ip:
        sampler = module.ObjectPatchSampler(operator=operator, method=method, pad=1)

        def extract():
            return sampler.extract(image, positions, patches.shape[-2:])

        def place():
            return sampler.place(torch.zeros_like(image), positions, patches)
    elif operator == "full_object":

        def extract():
            return module.extract_patches_full_object(
                image, positions, patches.shape[-2:], method=method
            )

        def place():
            return module.place_patches_full_object(
                torch.zeros_like(image), positions, patches, method=method
            )
    else:
        # Pre-refactor snapshots use their original method-specific functions.
        name = "fourier" if method == "fourier" else "bilinear"
        extractor = getattr(module, f"extract_patches_{name}_shift")
        placer = getattr(module, f"place_patches_{name}_shift")
        kwargs = {} if method == "fourier" else {"round_positions": method == "nearest"}

        def extract():
            return extractor(image, positions, patches.shape[-2:], **kwargs)

        def place():
            return placer(torch.zeros_like(image), positions, patches, **kwargs)

    return {"extract": extract, "place": place}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 8, 32])
    parser.add_argument("--sizes", nargs="+", type=int, default=[32, 128])
    parser.add_argument("--methods", nargs="+", default=["fourier", "bilinear", "nearest"])
    parser.add_argument("--operators", nargs="+", default=["patch", "full_object"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--min-time", type=float, default=0.15)
    parser.add_argument("--grad", action="store_true", help="Include input-gradient computation.")
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("sampling_reference", args.reference)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    torch.set_num_threads(2)
    torch.set_default_device(args.device)
    torch.manual_seed(42)
    with torch.set_grad_enabled(args.grad):
        for size in args.sizes:
            for batch in args.batches:
                image = torch.randn(256, 256, dtype=torch.complex64)
                positions = 127.5 + 8 * torch.rand(batch, 2)
                patches = torch.randn(batch, size, size, dtype=image.dtype)
                image.requires_grad_(args.grad)
                patches.requires_grad_(args.grad)
                for operator in args.operators:
                    for method in args.methods:
                        if operator == "image" and method == "nearest":
                            continue  # The reference image-shift dispatcher lacks nearest.
                        calls = {
                            name: operations(module, operator, method, image, positions, patches)
                            for name, module in (("reference", reference), ("current", ip))
                        }
                        for operation in ("extract", "place"):
                            measurements = {name: [] for name in calls}
                            memory = {}
                            for repeat in range(args.repeats):
                                order = list(calls) if repeat % 2 == 0 else list(reversed(calls))
                                for name in order:
                                    call = calls[name][operation]
                                    if args.grad:
                                        source = (
                                            image
                                            if operation == "extract" and operator != "image"
                                            else patches
                                        )
                                        cotangent = (
                                            patches
                                            if operation == "extract" or operator == "image"
                                            else image
                                        )
                                        raw_call = call

                                        def call():
                                            return torch.autograd.grad(
                                                raw_call(), source, cotangent
                                            )[0]

                                    call()
                                    if args.device == "cuda":
                                        torch.cuda.synchronize()
                                        torch.cuda.reset_peak_memory_stats()
                                        allocated = torch.cuda.memory_allocated()
                                        call()
                                        torch.cuda.synchronize()
                                        memory[name] = torch.cuda.max_memory_allocated() - allocated
                                    result = Timer("call()", globals={"call": call}, num_threads=2)
                                    measurements[name].append(
                                        result.blocked_autorange(min_run_time=args.min_time).median
                                        * 1000
                                    )
                            medians = {
                                name: statistics.median(v) for name, v in measurements.items()
                            }
                            print(
                                json.dumps(
                                    dict(
                                        device=args.device,
                                        size=size,
                                        batch=batch,
                                        operator=operator,
                                        method=method,
                                        operation=operation,
                                        autograd=args.grad,
                                        milliseconds=medians,
                                        ratio=medians["current"] / medians["reference"],
                                        samples=measurements,
                                        peak_extra_bytes=memory,
                                    )
                                ),
                                flush=True,
                            )


if __name__ == "__main__":
    main()
