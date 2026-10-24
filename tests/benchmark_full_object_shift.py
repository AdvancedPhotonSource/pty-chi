"""Benchmark patch and full-object Fourier sampling in isolated processes.

Example: python tests/benchmark_full_object_shift.py --device cpu --size 512
CPU memory is process peak RSS (including imports); CUDA memory is peak allocated
tensor memory. Each row runs in a fresh process so peaks are independent.
"""

import argparse
import json
import multiprocessing
import resource

import torch
from torch.utils.benchmark import Timer

import ptychi.image_proc as ip


def measure(queue, device, size, ratio, batch, operator):
    torch.set_num_threads(2)
    torch.manual_seed(42)
    shape = (round(size * ratio), round(size * ratio))
    image = torch.randn(size, size, device=device, dtype=torch.complex64)
    positions = (size - 1) / 2 + torch.rand(batch, 2, device=device) * 4 - 2
    patches = torch.randn(batch, *shape, device=device, dtype=torch.complex64)
    if operator == "full_object":
        extract, place = ip.extract_patches_full_object, ip.place_patches_full_object
    else:
        extract, place = ip.extract_patches_fourier_shift, ip.place_patches_fourier_shift
    arguments = dict(image=image, positions=positions, shape=shape, patches=patches,
                     extract=extract, place=place, torch=torch)
    with torch.no_grad():
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        forward = Timer("extract(image, positions, shape)", globals=arguments, num_threads=2)
        adjoint = Timer("place(torch.zeros_like(image), positions, patches)", globals=arguments, num_threads=2)
        forward_ms = forward.blocked_autorange(min_run_time=0.3).median * 1000
        adjoint_ms = adjoint.blocked_autorange(min_run_time=0.3).median * 1000
        peak = (
            torch.cuda.max_memory_allocated() / 1024**2
            if device == "cuda" else resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        )
    queue.put(dict(device=device, object_size=size, ratio=ratio, batch=batch,
                   operator=operator, forward_ms=forward_ms, adjoint_ms=adjoint_ms,
                   peak_memory_mib=peak))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--ratios", type=float, nargs="+", default=[0.5, 0.85, 1., 1.2])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32])
    args = parser.parse_args()
    context = multiprocessing.get_context("spawn")
    for ratio in args.ratios:
        for batch in args.batch_sizes:
            for operator in ("patch", "full_object"):
                queue = context.Queue()
                process = context.Process(target=measure, args=(
                    queue, args.device, args.size, ratio, batch, operator,
                ))
                process.start()
                process.join()
                if process.exitcode:
                    raise RuntimeError(f"Benchmark worker exited with {process.exitcode}")
                print(json.dumps(queue.get()), flush=True)


if __name__ == "__main__":
    main()
