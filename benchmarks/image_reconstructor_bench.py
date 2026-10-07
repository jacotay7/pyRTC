"""Per-frame latency of :class:`pyrtc.image_reconstructor.TorchModelRunner`.

Times the reconstructor's real-time path for one WFS image (NumPy int32
frame in, float32 signal on the host out): host staging, preprocessing,
transfers, the model and the device synchronisation. That is what
``TorchImageReconstructor.compute_signal`` spends between reading the
``wfs`` stream and writing ``signal``; stream handoffs are measured
separately by ``stream_handoff_bench.py`` and ``pipeline_latency_bench.py``.

Two models map a 64x64 image to 120 outputs:

- ``cnn``: five strided 3x3 convolutions and two linear layers, about 15M
  parameters;
- ``mlp``: 4096 -> 256 -> 256 -> 120, about 1.1M parameters.

Each runs on the CPU and, when CUDA is available, on the GPU eagerly and as
a CUDA graph, in float32 and float16.

Usage::

    python benchmarks/image_reconstructor_bench.py --frames 2000 --output recon.json
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from pyrtc.image_reconstructor import TorchModelRunner  # noqa: E402

IMAGE_SIDE = 64
NUM_OUTPUTS = 120


def build_cnn() -> torch.nn.Module:
    """A ~15M-parameter CNN: 64x64 image -> 120 outputs."""

    def block(c_in, c_out, stride):
        return [torch.nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1), torch.nn.ReLU()]

    return torch.nn.Sequential(
        *block(1, 64, 1),  # 64x64
        *block(64, 128, 2),  # 32x32
        *block(128, 256, 2),  # 16x16
        *block(256, 512, 2),  # 8x8
        *block(512, 512, 1),  # 8x8
        *block(512, 512, 2),  # 4x4
        torch.nn.Flatten(),
        torch.nn.Linear(512 * 4 * 4, 1024),
        torch.nn.ReLU(),
        torch.nn.Linear(1024, NUM_OUTPUTS),
    )


def build_mlp() -> torch.nn.Module:
    """A small MLP: flattened 64x64 image -> 120 outputs."""

    return torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(IMAGE_SIDE * IMAGE_SIDE, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, NUM_OUTPUTS),
    )


MODELS = {"cnn": build_cnn, "mlp": build_mlp}


def _stats_us(values) -> dict:
    arr = np.asarray(values, dtype=np.float64) * 1e6
    return {
        "count": int(arr.size),
        "mean_us": float(arr.mean()),
        "p50_us": float(np.percentile(arr, 50)),
        "p99_us": float(np.percentile(arr, 99)),
        "max_us": float(arr.max()),
    }


def bench_case(model_name, device, dtype, cuda_graph, frames, seed=0) -> dict:
    torch.manual_seed(seed)
    model = MODELS[model_name]()
    params = sum(p.numel() for p in model.parameters())
    runner = TorchModelRunner(
        model,
        image_shape=(IMAGE_SIDE, IMAGE_SIDE),
        image_dtype=np.int32,
        signal_size=NUM_OUTPUTS,
        device=device,
        dtype=dtype,
        flux_normalization="sum",
        cuda_graph=cuda_graph,
        warmup_iters=20,
    )
    rng = np.random.default_rng(seed)
    images = rng.integers(0, 4000, size=(16, IMAGE_SIDE, IMAGE_SIDE)).astype(np.int32)
    for i in range(50):  # warm-up through the full path
        runner.run(images[i % len(images)])
    times = np.empty(frames)
    for i in range(frames):
        image = images[i % len(images)]
        start = time.perf_counter()
        runner.run(image)
        times[i] = time.perf_counter() - start
    return {
        "model": model_name,
        "parameters": int(params),
        "device": str(runner.device),
        "dtype": dtype,
        "cuda_graph": runner.graph_active,
        **_stats_us(times),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--frames", type=int, default=2000, help="timed frames per GPU case")
    parser.add_argument("--cpu-frames", type=int, default=300, help="timed frames per CPU case")
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--no-cpu", action="store_true", help="skip the CPU cases")
    parser.add_argument(
        "--cpu-threads", type=int, default=None, help="torch.set_num_threads for CPU cases"
    )
    parser.add_argument("--output", type=Path, default=None, help="write a JSON report")
    args = parser.parse_args(argv)

    if args.cpu_threads:
        torch.set_num_threads(args.cpu_threads)
    cases = []
    for model_name in args.models:
        if not args.no_cpu:
            cases.append((model_name, "cpu", "float32", False, args.cpu_frames))
        if torch.cuda.is_available():
            for dtype in ("float32", "float16"):
                for graph in (False, True):
                    cases.append((model_name, "cuda", dtype, graph, args.frames))

    results = []
    header = f"{'model':<5} {'params':>7} {'device':<7} {'dtype':<8} {'graph':<5} {'median us':>10} {'p99 us':>9}"
    print(header)
    print("-" * len(header))
    for case in cases:
        result = bench_case(*case)
        results.append(result)
        print(
            f"{result['model']:<5} {result['parameters'] / 1e6:>6.1f}M {result['device']:<7} "
            f"{result['dtype']:<8} {str(result['cuda_graph']):<5} "
            f"{result['p50_us']:>10.1f} {result['p99_us']:>9.1f}",
            flush=True,
        )

    report = {
        "benchmark": "image_reconstructor",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "machine": platform.machine(),
        "cpu_threads": torch.get_num_threads(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "results": results,
    }
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
