"""Lightweight performance smoke checks for release validation.

This module combines a few fast-running timing checks with an optional pass over
the core compute benchmark suite. It is intended for CI and release workflows
that need coarse performance coverage without paying the cost of the full
benchmark stack.
"""

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
from benchmarks.core_compute_bench import run_core_compute_benchmarks

from pyRTC.logging_utils import add_logging_cli_args, configure_logging_from_args, get_logger
from pyRTC.scripts.measure_latency import compute_latency_seconds
from pyRTC.utils import measure_execution_time


logger = get_logger(__name__)


def _noop(_x):
    return None


def _safe_mean(values) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return 0.0
    return float(np.add.reduce(arr, dtype=np.float64) / arr.size)


def _safe_percentile(values, pct: float) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return 0.0
    sorted_vals = sorted(float(x) for x in arr)
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    rank = (pct / 100.0) * (len(sorted_vals) - 1)
    low = int(rank)
    high = min(low + 1, len(sorted_vals) - 1)
    if low == high:
        return sorted_vals[low]
    weight = rank - low
    return sorted_vals[low] * (1.0 - weight) + sorted_vals[high] * weight


def _benchmark_measure_execution_time(num_iters: int):
    start = time.perf_counter()
    median, iqr, ci1, ci99 = measure_execution_time(_noop, (1,), numIters=num_iters)
    elapsed = time.perf_counter() - start
    return {
        "median": float(median),
        "iqr": float(iqr),
        "ci1": float(ci1),
        "ci99": float(ci99),
        "elapsed_wall_s": float(elapsed),
    }


def _benchmark_latency_math(num_samples: int):
    source = np.linspace(0.0, 1.0, num_samples, dtype=np.float64)
    target = source + 0.001
    start = time.perf_counter()
    latency, frame_shift = compute_latency_seconds(source, target)
    elapsed = time.perf_counter() - start
    return {
        "mean_latency_s": _safe_mean(latency),
        "p99_latency_s": _safe_percentile(latency, 99),
        "frame_shift": int(frame_shift),
        "elapsed_wall_s": float(elapsed),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run lightweight pyRTC performance smoke checks.")
    parser.add_argument("--output", type=str, default="benchmarks/perf_smoke_report.json", help="JSON output path")
    parser.add_argument("--num-iters", type=int, default=200, help="Iterations for measure_execution_time benchmark")
    parser.add_argument("--num-samples", type=int, default=10000, help="Sample size for latency math benchmark")
    parser.add_argument("--skip-core", action="store_true", help="Skip core compute benchmark section")
    parser.add_argument("--core-iterations", type=int, default=1000, help="Timed iterations per core compute kernel")
    parser.add_argument("--core-warmup", type=int, default=100, help="Warmup iterations per core compute kernel")
    parser.add_argument("--core-system-sizes", type=int, nargs="+", default=[10, 20, 60], help="Core benchmark system sizes as NxN grids")
    parser.add_argument("--core-full", action="store_true", help="Use larger benchmark problem sizes instead of quick mode")
    parser.add_argument("--core-cpu-only", action="store_true", help="Disable GPU attempts for core compute benchmarks")
    add_logging_cli_args(parser)
    return parser


def run_perf_smoke(
    num_iters: int,
    num_samples: int,
    include_core: bool = True,
    core_iterations: int = 3,
    core_warmup: int = 100,
    core_quick: bool = True,
    core_include_gpu: bool = True,
    core_system_sizes=None,
):
    """Run the smoke benchmarks and return a JSON-serializable report."""

    if core_system_sizes is None:
        core_system_sizes = [10, 20, 60]

    report = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "timestamp_unix": time.time(),
        "measure_execution_time": _benchmark_measure_execution_time(num_iters=num_iters),
        "latency_math": _benchmark_latency_math(num_samples=num_samples),
    }

    if include_core:
        report["core_compute"] = run_core_compute_benchmarks(
            iterations=core_iterations,
            warmup=core_warmup,
            include_gpu=core_include_gpu,
            quick=core_quick,
            system_sizes=core_system_sizes,
        )

    return report


def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    configure_logging_from_args(args, app_name="pyrtc-perf-smoke", component_name="benchmarks.perf_smoke")

    report = run_perf_smoke(
        num_iters=args.num_iters,
        num_samples=args.num_samples,
        include_core=not args.skip_core,
        core_iterations=args.core_iterations,
        core_warmup=args.core_warmup,
        core_quick=not args.core_full,
        core_include_gpu=not args.core_cpu_only,
        core_system_sizes=args.core_system_sizes,
    )

    output_path = Path(args.output)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Wrote performance smoke report to %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
