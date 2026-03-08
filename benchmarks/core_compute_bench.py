import argparse
import json
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict

import numpy as np

from pyRTC.Loop import leakIntegratorGPU, leakyIntegratorNumba
from pyRTC.Pipeline import gpu_torch_available
from pyRTC.logging_utils import add_logging_cli_args, configure_logging_from_args, get_logger
from pyRTC.SlopesProcess import (
    computeSlopesPYWFSOptimNumba,
    computeSlopesPYWFSTorch,
    computeSlopesSHWFSOptimNumba,
)
from pyRTC.WavefrontCorrector import ModaltoZonalWithFlat
from pyRTC.WavefrontSensor import downsample_int32_image_jit, rotate_image_jit


logger = get_logger(__name__)


def _format_stats(stats: dict[str, float] | None) -> str:
    if not stats or "p99_hz" not in stats or "p99_s" not in stats:
        return "-"
    hz = float(stats["p99_hz"])
    us = float(stats["p99_s"]) * 1e6
    if hz >= 1000.0:
        return f"{hz / 1000.0:.1f} kHz / {us:.1f} us"
    return f"{hz:.0f} Hz / {us:.1f} us"


def _log_benchmark_summary(results: Dict[str, Any]) -> None:
    profiles = results.get("profiles", {})
    gpu_profiles = results.get("gpu_profiles", {})
    logger.info("Core compute benchmark summary:")
    for profile_name, kernels in profiles.items():
        logger.info("  %s", profile_name)
        for kernel_name, stats in kernels.items():
            summary = _format_stats(stats)
            gpu_summary = ""
            gpu_profile = gpu_profiles.get(profile_name, {})
            if kernel_name == "loop.leakyIntegratorNumba":
                gpu_stats = gpu_profile.get("loop.leakIntegratorGPU")
                if gpu_profile.get("status", {}).get("available") is True and gpu_stats:
                    gpu_summary = f" | GPU { _format_stats(gpu_stats) }"
            elif kernel_name == "slopes.computeSlopesPYWFSOptimNumba":
                gpu_stats = gpu_profile.get("slopes.computeSlopesPYWFSTorch")
                if gpu_profile.get("status", {}).get("available") is True and gpu_stats:
                    gpu_summary = f" | GPU { _format_stats(gpu_stats) }"
            logger.info("    %s: CPU %s%s", kernel_name, summary, gpu_summary)


def _safe_percentile(values, pct: float) -> float:
    vals = sorted(float(x) for x in values)
    if not vals:
        return 0.0
    if len(vals) == 1:
        return vals[0]
    rank = (pct / 100.0) * (len(vals) - 1)
    low = int(rank)
    high = min(low + 1, len(vals) - 1)
    if low == high:
        return vals[low]
    weight = rank - low
    return vals[low] * (1.0 - weight) + vals[high] * weight


def _run_command_lines(command: list[str]) -> list[str]:
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
    except Exception:
        return []
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def collect_system_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_model": platform.processor() or "unknown",
        "cpu_count": int(os.cpu_count() or 0),
    }

    lscpu_path = shutil.which("lscpu")
    if lscpu_path:
        for line in _run_command_lines([lscpu_path]):
            if line.startswith("Model name:"):
                info["cpu_model"] = line.split(":", 1)[1].strip()
            elif line.startswith("CPU(s):") and "NUMA" not in line:
                try:
                    info["cpu_count"] = int(line.split(":", 1)[1].strip())
                except ValueError:
                    pass

    gpu_info = {
        "available": False,
    }
    if gpu_torch_available():
        try:
            import torch

            gpu_info["torch_version"] = torch.__version__
            gpu_info["cuda_runtime"] = torch.version.cuda
            gpu_info["available"] = bool(torch.cuda.is_available())
            if torch.cuda.is_available():
                gpu_info["device_name"] = str(torch.cuda.get_device_name(0))
        except Exception:
            pass

    nvidia_smi_path = shutil.which("nvidia-smi")
    if nvidia_smi_path:
        lines = _run_command_lines(
            [nvidia_smi_path, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]
        )
        if lines:
            parts = [part.strip() for part in lines[0].split(",")]
            gpu_info["available"] = True
            if len(parts) >= 1:
                gpu_info.setdefault("device_name", parts[0])
            if len(parts) >= 2:
                gpu_info["driver_version"] = parts[1]
            if len(parts) >= 3:
                gpu_info["memory_total"] = parts[2]

    info["gpu"] = gpu_info
    return info


def _time_kernel(func: Callable[[], Any], iterations: int, warmup: int) -> Dict[str, float]:
    for _ in range(warmup):
        func()

    timings = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        start = time.perf_counter()
        func()
        timings[index] = time.perf_counter() - start

    timings_list = [float(x) for x in timings]
    sorted_timings = sorted(timings_list)
    p99_s = _safe_percentile(sorted_timings, 99)
    return {
        "iterations": float(iterations),
        "warmup": float(warmup),
        "mean_s": float(sum(sorted_timings) / len(sorted_timings)) if sorted_timings else 0.0,
        "median_s": _safe_percentile(sorted_timings, 50),
        "p95_s": _safe_percentile(sorted_timings, 95),
        "p99_s": p99_s,
        "p99_hz": float(1.0 / p99_s) if p99_s > 0 else float("inf"),
        "min_s": sorted_timings[0] if sorted_timings else 0.0,
        "max_s": sorted_timings[-1] if sorted_timings else 0.0,
    }


def _bench_wfs_downsample(iterations: int, warmup: int, side: int) -> Dict[str, float]:
    image = (np.random.RandomState(0).rand(side, side) * 1000).astype(np.int32)
    return _time_kernel(lambda: downsample_int32_image_jit(image, 2), iterations=iterations, warmup=warmup)


def _bench_wfs_rotate(iterations: int, warmup: int, side: int) -> Dict[str, float]:
    image = np.random.RandomState(1).rand(side, side).astype(np.float32)
    angle_rad = np.float32(0.1)
    return _time_kernel(lambda: rotate_image_jit(image, angle_rad), iterations=iterations, warmup=warmup)


def _bench_wfc_modal_to_zonal(iterations: int, warmup: int, num_modes: int, num_actuators: int) -> Dict[str, float]:
    rng = np.random.RandomState(2)
    correction = rng.randn(num_modes).astype(np.float32)
    m2c = rng.randn(num_actuators, num_modes).astype(np.float32)
    flat = rng.randn(num_actuators).astype(np.float32)
    return _time_kernel(
        lambda: ModaltoZonalWithFlat(correction=correction, M2C=m2c, flat=flat),
        iterations=iterations,
        warmup=warmup,
    )


def _bench_loop_leaky_integrator(iterations: int, warmup: int, signal_size: int, num_modes: int) -> Dict[str, float]:
    rng = np.random.RandomState(3)
    slopes = rng.randn(signal_size).astype(np.float32)
    recon = rng.randn(num_modes, signal_size).astype(np.float32)
    old_correction = rng.randn(num_modes).astype(np.float32)
    correction = np.zeros(num_modes, dtype=np.float32)
    leak = np.float32(0.05)
    num_active_modes = max(1, num_modes - 2)

    return _time_kernel(
        lambda: leakyIntegratorNumba(slopes, recon, old_correction, correction, leak, num_active_modes),
        iterations=iterations,
        warmup=warmup,
    )


def _build_pywfs_masks(length: int, pixels_per_pupil: int):
    p1 = np.zeros(length, dtype=np.bool_)
    p2 = np.zeros(length, dtype=np.bool_)
    p3 = np.zeros(length, dtype=np.bool_)
    p4 = np.zeros(length, dtype=np.bool_)

    p1[0:pixels_per_pupil] = True
    p2[pixels_per_pupil : 2 * pixels_per_pupil] = True
    p3[2 * pixels_per_pupil : 3 * pixels_per_pupil] = True
    p4[3 * pixels_per_pupil : 4 * pixels_per_pupil] = True
    return p1, p2, p3, p4


def _bench_slopes_pywfs_numba(iterations: int, warmup: int, image_side: int, pixels_per_pupil: int) -> Dict[str, float]:
    rng = np.random.RandomState(4)
    length = image_side * image_side
    image = (rng.rand(length) * 5000).astype(np.float32)

    p1_mask, p2_mask, p3_mask, p4_mask = _build_pywfs_masks(length, pixels_per_pupil)

    p1 = np.zeros(pixels_per_pupil, dtype=np.float32)
    p2 = np.zeros(pixels_per_pupil, dtype=np.float32)
    p3 = np.zeros(pixels_per_pupil, dtype=np.float32)
    p4 = np.zeros(pixels_per_pupil, dtype=np.float32)
    tmp1 = np.zeros(pixels_per_pupil, dtype=np.float32)
    tmp2 = np.zeros(pixels_per_pupil, dtype=np.float32)
    slopes = np.zeros(2 * pixels_per_pupil, dtype=np.float32)
    ref_slopes = np.zeros(2 * pixels_per_pupil, dtype=np.float32)

    return _time_kernel(
        lambda: computeSlopesPYWFSOptimNumba(
            image,
            p1_mask,
            p2_mask,
            p3_mask,
            p4_mask,
            p1,
            p2,
            p3,
            p4,
            tmp1,
            tmp2,
            pixels_per_pupil,
            slopes,
            ref_slopes,
        ),
        iterations=iterations,
        warmup=warmup,
    )


def _bench_slopes_shwfs_numba(iterations: int, warmup: int, num_regions: int, spacing: int, int_n: int) -> Dict[str, float]:
    rng = np.random.RandomState(5)
    side = num_regions * spacing
    image = (rng.rand(side, side) * 3000).astype(np.float32)

    slopes = np.zeros((2 * num_regions, num_regions), dtype=np.float32)
    unaberrated = np.zeros((2 * num_regions, num_regions), dtype=np.float32)
    xvals = np.arange(int_n * int_n, dtype=np.float32).reshape(int_n, int_n)

    threshold = np.float32(1.0)
    spacing_val = np.float32(spacing)

    return _time_kernel(
        lambda: computeSlopesSHWFSOptimNumba(
            image,
            slopes,
            unaberrated,
            threshold,
            spacing_val,
            xvals,
            0,
            0,
            int_n,
        ),
        iterations=iterations,
        warmup=warmup,
    )


def _bench_gpu_kernels(iterations: int, warmup: int, signal_size: int, num_modes: int, pixels_per_pupil: int) -> Dict[str, Dict[str, float]]:
    if not gpu_torch_available():
        return {"status": {"available": False, "reason": "PyTorch not available"}}

    import torch

    if not torch.cuda.is_available():
        return {"status": {"available": False, "reason": "CUDA not available"}}

    rng = np.random.RandomState(6)

    slopes_np = rng.randn(signal_size).astype(np.float32)
    recon_np = rng.randn(num_modes, signal_size).astype(np.float32)
    old_np = rng.randn(num_modes).astype(np.float32)

    recon_gpu = torch.tensor(recon_np, device="cuda")

    leak_stats = _time_kernel(
        lambda: leakIntegratorGPU(slopes_np, recon_gpu, old_np, 0.05, max(1, num_modes - 2)),
        iterations=iterations,
        warmup=warmup,
    )

    length = max(signal_size, 4 * pixels_per_pupil)
    image = torch.tensor((rng.rand(length) * 5000).astype(np.float32), device="cuda")

    p1_mask_np, p2_mask_np, p3_mask_np, p4_mask_np = _build_pywfs_masks(length, pixels_per_pupil)
    p1_mask = torch.tensor(p1_mask_np, device="cuda")
    p2_mask = torch.tensor(p2_mask_np, device="cuda")
    p3_mask = torch.tensor(p3_mask_np, device="cuda")
    p4_mask = torch.tensor(p4_mask_np, device="cuda")
    slopes = torch.zeros(2 * pixels_per_pupil, device="cuda", dtype=torch.float32)
    ref = torch.zeros(2 * pixels_per_pupil, device="cuda", dtype=torch.float32)

    slopes_stats = _time_kernel(
        lambda: computeSlopesPYWFSTorch(
            image,
            p1_mask,
            p2_mask,
            p3_mask,
            p4_mask,
            pixels_per_pupil,
            slopes,
            ref,
        ),
        iterations=iterations,
        warmup=warmup,
    )

    return {
        "status": {"available": True, "device": str(torch.cuda.get_device_name(0))},
        "loop.leakIntegratorGPU": leak_stats,
        "slopes.computeSlopesPYWFSTorch": slopes_stats,
    }


def _run_profile_benchmarks(grid_size: int, iterations: int, warmup: int):
    side = max(16, 2 * grid_size)
    signal_size = 2 * grid_size * grid_size
    num_modes = grid_size * grid_size
    pywfs_side = max(32, 4 * grid_size)
    pixels_per_pupil = grid_size * grid_size
    num_regions = grid_size
    spacing = 2
    int_n = 2

    return {
        "wavefront_sensor.downsample_int32_image_jit": _bench_wfs_downsample(iterations, warmup, side),
        "wavefront_sensor.rotate_image_jit": _bench_wfs_rotate(iterations, warmup, side),
        "wavefront_corrector.ModaltoZonalWithFlat": _bench_wfc_modal_to_zonal(
            iterations, warmup, num_modes, num_modes
        ),
        "loop.leakyIntegratorNumba": _bench_loop_leaky_integrator(iterations, warmup, signal_size, num_modes),
        "slopes.computeSlopesPYWFSOptimNumba": _bench_slopes_pywfs_numba(
            iterations, warmup, pywfs_side, pixels_per_pupil
        ),
        "slopes.computeSlopesSHWFSOptimNumba": _bench_slopes_shwfs_numba(
            iterations, warmup, num_regions, spacing, int_n
        ),
    }


def run_core_compute_benchmarks(
    iterations: int = 50,
    warmup: int = 5,
    include_gpu: bool = True,
    quick: bool = False,
    system_sizes=None,
):
    if system_sizes is None:
        system_sizes = [10, 20, 60]

    profile_sizes = list(system_sizes)

    results = {
        "meta": {
            "system_info": collect_system_info(),
            "iterations": iterations,
            "warmup": warmup,
            "quick": quick,
            "include_gpu": include_gpu,
            "system_sizes": list(system_sizes),
            "profile_sizes_used": profile_sizes,
            "timestamp_unix": time.time(),
        },
        "profiles": {},
    }

    for size in profile_sizes:
        profile_key = f"{size}x{size}"
        results["profiles"][profile_key] = _run_profile_benchmarks(
            grid_size=size,
            iterations=iterations,
            warmup=warmup,
        )

    if include_gpu:
        results["gpu_profiles"] = {}
        for size in profile_sizes:
            signal_size = 2 * size * size
            num_modes = size * size
            pixels_per_pupil = size * size
            results["gpu_profiles"][f"{size}x{size}"] = _bench_gpu_kernels(
                iterations=max(5, iterations // 2),
                warmup=max(1, warmup // 2),
                signal_size=signal_size,
                num_modes=num_modes,
                pixels_per_pupil=pixels_per_pupil,
            )

        gpu_grid = max(profile_sizes)
        signal_size = 2 * gpu_grid * gpu_grid
        num_modes = gpu_grid * gpu_grid
        pixels_per_pupil = gpu_grid * gpu_grid
        results["gpu_kernels"] = _bench_gpu_kernels(
            iterations=max(5, iterations // 2),
            warmup=max(1, warmup // 2),
            signal_size=signal_size,
            num_modes=num_modes,
            pixels_per_pupil=pixels_per_pupil,
        )

    return results


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark core pyRTC compute kernels (JIT + optional GPU paths)."
    )
    parser.add_argument("--iterations", type=int, default=2000, help="Timed iterations per kernel")
    parser.add_argument("--warmup", type=int, default=200, help="Warmup iterations per kernel")
    parser.add_argument("--quick", action="store_true", help="Run smaller/faster problem sizes")
    parser.add_argument("--cpu-only", action="store_true", help="Skip GPU kernel benchmarks")
    parser.add_argument(
        "--system-sizes",
        type=int,
        nargs="+",
        default=[10, 20, 60],
        help="System grid sizes to benchmark (interpreted as NxN)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional JSON output path",
    )
    add_logging_cli_args(parser)
    return parser


def main(argv=None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    configure_logging_from_args(args, app_name="pyrtc-core-bench", component_name="benchmarks.core_compute_bench")

    results = run_core_compute_benchmarks(
        iterations=args.iterations,
        warmup=args.warmup,
        include_gpu=not args.cpu_only,
        quick=args.quick,
        system_sizes=args.system_sizes,
    )

    _log_benchmark_summary(results)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        logger.info("Wrote core benchmark report to %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
