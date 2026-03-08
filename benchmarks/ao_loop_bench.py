import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from benchmarks.core_compute_bench import collect_system_info
from pyRTC.logging_utils import add_logging_cli_args, configure_logging_from_args, get_logger
from pyRTC.Loop import leakyIntegratorNumba
from pyRTC.Pipeline import gpu_torch_available
from pyRTC.SlopesProcess import computeSlopesPYWFSOptimNumba, computeSlopesPYWFSTorch, computeSlopesSHWFSOptimNumba


logger = get_logger(__name__)


AO_SENSOR_LABELS = {
    "pywfs": "PYWFS",
    "shwfs": "SHWFS",
}


def _format_stats(stats: dict[str, float] | None) -> str:
    if not stats or "p99_hz" not in stats or "p99_s" not in stats:
        return "-"
    hz = float(stats["p99_hz"])
    us = float(stats["p99_s"]) * 1e6
    if hz >= 1000.0:
        return f"{hz / 1000.0:.1f} kHz / {us:.1f} us"
    return f"{hz:.0f} Hz / {us:.1f} us"


def _render_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def _fmt_row(row: list[str]) -> str:
        return " | ".join(cell.ljust(widths[index]) for index, cell in enumerate(row))

    separator = "-+-".join("-" * width for width in widths)
    lines = [_fmt_row(headers), separator]
    lines.extend(_fmt_row(row) for row in rows)
    return "\n".join(lines)


def _build_summary_table(results: dict[str, Any]) -> str:
    rows: list[list[str]] = []
    for sensor_name, profiles in results.get("results", {}).items():
        for profile_name, variants in profiles.items():
            cpu_summary = _format_stats(variants.get("cpu"))
            gpu_stats = variants.get("gpu")
            gpu_summary = _format_stats(gpu_stats if isinstance(gpu_stats, dict) and "status" not in gpu_stats else None)
            rows.append([
                AO_SENSOR_LABELS.get(sensor_name, sensor_name),
                profile_name,
                cpu_summary,
                gpu_summary,
            ])
    return _render_table(["Sensor", "Size", "CPU p99", "GPU p99"], rows)


def _log_benchmark_summary(results: dict[str, Any]) -> None:
    logger.info("Synthetic AO loop benchmark summary:\n%s", _build_summary_table(results))


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


def _time_kernel(
    func: Callable[[], Any],
    iterations: int,
    warmup: int,
    sync: Callable[[], None] | None = None,
) -> dict[str, float]:
    for _ in range(warmup):
        func()
    if sync is not None:
        sync()

    timings = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        if sync is not None:
            sync()
        start = time.perf_counter()
        func()
        if sync is not None:
            sync()
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


def _build_dense_response(signal_size: int, num_modes: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    response = rng.normal(size=(signal_size, num_modes)).astype(np.float32)
    norms = np.linalg.norm(response, axis=0)
    norms[norms == 0.0] = 1.0
    response /= norms
    response *= np.float32(0.08)
    return response


def _build_modal_drive(num_modes: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    amplitudes = (0.2 / np.sqrt(np.arange(1, num_modes + 1, dtype=np.float32))).astype(np.float32)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=num_modes).astype(np.float32)
    phase_rates = np.linspace(0.03, 0.21, num_modes, dtype=np.float32)
    return amplitudes, phases, phase_rates


def _build_pywfs_masks(num_pixels_in_pupils: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    total_length = 4 * num_pixels_in_pupils
    p1 = np.zeros(total_length, dtype=np.bool_)
    p2 = np.zeros(total_length, dtype=np.bool_)
    p3 = np.zeros(total_length, dtype=np.bool_)
    p4 = np.zeros(total_length, dtype=np.bool_)
    p1[0:num_pixels_in_pupils] = True
    p2[num_pixels_in_pupils : 2 * num_pixels_in_pupils] = True
    p3[2 * num_pixels_in_pupils : 3 * num_pixels_in_pupils] = True
    p4[3 * num_pixels_in_pupils : 4 * num_pixels_in_pupils] = True
    return p1, p2, p3, p4


def _build_pywfs_cpu_step(grid_size: int) -> Callable[[], None]:
    num_modes = grid_size * grid_size
    num_pixels = num_modes
    signal_size = 2 * num_pixels
    response = _build_dense_response(signal_size, num_modes, seed=10 + grid_size)
    reconstruction = response.T.copy()
    amplitudes, phase, phase_rates = _build_modal_drive(num_modes, seed=20 + grid_size)
    correction = np.zeros(num_modes, dtype=np.float32)
    correction_buffer = np.zeros(num_modes, dtype=np.float32)
    slopes = np.zeros(signal_size, dtype=np.float32)
    ref_slopes = np.zeros(signal_size, dtype=np.float32)
    p1_mask, p2_mask, p3_mask, p4_mask = _build_pywfs_masks(num_pixels)
    p1 = np.zeros(num_pixels, dtype=np.float32)
    p2 = np.zeros(num_pixels, dtype=np.float32)
    p3 = np.zeros(num_pixels, dtype=np.float32)
    p4 = np.zeros(num_pixels, dtype=np.float32)
    tmp1 = np.zeros(num_pixels, dtype=np.float32)
    tmp2 = np.zeros(num_pixels, dtype=np.float32)
    image = np.zeros(4 * num_pixels, dtype=np.float32)
    leak = np.float32(0.02)
    flux = np.float32(2048.0 * 0.25)
    slope_limit = np.float32(0.8)

    def step() -> None:
        nonlocal phase, correction
        phase = phase + phase_rates
        disturbance = amplitudes * np.sin(phase)
        residual_modes = disturbance - correction
        target_slopes = response @ residual_modes
        np.clip(target_slopes, -slope_limit, slope_limit, out=target_slopes)
        sx = target_slopes[:num_pixels]
        sy = target_slopes[num_pixels:]

        image[0:num_pixels] = flux * (1.0 + sx + sy)
        image[num_pixels : 2 * num_pixels] = flux * (1.0 + sx - sy)
        image[2 * num_pixels : 3 * num_pixels] = flux * (1.0 - sx + sy)
        image[3 * num_pixels : 4 * num_pixels] = flux * (1.0 - sx - sy)

        measured_slopes = computeSlopesPYWFSOptimNumba(
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
            num_pixels,
            slopes,
            ref_slopes,
        )
        leakyIntegratorNumba(measured_slopes, reconstruction, correction, correction_buffer, leak, num_modes - 1)
        np.clip(correction_buffer, -1.0, 1.0, out=correction)

    return step


def _build_shwfs_cpu_step(grid_size: int) -> Callable[[], None]:
    num_modes = grid_size * grid_size
    signal_size = 2 * num_modes
    response = _build_dense_response(signal_size, num_modes, seed=30 + grid_size)
    reconstruction = response.T.copy()
    amplitudes, phase, phase_rates = _build_modal_drive(num_modes, seed=40 + grid_size)
    correction = np.zeros(num_modes, dtype=np.float32)
    correction_buffer = np.zeros(num_modes, dtype=np.float32)
    image = np.zeros((2 * grid_size, 2 * grid_size), dtype=np.float32)
    slopes_2d = np.zeros((2 * grid_size, grid_size), dtype=np.float32)
    ref_2d = np.zeros_like(slopes_2d)
    xvals = np.array([[-1.0, 1.0], [-1.0, 1.0]], dtype=np.float32)
    leak = np.float32(0.02)
    flux = np.float32(4096.0 * 0.25)
    slope_limit = np.float32(0.8)

    def step() -> None:
        nonlocal phase, correction
        phase = phase + phase_rates
        disturbance = amplitudes * np.sin(phase)
        residual_modes = disturbance - correction
        target_slopes = response @ residual_modes
        np.clip(target_slopes, -slope_limit, slope_limit, out=target_slopes)
        sx = target_slopes[:num_modes].reshape(grid_size, grid_size)
        sy = target_slopes[num_modes:].reshape(grid_size, grid_size)

        image[0::2, 0::2] = flux * (1.0 - sx - sy)
        image[0::2, 1::2] = flux * (1.0 + sx - sy)
        image[1::2, 0::2] = flux * (1.0 - sx + sy)
        image[1::2, 1::2] = flux * (1.0 + sx + sy)

        measured = computeSlopesSHWFSOptimNumba(
            image,
            slopes_2d,
            ref_2d,
            np.float32(1.0),
            np.float32(2.0),
            xvals,
            0,
            0,
            2,
        ).reshape(signal_size)
        leakyIntegratorNumba(measured, reconstruction, correction, correction_buffer, leak, num_modes - 1)
        np.clip(correction_buffer, -1.0, 1.0, out=correction)

    return step


def _compute_slopes_shwfs_torch(image: Any, num_regions: int, threshold: float, ref_slopes: Any) -> Any:
    import torch

    patches = image.reshape(num_regions, 2, num_regions, 2).permute(0, 2, 1, 3)
    valid = patches > threshold
    weighted = torch.where(valid, patches, torch.zeros_like(patches))
    norm = weighted.sum(dim=(-1, -2))
    safe_norm = torch.where(norm > 0, norm, torch.ones_like(norm))

    x_weights = torch.tensor([[-1.0, 1.0], [-1.0, 1.0]], device=image.device, dtype=image.dtype)
    y_weights = torch.tensor([[-1.0, -1.0], [1.0, 1.0]], device=image.device, dtype=image.dtype)
    slope_x = (weighted * x_weights).sum(dim=(-1, -2)) / safe_norm
    slope_y = (weighted * y_weights).sum(dim=(-1, -2)) / safe_norm
    slope_x = torch.where(norm > 0, slope_x, torch.zeros_like(slope_x))
    slope_y = torch.where(norm > 0, slope_y, torch.zeros_like(slope_y))
    return torch.cat((slope_x.reshape(-1), slope_y.reshape(-1))) - ref_slopes


def _build_gpu_sync(device: Any) -> Callable[[], None]:
    import torch

    if device.type != "cuda":
        return lambda: None
    return lambda: torch.cuda.synchronize(device)


def _build_pywfs_gpu_step(grid_size: int) -> tuple[Callable[[], None], Callable[[], None]] | None:
    if not gpu_torch_available():
        return None

    import torch

    if not torch.cuda.is_available():
        return None

    device = torch.device("cuda")
    num_modes = grid_size * grid_size
    num_pixels = num_modes
    signal_size = 2 * num_pixels
    response = torch.tensor(_build_dense_response(signal_size, num_modes, seed=10 + grid_size), device=device)
    reconstruction = response.transpose(0, 1).contiguous()
    amplitudes_np, phase_np, phase_rates_np = _build_modal_drive(num_modes, seed=20 + grid_size)
    amplitudes = torch.tensor(amplitudes_np, device=device)
    phase = torch.tensor(phase_np, device=device)
    phase_rates = torch.tensor(phase_rates_np, device=device)
    correction = torch.zeros(num_modes, dtype=torch.float32, device=device)
    image = torch.zeros(4 * num_pixels, dtype=torch.float32, device=device)
    slopes = torch.zeros(signal_size, dtype=torch.float32, device=device)
    ref_slopes = torch.zeros(signal_size, dtype=torch.float32, device=device)
    p1_mask, p2_mask, p3_mask, p4_mask = _build_pywfs_masks(num_pixels)
    p1_mask_t = torch.tensor(p1_mask, device=device)
    p2_mask_t = torch.tensor(p2_mask, device=device)
    p3_mask_t = torch.tensor(p3_mask, device=device)
    p4_mask_t = torch.tensor(p4_mask, device=device)
    leak = torch.tensor(0.02, dtype=torch.float32, device=device)
    flux = torch.tensor(2048.0 * 0.25, dtype=torch.float32, device=device)
    slope_limit = torch.tensor(0.8, dtype=torch.float32, device=device)

    def step() -> None:
        nonlocal phase, correction
        phase = phase + phase_rates
        disturbance = amplitudes * torch.sin(phase)
        residual_modes = disturbance - correction
        target_slopes = torch.matmul(response, residual_modes)
        target_slopes = torch.clamp(target_slopes, -slope_limit, slope_limit)
        sx = target_slopes[:num_pixels]
        sy = target_slopes[num_pixels:]

        image[0:num_pixels] = flux * (1.0 + sx + sy)
        image[num_pixels : 2 * num_pixels] = flux * (1.0 + sx - sy)
        image[2 * num_pixels : 3 * num_pixels] = flux * (1.0 - sx + sy)
        image[3 * num_pixels : 4 * num_pixels] = flux * (1.0 - sx - sy)

        measured = computeSlopesPYWFSTorch(
            image,
            p1_mask_t,
            p2_mask_t,
            p3_mask_t,
            p4_mask_t,
            num_pixels,
            slopes,
            ref_slopes,
        )
        correction = (1.0 - leak) * correction - torch.matmul(reconstruction, measured)
        correction[num_modes:] = 0
        correction = torch.clamp(correction, -1.0, 1.0)

    return step, _build_gpu_sync(device)


def _build_shwfs_gpu_step(grid_size: int) -> tuple[Callable[[], None], Callable[[], None]] | None:
    if not gpu_torch_available():
        return None

    import torch

    if not torch.cuda.is_available():
        return None

    device = torch.device("cuda")
    num_modes = grid_size * grid_size
    signal_size = 2 * num_modes
    response = torch.tensor(_build_dense_response(signal_size, num_modes, seed=30 + grid_size), device=device)
    reconstruction = response.transpose(0, 1).contiguous()
    amplitudes_np, phase_np, phase_rates_np = _build_modal_drive(num_modes, seed=40 + grid_size)
    amplitudes = torch.tensor(amplitudes_np, device=device)
    phase = torch.tensor(phase_np, device=device)
    phase_rates = torch.tensor(phase_rates_np, device=device)
    correction = torch.zeros(num_modes, dtype=torch.float32, device=device)
    image = torch.zeros((2 * grid_size, 2 * grid_size), dtype=torch.float32, device=device)
    ref_slopes = torch.zeros(signal_size, dtype=torch.float32, device=device)
    leak = torch.tensor(0.02, dtype=torch.float32, device=device)
    flux = torch.tensor(4096.0 * 0.25, dtype=torch.float32, device=device)
    slope_limit = torch.tensor(0.8, dtype=torch.float32, device=device)

    def step() -> None:
        nonlocal phase, correction
        phase = phase + phase_rates
        disturbance = amplitudes * torch.sin(phase)
        residual_modes = disturbance - correction
        target_slopes = torch.matmul(response, residual_modes)
        target_slopes = torch.clamp(target_slopes, -slope_limit, slope_limit)
        sx = target_slopes[:num_modes].reshape(grid_size, grid_size)
        sy = target_slopes[num_modes:].reshape(grid_size, grid_size)

        image[0::2, 0::2] = flux * (1.0 - sx - sy)
        image[0::2, 1::2] = flux * (1.0 + sx - sy)
        image[1::2, 0::2] = flux * (1.0 - sx + sy)
        image[1::2, 1::2] = flux * (1.0 + sx + sy)

        measured = _compute_slopes_shwfs_torch(image, grid_size, threshold=1.0, ref_slopes=ref_slopes)
        correction = (1.0 - leak) * correction - torch.matmul(reconstruction, measured)
        correction[num_modes:] = 0
        correction = torch.clamp(correction, -1.0, 1.0)

    return step, _build_gpu_sync(device)


def _benchmark_sensor(sensor_type: str, grid_size: int, iterations: int, warmup: int, include_gpu: bool) -> dict[str, Any]:
    if sensor_type == "pywfs":
        cpu_step = _build_pywfs_cpu_step(grid_size)
        gpu_builder = _build_pywfs_gpu_step
    elif sensor_type == "shwfs":
        cpu_step = _build_shwfs_cpu_step(grid_size)
        gpu_builder = _build_shwfs_gpu_step
    else:
        raise ValueError(f"Unknown sensor type: {sensor_type}")

    result: dict[str, Any] = {
        "cpu": _time_kernel(cpu_step, iterations=iterations, warmup=warmup),
    }

    if include_gpu:
        gpu_variant = gpu_builder(grid_size)
        if gpu_variant is None:
            result["gpu"] = {"status": {"available": False, "reason": "CUDA not available"}}
        else:
            gpu_step, sync = gpu_variant
            result["gpu"] = _time_kernel(gpu_step, iterations=iterations, warmup=warmup, sync=sync)

    return result


def run_ao_loop_benchmarks(
    iterations: int = 200,
    warmup: int = 25,
    include_gpu: bool = True,
    system_sizes=None,
):
    if system_sizes is None:
        system_sizes = [10, 20, 60]

    results: dict[str, Any] = {
        "meta": {
            "system_info": collect_system_info(),
            "iterations": iterations,
            "warmup": warmup,
            "include_gpu": include_gpu,
            "system_sizes": list(system_sizes),
            "timestamp_unix": time.time(),
            "benchmark_type": "synthetic_closed_loop",
        },
        "results": {
            "pywfs": {},
            "shwfs": {},
        },
    }

    for sensor_type in ("pywfs", "shwfs"):
        for size in system_sizes:
            profile_key = f"{size}x{size}"
            logger.info("Benchmarking %s closed loop for %s", sensor_type, profile_key)
            results["results"][sensor_type][profile_key] = _benchmark_sensor(
                sensor_type=sensor_type,
                grid_size=size,
                iterations=iterations,
                warmup=warmup,
                include_gpu=include_gpu,
            )

    return results


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark synthetic closed-loop AO iterations for SHWFS and PYWFS.")
    parser.add_argument("--iterations", type=int, default=2000, help="Timed iterations per benchmark")
    parser.add_argument("--warmup", type=int, default=200, help="Warmup iterations per benchmark")
    parser.add_argument("--cpu-only", action="store_true", help="Skip GPU variants")
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
    configure_logging_from_args(args, app_name="pyrtc-ao-loop-bench", component_name="benchmarks.ao_loop_bench")

    results = run_ao_loop_benchmarks(
        iterations=args.iterations,
        warmup=args.warmup,
        include_gpu=not args.cpu_only,
        system_sizes=args.system_sizes,
    )

    _log_benchmark_summary(results)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        logger.info("Wrote AO loop benchmark report to %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())