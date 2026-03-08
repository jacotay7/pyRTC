"""Render benchmark JSON reports into README-ready markdown tables."""

import argparse
import json
from pathlib import Path

from pyRTC.logging_utils import add_logging_cli_args, configure_logging_from_args, get_logger


logger = get_logger(__name__)


LOOP_ROWS = [
    ("pywfs", "PYWFS full loop"),
    ("shwfs", "SHWFS full loop"),
]


def _load_report(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt_metric(stats: dict | None) -> str:
    if not stats or "p99_hz" not in stats or "p99_s" not in stats:
        return "-"
    hz = float(stats["p99_hz"])
    us = float(stats["p99_s"]) * 1e6
    if hz >= 1000:
        return f"{hz/1000:.1f} kHz / {us:.1f} us"
    return f"{hz:.0f} Hz / {us:.1f} us"


def _build_closed_loop_markdown(report: dict) -> str:
    meta = report.get("meta", {})
    system = meta.get("system_info", {})
    gpu = system.get("gpu", {})
    results = report.get("results", {})
    sizes = list(meta.get("system_sizes", []))

    lines = []
    lines.append("### Benchmark Host")
    lines.append("")
    lines.append("| Component | Value |")
    lines.append("| --- | --- |")
    lines.append(f"| CPU | {system.get('cpu_model', 'unknown')} |")
    lines.append(f"| CPU Threads | {system.get('cpu_count', 'unknown')} |")
    lines.append(f"| GPU | {gpu.get('device_name', 'not detected')} |")
    lines.append(f"| GPU Memory | {gpu.get('memory_total', '-')} |")
    lines.append(f"| NVIDIA Driver | {gpu.get('driver_version', '-')} |")
    lines.append(f"| Python | {system.get('python', meta.get('python', '-'))} |")
    lines.append(f"| Torch | {gpu.get('torch_version', '-')} |")
    lines.append(f"| CUDA | {gpu.get('cuda_runtime', '-')} |")
    lines.append("")
    lines.append("### Synthetic AO Loop Benchmarks")
    lines.append("")
    lines.append("Values are reported as `p99 throughput / p99 latency`.")
    lines.append("")
    lines.append("| Loop | 10x10 CPU | 10x10 GPU | 20x20 CPU | 20x20 GPU | 60x60 CPU | 60x60 GPU |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")

    for sensor_key, label in LOOP_ROWS:
        row = [label]
        sensor_results = results.get(sensor_key, {})
        for size in sizes:
            profile = sensor_results.get(f"{size}x{size}", {})
            row.append(_fmt_metric(profile.get("cpu")))
            gpu_stats = profile.get("gpu")
            if isinstance(gpu_stats, dict) and "status" in gpu_stats:
                gpu_stats = None
            row.append(_fmt_metric(gpu_stats))
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines) + "\n"


def build_markdown(report: dict) -> str:
    """Convert either core-compute or closed-loop reports into markdown."""

    if report.get("meta", {}).get("benchmark_type") == "synthetic_closed_loop":
        return _build_closed_loop_markdown(report)

    core = report.get("core_compute", report)
    meta = core.get("meta", {})
    system = meta.get("system_info", {})
    gpu = system.get("gpu", {})
    profiles = core.get("profiles", {})
    gpu_profiles = core.get("gpu_profiles", {})
    sizes = list(meta.get("profile_sizes_used", []))

    lines = []
    lines.append("### Benchmark Host")
    lines.append("")
    lines.append("| Component | Value |")
    lines.append("| --- | --- |")
    lines.append(f"| CPU | {system.get('cpu_model', 'unknown')} |")
    lines.append(f"| CPU Threads | {system.get('cpu_count', 'unknown')} |")
    lines.append(f"| GPU | {gpu.get('device_name', 'not detected')} |")
    lines.append(f"| GPU Memory | {gpu.get('memory_total', '-')} |")
    lines.append(f"| NVIDIA Driver | {gpu.get('driver_version', '-')} |")
    lines.append(f"| Python | {system.get('python', meta.get('python', '-'))} |")
    lines.append(f"| Torch | {gpu.get('torch_version', '-')} |")
    lines.append(f"| CUDA | {gpu.get('cuda_runtime', '-')} |")
    lines.append("")
    lines.append("### Core Compute Benchmarks")
    lines.append("")
    lines.append("Values are reported as `p99 throughput / p99 latency`.")
    lines.append("")
    lines.append("| Kernel | 10x10 CPU | 10x10 GPU | 20x20 CPU | 20x20 GPU | 60x60 CPU | 60x60 GPU |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")

    for kernel_key, label in CPU_KERNELS:
        row = [label]
        gpu_key = GPU_KERNELS.get(kernel_key)
        for size in sizes:
            profile_key = f"{size}x{size}"
            cpu_stats = profiles.get(profile_key, {}).get(kernel_key)
            row.append(_fmt_metric(cpu_stats))
            gpu_stats = None
            if gpu_key:
                gpu_profile = gpu_profiles.get(profile_key, {})
                if gpu_profile.get("status", {}).get("available") is True:
                    gpu_stats = gpu_profile.get(gpu_key)
            row.append(_fmt_metric(gpu_stats))
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Generate README markdown tables from a benchmark JSON report.")
    parser.add_argument("--report", default="benchmarks/readme_benchmark_report.json", help="Benchmark JSON path")
    parser.add_argument("--output", default="benchmarks/readme_benchmark_table.md", help="Markdown output path")
    add_logging_cli_args(parser)
    args = parser.parse_args(argv)
    configure_logging_from_args(
        args,
        app_name="pyrtc-readme-benchmark-table",
        component_name="benchmarks.readme_benchmark_table",
    )

    report = _load_report(Path(args.report))
    markdown = build_markdown(report)
    Path(args.output).write_text(markdown, encoding="utf-8")
    logger.info("Wrote README benchmark markdown to %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())