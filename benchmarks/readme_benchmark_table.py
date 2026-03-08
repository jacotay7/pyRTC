import argparse
import json
from pathlib import Path


CPU_KERNELS = [
    ("wavefront_sensor.downsample_int32_image_jit", "WFS downsample"),
    ("wavefront_sensor.rotate_image_jit", "WFS rotate"),
    ("wavefront_corrector.ModaltoZonalWithFlat", "WFC modal->zonal"),
    ("loop.leakyIntegratorNumba", "Loop leaky integrator"),
    ("slopes.computeSlopesPYWFSOptimNumba", "PYWFS slopes"),
    ("slopes.computeSlopesSHWFSOptimNumba", "SHWFS slopes"),
]

GPU_KERNELS = {
    "loop.leakyIntegratorNumba": "loop.leakIntegratorGPU",
    "slopes.computeSlopesPYWFSOptimNumba": "slopes.computeSlopesPYWFSTorch",
}


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


def build_markdown(report: dict) -> str:
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
    args = parser.parse_args(argv)

    report = _load_report(Path(args.report))
    markdown = build_markdown(report)
    Path(args.output).write_text(markdown, encoding="utf-8")
    print(f"Wrote README benchmark markdown to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())