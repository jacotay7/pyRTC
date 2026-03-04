import argparse
import json
from pathlib import Path
from typing import Dict


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare perf smoke report against committed baseline.")
    parser.add_argument("--current", type=str, default="benchmarks/perf_smoke_report.json", help="Current report path")
    parser.add_argument("--baseline", type=str, default="benchmarks/perf_smoke_baseline.json", help="Baseline report path")
    return parser


def _require_path(data: Dict, dotted_path: str):
    node = data
    for segment in dotted_path.split("."):
        if segment not in node:
            raise KeyError(dotted_path)
        node = node[segment]
    return node


def _required_metric_paths(current: Dict):
    paths = [
        "measure_execution_time.median",
        "measure_execution_time.iqr",
        "measure_execution_time.ci1",
        "measure_execution_time.ci99",
        "latency_math.mean_latency_s",
        "latency_math.p99_latency_s",
        "latency_math.frame_shift",
    ]

    profiles = current.get("core_compute", {}).get("profiles", {})
    for profile_name, kernels in profiles.items():
        for kernel_name in kernels.keys():
            base = f"core_compute.profiles.{profile_name}.{kernel_name}"
            paths.extend(
                [
                    f"{base}.mean_s",
                    f"{base}.median_s",
                    f"{base}.p95_s",
                    f"{base}.p99_s",
                    f"{base}.p99_hz",
                ]
            )

    gpu_section = current.get("core_compute", {}).get("gpu_kernels", {})
    gpu_status = gpu_section.get("status", {})
    if gpu_status.get("available") is True:
        for kernel_name in gpu_section.keys():
            if kernel_name == "status":
                continue
            base = f"core_compute.gpu_kernels.{kernel_name}"
            paths.extend(
                [
                    f"{base}.mean_s",
                    f"{base}.median_s",
                    f"{base}.p95_s",
                    f"{base}.p99_s",
                    f"{base}.p99_hz",
                ]
            )

    return paths


def compare_against_baseline(current: Dict, baseline: Dict):
    required = _required_metric_paths(current)
    missing = []
    comparison = {}

    for path in required:
        try:
            current_value = _require_path(current, path)
            baseline_value = _require_path(baseline, path)
        except KeyError:
            missing.append(path)
            continue

        if isinstance(current_value, (int, float)) and isinstance(baseline_value, (int, float)):
            ratio = None
            if baseline_value != 0:
                ratio = float(current_value) / float(baseline_value)
            comparison[path] = {
                "current": float(current_value),
                "baseline": float(baseline_value),
                "ratio": ratio,
            }

    return missing, comparison


def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    current_path = Path(args.current)
    baseline_path = Path(args.baseline)

    if not current_path.exists():
        raise SystemExit(f"Current perf report not found: {current_path}")

    if not baseline_path.exists():
        raise SystemExit(
            f"Baseline perf report not found: {baseline_path}. Commit a baseline JSON to enable CI comparison."
        )

    current = json.loads(current_path.read_text(encoding="utf-8"))
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))

    missing, comparison = compare_against_baseline(current, baseline)

    print(json.dumps(comparison, indent=2))

    if missing:
        raise SystemExit(
            "Missing baseline metrics for comparison:\n" + "\n".join(sorted(missing))
        )

    print("Baseline comparison succeeded: all required metrics found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
