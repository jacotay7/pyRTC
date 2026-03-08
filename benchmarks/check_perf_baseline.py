import argparse
import json
from pathlib import Path
from typing import Dict

from pyRTC.logging_utils import add_logging_cli_args, configure_logging_from_args, get_logger


logger = get_logger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare perf smoke report against committed baseline.")
    parser.add_argument("--current", type=str, default="benchmarks/perf_smoke_report.json", help="Current report path")
    parser.add_argument("--baseline", type=str, default="benchmarks/perf_smoke_baseline.json", help="Baseline report path")
    add_logging_cli_args(parser)
    return parser


def _require_path(data: Dict, path_segments):
    node = data
    for segment in path_segments:
        if segment not in node:
            raise KeyError(path_segments)
        node = node[segment]
    return node


def _segments_to_key(path_segments):
    return ".".join(path_segments)


def _normalize_report_shape(data: Dict):
    # Full perf smoke report shape
    if "core_compute" in data:
        return data
    # Core-only baseline shape (meta/profiles/gpu_kernels at top level)
    if "profiles" in data or "gpu_kernels" in data:
        return {"core_compute": data}
    return data


def _required_metric_paths(current: Dict):
    paths = []

    profiles = current.get("core_compute", {}).get("profiles", {})
    for profile_name, kernels in profiles.items():
        for kernel_name in kernels.keys():
            base = ("core_compute", "profiles", profile_name, kernel_name)
            paths.extend(
                [
                    (*base, "mean_s"),
                    (*base, "median_s"),
                    (*base, "p95_s"),
                    (*base, "p99_s"),
                    (*base, "p99_hz"),
                ]
            )

    gpu_section = current.get("core_compute", {}).get("gpu_kernels", {})
    gpu_status = gpu_section.get("status", {})
    if gpu_status.get("available") is True:
        for kernel_name in gpu_section.keys():
            if kernel_name == "status":
                continue
            base = ("core_compute", "gpu_kernels", kernel_name)
            paths.extend(
                [
                    (*base, "mean_s"),
                    (*base, "median_s"),
                    (*base, "p95_s"),
                    (*base, "p99_s"),
                    (*base, "p99_hz"),
                ]
            )

    return paths


def compare_against_baseline(current: Dict, baseline: Dict):
    current = _normalize_report_shape(current)
    baseline = _normalize_report_shape(baseline)

    required = _required_metric_paths(current)
    missing = []
    comparison = {}

    for path in required:
        try:
            current_value = _require_path(current, path)
            baseline_value = _require_path(baseline, path)
        except KeyError:
            missing.append(_segments_to_key(path))
            continue

        if isinstance(current_value, (int, float)) and isinstance(baseline_value, (int, float)):
            ratio = None
            if baseline_value != 0:
                ratio = float(current_value) / float(baseline_value)
            comparison[_segments_to_key(path)] = {
                "current": float(current_value),
                "baseline": float(baseline_value),
                "ratio": ratio,
            }

    return missing, comparison


def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    configure_logging_from_args(
        args,
        app_name="pyrtc-check-perf-baseline",
        component_name="benchmarks.check_perf_baseline",
    )

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

    logger.info("Baseline comparison details:\n%s", json.dumps(comparison, indent=2))

    if missing:
        raise SystemExit(
            "Missing baseline metrics for comparison:\n" + "\n".join(sorted(missing))
        )

    logger.info("Baseline comparison succeeded: all required metrics found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
