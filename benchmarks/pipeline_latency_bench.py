"""End-to-end pipeline latency of the running synthetic SHWFS system.

``benchmarks/ao_loop_bench.py`` times the compute kernels of one loop
iteration in a single thread. This benchmark instead launches the synthetic
SHWFS example (``examples/synthetic_shwfs/config.yaml``) through
``RTCManager`` — worker threads in soft-RTC mode, one process per component in
hard-RTC mode — lets it settle, and measures ``manager.latency()``: the
frame-id aligned time from the WFS publishing a frame to the loop publishing
the DM command for it (``wfs -> signal -> wfc``), per segment and in total.
That includes stream handoffs, thread scheduling, and GIL contention, which the
kernel benchmark does not see.

Every run uses stream names with a unique prefix, so it cannot collide with
another pyrtc system on the host, and removes its streams afterwards.

Usage::

    python -m benchmarks.pipeline_latency_bench --samples 2048 \\
        --output benchmarks/pipeline_latency_report.json
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pyrtc.logging_utils import (  # noqa: E402
    add_logging_cli_args,
    configure_logging_from_args,
    get_logger,
)
from pyrtc.streams import STREAM_NOTIFY_ENV, clear_shms  # noqa: E402

logger = get_logger(__name__)

EXAMPLE_DIR = REPO_ROOT / "examples" / "synthetic_shwfs"
EXAMPLE_CONFIG = EXAMPLE_DIR / "config.yaml"
LATENCY_PATH = ("wfs", "signal", "wfc")
MODES = ("soft", "hard")


def _absolute(path_value: str) -> str:
    path = Path(path_value)
    if not path.is_absolute():
        path = (EXAMPLE_DIR / path).resolve()
    return str(path)


def build_benchmark_config(
    prefix: str,
    workdir: Path,
    *,
    frame_rate_hz: float | None = None,
    include_psf: bool = True,
) -> tuple[Path, dict[str, str]]:
    """Write a copy of the synthetic SHWFS config with private stream names.

    Every input and output stream is renamed to ``<prefix>_<name>``, class
    files and the IM file become absolute paths (hard-RTC children load the
    YAML from ``workdir``), and the loop's interaction matrix is generated.
    Returns the config path and the canonical-to-private stream name map.
    """
    raw = yaml.safe_load(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    config = copy.deepcopy(raw)
    if not include_psf:
        config.pop("psf", None)
        for key in ("component_classes", "component_files"):
            config.get("manager", {}).get(key, {}).pop("psf", None)
        config.get("manager", {}).pop("graph_layout", None)

    names: dict[str, str] = {}

    def _private(stream: str) -> str:
        return names.setdefault(stream, f"{prefix}_{stream}")

    for section_name, section in config.items():
        if not isinstance(section, dict) or "class_name" not in section:
            continue
        for direction in ("input_streams", "output_streams"):
            aliases = section.get(direction) or {}
            section[direction] = {key: _private(value) for key, value in aliases.items()}
        if section.get("class_file"):
            section["class_file"] = _absolute(section["class_file"])
    # The synthetic WFS and science camera open these streams by their
    # canonical names unless an input alias says otherwise.
    config["wfs"]["input_streams"] = {"wfc": _private("wfc")}
    if "psf" in config:
        config["psf"]["input_streams"] = {"signal": _private("signal")}
    if frame_rate_hz is not None:
        config["wfs"]["frame_rate_hz"] = float(frame_rate_hz)

    manager_conf = config.setdefault("manager", {})
    files = manager_conf.get("component_files", {})
    manager_conf["component_files"] = {key: _absolute(value) for key, value in files.items()}
    config["loop"]["im_file"] = str(workdir / "synthetic_im.npy")
    _write_interaction_matrix(config)

    config_path = workdir / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path, names


def _write_interaction_matrix(config: dict) -> None:
    from pyrtc.hardware.synthetic_systems import (
        _default_wfc_layout,
        build_synthetic_shwfs_response_matrix,
    )

    wfs_conf = config["wfs"]
    width = int(wfs_conf["width"])
    height = int(wfs_conf["height"])
    downsample = int(wfs_conf.get("downsample_factor", 0) or 0)
    if downsample > 0:
        width //= downsample
        height //= downsample
    num_regions = min(width, height) // int(config["slopes"]["sub_ap_spacing"])
    layout = _default_wfc_layout(int(config["wfc"]["num_actuators"]))
    matrix = build_synthetic_shwfs_response_matrix(
        num_regions, int(config["wfc"]["num_modes"]), layout
    )
    np.save(config["loop"]["im_file"], matrix.astype(np.float32))


@contextlib.contextmanager
def _stream_notify_env(notify: bool):
    """Set ``PYRTC_STREAM_NOTIFY`` for this process and launched children."""
    previous = os.environ.get(STREAM_NOTIFY_ENV)
    os.environ[STREAM_NOTIFY_ENV] = "1" if notify else "0"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(STREAM_NOTIFY_ENV, None)
        else:
            os.environ[STREAM_NOTIFY_ENV] = previous


def _summarize(statistics: dict[str, Any]) -> dict[str, float]:
    return {
        "count": int(statistics["sample_count"]),
        "mean_us": float(statistics["mean_seconds"]) * 1e6,
        "p50_us": float(statistics["p50_seconds"]) * 1e6,
        "p99_us": float(statistics["p99_seconds"]) * 1e6,
        "max_us": float(statistics["max_seconds"]) * 1e6,
        "jitter_us": float(statistics["jitter_seconds"]) * 1e6,
    }


def _stored_notify(stream_name: str) -> bool | None:
    import pyshmem

    try:
        return bool(pyshmem.stat(stream_name)["notify"])
    except Exception:
        return None


def run_pipeline_latency(
    mode: str,
    notify: bool,
    *,
    samples: int = 2048,
    settle_seconds: float = 3.0,
    frame_rate_hz: float | None = None,
    include_psf: bool = True,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    """Launch the synthetic system once and measure its WFS -> DM latency."""
    from pyrtc.manager import RTCManager

    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    prefix = f"bench_{uuid.uuid4().hex[:10]}"
    with tempfile.TemporaryDirectory(prefix="pyrtc_pipeline_bench_") as tmp:
        config_path, names = build_benchmark_config(
            prefix, Path(tmp), frame_rate_hz=frame_rate_hz, include_psf=include_psf
        )
        path = [names[stream] for stream in LATENCY_PATH]
        manager = None
        try:
            with _stream_notify_env(notify):
                manager = RTCManager.from_config_file(config_path, mode=mode)
                manager.start()
                time.sleep(settle_seconds)
                started = time.perf_counter()
                report = manager.latency(
                    stream_path=path, samples=samples, timeout_seconds=timeout_seconds
                )
                elapsed = time.perf_counter() - started
                notify_active = _stored_notify(path[0])
        finally:
            if manager is not None:
                try:
                    manager.stop()
                except Exception:
                    logger.warning("Failed to stop benchmark manager", exc_info=True)
            clear_shms(sorted(set(names.values())))

    segments = [
        {
            "source": segment["source_shm"].removeprefix(prefix + "_"),
            "target": segment["target_shm"].removeprefix(prefix + "_"),
            **_summarize(segment["statistics"]),
        }
        for segment in report["segments"]
    ]
    return {
        "mode": f"{mode}-rtc",
        "stream_prefix": prefix,
        "notify": bool(notify),
        "notify_flag_on_streams": notify_active,
        "stream_path": list(LATENCY_PATH),
        "frame_rate_hz": frame_rate_hz,
        "include_psf": include_psf,
        "measurement_seconds": elapsed,
        "total": _summarize(report["total"]["statistics"]),
        "segments": segments,
    }


def run_pipeline_benchmarks(
    *,
    modes=MODES,
    notify_settings=(False, True),
    samples: int = 2048,
    settle_seconds: float = 3.0,
    frame_rate_hz: float | None = None,
    include_psf: bool = True,
    timeout_seconds: float | None = None,
    repeats: int = 1,
) -> dict[str, Any]:
    """Measure every mode/notify combination ``repeats`` times.

    Repeats are interleaved (all combinations once, then again), so slow
    drift in host load affects every combination alike. ``summary`` holds
    the median across repeats of each statistic.
    """
    from benchmarks.core_compute_bench import collect_system_info

    results = []
    for repeat in range(max(1, int(repeats))):
        for mode in modes:
            for notify in notify_settings:
                logger.info(
                    "Measuring %s-rtc pipeline latency (notify=%s, repeat %d)",
                    mode,
                    notify,
                    repeat,
                )
                row = run_pipeline_latency(
                    mode,
                    notify,
                    samples=samples,
                    settle_seconds=settle_seconds,
                    frame_rate_hz=frame_rate_hz,
                    include_psf=include_psf,
                    timeout_seconds=timeout_seconds,
                )
                row["repeat"] = repeat
                results.append(row)
    return {
        "meta": {
            "benchmark_type": "pipeline_latency",
            "config": str(EXAMPLE_CONFIG.relative_to(REPO_ROOT)),
            "samples": int(samples),
            "settle_seconds": float(settle_seconds),
            "repeats": max(1, int(repeats)),
            "frame_rate_hz": frame_rate_hz,
            "system": collect_system_info(),
        },
        "results": results,
        "summary": summarize_repeats(results),
    }


_STAT_KEYS = ("mean_us", "p50_us", "p99_us", "max_us", "jitter_us")


def _median_stats(entries: list[dict[str, Any]]) -> dict[str, float]:
    merged = {key: float(np.median([entry[key] for entry in entries])) for key in _STAT_KEYS}
    merged["count"] = int(sum(entry["count"] for entry in entries))
    return merged


def summarize_repeats(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Median of each statistic across repeats, per mode and notify setting."""
    groups: dict[tuple[str, bool], list[dict[str, Any]]] = {}
    for row in results:
        groups.setdefault((row["mode"], row["notify"]), []).append(row)
    summary = []
    for (mode, notify), rows in groups.items():
        segments = []
        for index, segment in enumerate(rows[0]["segments"]):
            segments.append(
                {
                    "source": segment["source"],
                    "target": segment["target"],
                    **_median_stats([row["segments"][index] for row in rows]),
                }
            )
        summary.append(
            {
                "mode": mode,
                "notify": notify,
                "runs": len(rows),
                "stream_path": rows[0]["stream_path"],
                "total": _median_stats([row["total"] for row in rows]),
                "segments": segments,
            }
        )
    return summary


def format_report(report: dict[str, Any]) -> str:
    lines = [
        "mode      notify  segment          mean_us   p50_us   p99_us  jitter_us  samples",
    ]
    for row in report.get("summary") or report["results"]:
        entries = [("total " + "->".join(row["stream_path"]), row["total"])]
        entries += [(f"{seg['source']}->{seg['target']}", seg) for seg in row["segments"]]
        for label, stats in entries:
            lines.append(
                f"{row['mode']:<9} {'on' if row['notify'] else 'off':<7} {label:<16} "
                f"{stats['mean_us']:8.1f} {stats['p50_us']:8.1f} {stats['p99_us']:8.1f} "
                f"{stats['jitter_us']:10.1f} {stats['count']:8d}"
            )
    return "\n".join(lines)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measure end-to-end pipeline latency.")
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument(
        "--settle", type=float, default=3.0, help="Seconds to run before measuring."
    )
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument(
        "--notify",
        nargs="+",
        choices=("on", "off"),
        default=["off", "on"],
        help="pyshmem notify settings to measure.",
    )
    parser.add_argument(
        "--frame-rate-hz",
        type=float,
        default=None,
        help="Override the WFS frame rate (default: the example's 200 Hz).",
    )
    parser.add_argument("--no-psf", action="store_true", help="Leave out the science camera.")
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Interleaved repeats of every combination; the summary is their median.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Fail a run that has not collected its samples after this many seconds.",
    )
    parser.add_argument("--output", type=str, default=None)
    add_logging_cli_args(parser)
    return parser


def main(argv=None) -> int:
    args = _build_arg_parser().parse_args(argv)
    configure_logging_from_args(
        args, app_name="pyrtc-pipeline-bench", component_name="benchmarks.pipeline_latency_bench"
    )
    report = run_pipeline_benchmarks(
        modes=tuple(args.modes),
        notify_settings=tuple(setting == "on" for setting in args.notify),
        samples=args.samples,
        settle_seconds=args.settle,
        frame_rate_hz=args.frame_rate_hz,
        include_psf=not args.no_psf,
        timeout_seconds=args.timeout,
        repeats=args.repeats,
    )
    print(format_report(report))
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.info("Wrote pipeline latency report to %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
