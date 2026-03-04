import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pyRTC import initExistingShm


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


def _safe_min(values) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return 0.0
    return min(float(x) for x in arr)


def _safe_max(values) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return 0.0
    return max(float(x) for x in arr)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure latency between two pyRTC shared-memory streams."
    )
    parser.add_argument("source_shm", type=str, help="Name of source SHM (earlier in pipeline)")
    parser.add_argument("target_shm", type=str, help="Name of target SHM (later in pipeline)")
    parser.add_argument("--samples", type=int, default=30000, help="Number of timestamp pairs to collect")
    parser.add_argument("--tag", type=str, default="latency", help="Output tag used in filename/title")
    parser.add_argument("--bins", type=int, default=200, help="Number of histogram bins")
    parser.add_argument(
        "--xrange",
        type=float,
        nargs=2,
        default=(1e-4, 10 ** -2.5),
        metavar=("LOW", "HIGH"),
        help="Latency histogram range in seconds",
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bar")
    parser.add_argument("--no-show", action="store_true", help="Do not show plot window")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output figure path (default: jitter_<source>_to_<target>_<tag>.pdf)",
    )
    return parser


def collect_timestamps(source_shm, target_shm, samples: int, show_progress: bool = True):
    source_write_times = np.empty(samples, dtype=np.float64)
    target_write_times = np.empty(samples, dtype=np.float64)
    source_counts = np.empty(samples, dtype=np.float64)
    target_counts = np.empty(samples, dtype=np.float64)

    if show_progress:
        try:
            import tqdm

            iterator = tqdm.trange(samples)
        except ImportError:
            iterator = range(samples)
    else:
        iterator = range(samples)

    for index in iterator:
        source_shm.hold()
        source_counts[index] = source_shm.metadata[0]
        source_write_times[index] = source_shm.metadata[1]

        target_shm.hold()
        target_counts[index] = target_shm.metadata[0]
        target_write_times[index] = target_shm.metadata[1]

    return source_counts, source_write_times, target_counts, target_write_times


def compute_latency_seconds(source_write_times: np.ndarray, target_write_times: np.ndarray):
    sys_latency = target_write_times - source_write_times
    frame_shift = 0

    while _safe_mean(sys_latency) < 0 and frame_shift < source_write_times.size - 1:
        frame_shift += 1
        sys_latency = target_write_times[frame_shift:] - source_write_times[:-frame_shift]

    return sys_latency, frame_shift


def plot_latency_histogram(sys_latency: np.ndarray, args) -> plt.Figure:
    low, high = args.xrange
    bins = np.logspace(np.log10(low), np.log10(high), args.bins)

    fig = plt.figure(figsize=(10, 6))
    plt.hist(
        sys_latency,
        bins=bins,
        log=True,
        color="k",
        histtype="step",
        density=False,
    )

    p99 = _safe_percentile(sys_latency, 99)
    p999 = _safe_percentile(sys_latency, 99.9)
    p9999 = _safe_percentile(sys_latency, 99.99)

    plt.axvline(x=p99, color="green", label=f"1 in 100 > {1e6 * p99:.0f}us")
    plt.axvline(x=p999, color="orange", label=f"1 in 1,000 > {1e6 * p999:.0f}us")
    plt.axvline(x=p9999, color="red", label=f"1 in 10,000 > {1e6 * p9999:.0f}us")

    plt.xscale("log")
    plt.yscale("log")

    xticks = [1e-4, 5e-4, 1e-3, 5e-3]
    xtick_labels = ["100us", "500us", "1ms", "5ms"]
    plt.xticks(xticks, xtick_labels)

    plt.xlabel("System Latency [s]", size=16)
    plt.ylabel("Counts", size=16)
    plt.title(f"pyRTC Latency ({args.source_shm} -> {args.target_shm}, tag={args.tag})", size=18)
    plt.ylim(0.5, max(2.0, args.samples / 2))
    plt.xlim(_safe_min(xticks) * 0.9, _safe_max(xticks) * 1.1)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    return fig


def _default_output_path(args) -> Path:
    safe_source = args.source_shm.replace("/", "_")
    safe_target = args.target_shm.replace("/", "_")
    return Path(f"jitter_{safe_source}_to_{safe_target}_{args.tag}.pdf")


def main(argv=None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.samples < 2:
        raise SystemExit("--samples must be at least 2")

    source_shm, _, _ = initExistingShm(args.source_shm)
    target_shm, _, _ = initExistingShm(args.target_shm)

    source_counts, source_write_times, target_counts, target_write_times = collect_timestamps(
        source_shm,
        target_shm,
        args.samples,
        show_progress=not args.no_progress,
    )

    count_delta = source_counts - target_counts
    print(_safe_min(count_delta), _safe_max(count_delta))

    sys_latency, frame_shift = compute_latency_seconds(source_write_times, target_write_times)
    print(f"Applied frame shift: {frame_shift}")
    print(f"Mean latency: {_safe_mean(sys_latency) * 1e6:.2f} us")

    plot_latency_histogram(sys_latency, args)

    output_path = Path(args.output) if args.output else _default_output_path(args)
    plt.savefig(output_path)

    if not args.no_show:
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
