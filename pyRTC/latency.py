"""Latency measurement helpers for pyRTC shared-memory streams.

The helpers in this module stay in the control plane. They attach to existing
shared-memory streams, sample timestamp metadata, and summarize latency and
jitter without changing the steady-state RTC hot path.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import time
from typing import Any, Callable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np


def initExistingShm(shmName, gpuDevice=None):
    """Attach to an existing SHM stream.

    The import lives here so tests can monkeypatch this module directly without
    creating an import cycle with ``pyRTC.Pipeline`` at module import time.
    """

    from pyRTC.Pipeline import initExistingShm as _initExistingShm

    return _initExistingShm(shmName, gpuDevice=gpuDevice)


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


def _safe_std(values) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return 0.0
    return float(np.std(arr, dtype=np.float64))


def _format_seconds(seconds: float) -> str:
    """Format a latency value using human-scaled time units."""

    value = float(seconds)
    magnitude = abs(value)
    if magnitude >= 1.0:
        return f"{value:.3f} s"
    if magnitude >= 1e-3:
        return f"{value * 1e3:.3f} ms"
    return f"{value * 1e6:.3f} us"


def _format_seconds_with_rate(seconds: float) -> str:
    """Format latency with the equivalent maximum update rate in kHz."""

    value = float(seconds)
    if value <= 0.0:
        rate_text = "inf kHz"
    else:
        rate_text = f"{(1.0 / value) / 1e3:.3f} kHz"
    return f"{_format_seconds(value)} (max speed {rate_text})"


@dataclass(frozen=True)
class LatencyStatistics:
    sample_count: int
    mean_seconds: float
    std_seconds: float
    jitter_seconds: float
    min_seconds: float
    max_seconds: float
    p50_seconds: float
    p95_seconds: float
    p99_seconds: float
    p999_seconds: float

    @classmethod
    def from_samples(cls, samples) -> "LatencyStatistics":
        arr = np.asarray(samples, dtype=np.float64).reshape(-1)
        return cls(
            sample_count=int(arr.size),
            mean_seconds=_safe_mean(arr),
            std_seconds=_safe_std(arr),
            jitter_seconds=_safe_std(arr),
            min_seconds=_safe_min(arr),
            max_seconds=_safe_max(arr),
            p50_seconds=_safe_percentile(arr, 50.0),
            p95_seconds=_safe_percentile(arr, 95.0),
            p99_seconds=_safe_percentile(arr, 99.0),
            p999_seconds=_safe_percentile(arr, 99.9),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_count": self.sample_count,
            "mean_seconds": self.mean_seconds,
            "std_seconds": self.std_seconds,
            "jitter_seconds": self.jitter_seconds,
            "min_seconds": self.min_seconds,
            "max_seconds": self.max_seconds,
            "p50_seconds": self.p50_seconds,
            "p95_seconds": self.p95_seconds,
            "p99_seconds": self.p99_seconds,
            "p999_seconds": self.p999_seconds,
        }


@dataclass(frozen=True)
class LatencySegment:
    source_shm: str
    target_shm: str
    frame_shift: int
    count_offset: int
    count_delta_min: float
    count_delta_max: float
    statistics: LatencyStatistics
    processing_statistics: LatencyStatistics | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "source_shm": self.source_shm,
            "target_shm": self.target_shm,
            "frame_shift": self.frame_shift,
            "count_offset": self.count_offset,
            "count_delta_min": self.count_delta_min,
            "count_delta_max": self.count_delta_max,
            "statistics": self.statistics.to_dict(),
        }
        if self.processing_statistics is not None:
            payload["processing_statistics"] = self.processing_statistics.to_dict()
        return payload


@dataclass(frozen=True)
class LatencyReport:
    source_shm: str
    target_shm: str
    stream_path: tuple[str, ...]
    inferred_path: bool
    sample_count: int
    total: LatencySegment
    segments: tuple[LatencySegment, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_shm": self.source_shm,
            "target_shm": self.target_shm,
            "stream_path": list(self.stream_path),
            "inferred_path": self.inferred_path,
            "sample_count": self.sample_count,
            "total": self.total.to_dict(),
            "segments": [segment.to_dict() for segment in self.segments],
        }


def collect_timestamps(streams, samples: int, show_progress: bool = False):
    """Collect timestamp and counter samples for one or more SHM streams."""

    stream_items = list(streams.items())
    if not stream_items:
        raise ValueError("At least one stream is required for latency sampling")

    counts = {
        stream_name: np.empty(samples, dtype=np.float64)
        for stream_name, _ in stream_items
    }
    write_times = {
        stream_name: np.empty(samples, dtype=np.float64)
        for stream_name, _ in stream_items
    }

    if show_progress:
        try:
            import tqdm

            iterator = tqdm.trange(samples)
        except ImportError:
            iterator = range(samples)
    else:
        iterator = range(samples)

    for index in iterator:
        for stream_name, stream in stream_items:
            stream.hold()
            counts[stream_name][index] = stream.metadata[0]
            write_times[stream_name][index] = stream.metadata[1]

    return counts, write_times


def collect_stream_event_history(
    streams,
    samples: int,
    *,
    poll_interval_seconds: float = 1e-4,
    timeout_seconds: float | None = None,
    show_progress: bool = False,
):
    """Collect per-stream write events over one shared wall-clock window.

    Unlike ``collect_timestamps``, this sampler does not block on each stream in
    sequence. It polls metadata for all requested streams and records each new
    write event as it appears, which preserves cross-stream timing much better
    for asynchronous pipelines.

    For older test doubles that only advance metadata when ``hold()`` is called,
    the sampler performs a lightweight compatibility nudge before reading
    metadata. Real ``ImageSHM`` instances expose ``frame_metadata()`` and stay on
    the non-blocking path.
    """

    stream_items = list(streams.items())
    if not stream_items:
        raise ValueError("At least one stream is required for latency sampling")
    if samples < 1:
        raise ValueError("samples must be at least 1")

    counts = {
        stream_name: np.empty(samples, dtype=np.float64)
        for stream_name, _ in stream_items
    }
    write_times = {
        stream_name: np.empty(samples, dtype=np.float64)
        for stream_name, _ in stream_items
    }
    collected = {stream_name: 0 for stream_name, _ in stream_items}
    last_seen_count = {
        stream_name: int(np.rint(float(stream.metadata[0])))
        for stream_name, stream in stream_items
    }

    progress_bar = None
    total_needed = samples * len(stream_items)
    if show_progress:
        try:
            import tqdm

            progress_bar = tqdm.tqdm(total=total_needed)
        except ImportError:
            progress_bar = None

    start_time = time.perf_counter()
    try:
        while any(collected[stream_name] < samples for stream_name, _ in stream_items):
            progressed = False
            for stream_name, stream in stream_items:
                if collected[stream_name] >= samples:
                    continue
                if not hasattr(stream, "frame_metadata") and callable(getattr(stream, "hold", None)):
                    stream.hold()
                metadata = stream.metadata
                current_count = int(np.rint(float(metadata[0])))
                if current_count == last_seen_count[stream_name]:
                    continue
                last_seen_count[stream_name] = current_count
                index = collected[stream_name]
                counts[stream_name][index] = current_count
                write_times[stream_name][index] = float(metadata[1])
                collected[stream_name] = index + 1
                progressed = True
                if progress_bar is not None:
                    progress_bar.update(1)

            if all(collected[stream_name] >= samples for stream_name, _ in stream_items):
                break
            if timeout_seconds is not None and (time.perf_counter() - start_time) >= timeout_seconds:
                raise TimeoutError("Timed out while collecting latency samples")
            if not progressed:
                time.sleep(max(0.0, poll_interval_seconds))
    finally:
        if progress_bar is not None:
            progress_bar.close()

    return counts, write_times


def _stream_metadata_snapshot(stream) -> dict[str, float | int]:
    frame_metadata = getattr(stream, "frame_metadata", None)
    if callable(frame_metadata):
        snapshot = dict(frame_metadata())
    else:
        metadata = getattr(stream, "metadata", None)
        if metadata is None:
            snapshot = {
                "count": 0,
                "write_time": 0.0,
                "root_time": 0.0,
                "upstream_write_time": 0.0,
                "upstream_consume_time": 0.0,
            }
        else:
            snapshot = {
                "count": int(metadata[0]),
                "write_time": float(metadata[1]),
                "root_time": float(metadata[2]) if len(metadata) > 2 else 0.0,
                "upstream_write_time": float(metadata[3]) if len(metadata) > 3 else 0.0,
                "upstream_consume_time": float(metadata[4]) if len(metadata) > 4 else 0.0,
            }
    return snapshot


def _has_lineage_metadata(stream) -> bool:
    snapshot = _stream_metadata_snapshot(stream)
    return (
        float(snapshot.get("root_time", 0.0)) > 0.0
        or float(snapshot.get("upstream_write_time", 0.0)) > 0.0
        or float(snapshot.get("upstream_consume_time", 0.0)) > 0.0
    )


def collect_stream_metadata_history(stream, samples: int, *, show_progress: bool = False, timeout_seconds: float | None = None):
    if samples < 1:
        raise ValueError("samples must be at least 1")

    history = []
    progress_bar = None
    if show_progress:
        try:
            import tqdm

            progress_bar = tqdm.tqdm(total=samples)
        except ImportError:
            progress_bar = None

    start_time = time.perf_counter()
    try:
        while len(history) < samples:
            if timeout_seconds is not None and (time.perf_counter() - start_time) >= timeout_seconds:
                raise TimeoutError("Timed out while collecting stream metadata")
            stream.hold()
            history.append(_stream_metadata_snapshot(stream))
            if progress_bar is not None:
                progress_bar.update(1)
    finally:
        if progress_bar is not None:
            progress_bar.close()

    return history


def compute_latency_seconds(source_write_times: np.ndarray, target_write_times: np.ndarray):
    """Estimate latency samples and compensate for simple frame misalignment."""

    source_arr = np.asarray(source_write_times, dtype=np.float64).reshape(-1)
    target_arr = np.asarray(target_write_times, dtype=np.float64).reshape(-1)
    if source_arr.size != target_arr.size:
        raise ValueError("source_write_times and target_write_times must have the same length")

    sys_latency = target_arr - source_arr
    frame_shift = 0

    while _safe_mean(sys_latency) < 0 and frame_shift < source_arr.size - 1:
        frame_shift += 1
        sys_latency = target_arr[frame_shift:] - source_arr[:-frame_shift]

    return sys_latency, frame_shift


def compute_count_aligned_latency_seconds(
    source_counts: np.ndarray,
    source_write_times: np.ndarray,
    target_counts: np.ndarray,
    target_write_times: np.ndarray,
    *,
    neighbor_search: int = 2,
) -> tuple[np.ndarray, int, np.ndarray]:
    """Estimate latency after aligning source and target events by SHM count.

    Each stream maintains its own monotonically increasing count. Two connected
    streams often have a large constant count offset because they were created
    or started at different times. This function removes that startup offset and
    then matches writes by normalized count so the reported latency reflects the
    live pipeline delay rather than wall-clock age differences.
    """

    source_count_arr = np.rint(np.asarray(source_counts, dtype=np.float64).reshape(-1)).astype(np.int64)
    target_count_arr = np.rint(np.asarray(target_counts, dtype=np.float64).reshape(-1)).astype(np.int64)
    source_time_arr = np.asarray(source_write_times, dtype=np.float64).reshape(-1)
    target_time_arr = np.asarray(target_write_times, dtype=np.float64).reshape(-1)
    if not (
        source_count_arr.size == target_count_arr.size == source_time_arr.size == target_time_arr.size
    ):
        raise ValueError("count and timestamp arrays must all have the same length")
    if source_count_arr.size == 0:
        return np.empty(0, dtype=np.float64), 0, np.empty(0, dtype=np.int64)

    count_offset = int(np.rint(np.median(target_count_arr - source_count_arr)))
    source_by_count = {
        int(count): float(timestamp)
        for count, timestamp in zip(source_count_arr, source_time_arr)
    }

    matched_latencies = []
    matched_residual_deltas = []
    for target_count, target_time in zip(target_count_arr, target_time_arr):
        best_latency = None
        best_residual = None
        target_count_int = int(target_count)
        for residual in range(0, neighbor_search + 1):
            for signed_residual in (-residual, residual) if residual > 0 else (0,):
                candidate_source_count = target_count_int - count_offset - signed_residual
                source_time = source_by_count.get(candidate_source_count)
                if source_time is None:
                    continue
                latency = float(target_time - source_time)
                if latency < 0:
                    continue
                if best_latency is None or latency < best_latency:
                    best_latency = latency
                    best_residual = signed_residual
            if best_latency is not None:
                break
        if best_latency is None:
            continue
        matched_latencies.append(best_latency)
        matched_residual_deltas.append(0 if best_residual is None else int(best_residual))

    if matched_latencies:
        return (
            np.asarray(matched_latencies, dtype=np.float64),
            count_offset,
            np.asarray(matched_residual_deltas, dtype=np.int64),
        )

    fallback_latency, frame_shift = compute_latency_seconds(source_time_arr, target_time_arr)
    fallback_residual = np.full(fallback_latency.shape, frame_shift, dtype=np.int64)
    return fallback_latency, frame_shift, fallback_residual


def _build_latency_segment(
    source_shm: str,
    target_shm: str,
    source_counts: np.ndarray,
    source_write_times: np.ndarray,
    target_counts: np.ndarray,
    target_write_times: np.ndarray,
) -> tuple[LatencySegment, np.ndarray]:
    latency_seconds, count_offset, residual_count_delta = compute_count_aligned_latency_seconds(
        source_counts,
        source_write_times,
        target_counts,
        target_write_times,
    )

    segment = LatencySegment(
        source_shm=source_shm,
        target_shm=target_shm,
        frame_shift=0,
        count_offset=count_offset,
        count_delta_min=_safe_min(residual_count_delta),
        count_delta_max=_safe_max(residual_count_delta),
        statistics=LatencyStatistics.from_samples(latency_seconds),
    )
    return segment, np.asarray(latency_seconds, dtype=np.float64)


def _metadata_samples(history: Sequence[Mapping[str, Any]], key: str, fallback_key: str | None = None) -> np.ndarray:
    values = []
    for entry in history:
        write_time = float(entry.get("write_time", 0.0))
        reference = float(entry.get(key, 0.0))
        if reference <= 0.0 and fallback_key is not None:
            reference = float(entry.get(fallback_key, 0.0))
        if reference <= 0.0:
            continue
        values.append(max(0.0, write_time - reference))
    return np.asarray(values, dtype=np.float64)


def _build_metadata_segment(source_shm: str, target_shm: str, history: Sequence[Mapping[str, Any]]) -> tuple[LatencySegment, np.ndarray]:
    latency_seconds = _metadata_samples(history, "upstream_write_time")
    processing_seconds = _metadata_samples(history, "upstream_consume_time")
    segment = LatencySegment(
        source_shm=source_shm,
        target_shm=target_shm,
        frame_shift=0,
        count_offset=0,
        count_delta_min=0.0,
        count_delta_max=0.0,
        statistics=LatencyStatistics.from_samples(latency_seconds),
        processing_statistics=LatencyStatistics.from_samples(processing_seconds) if processing_seconds.size > 0 else None,
    )
    return segment, latency_seconds


def _build_metadata_total_segment(source_shm: str, target_shm: str, history: Sequence[Mapping[str, Any]]) -> tuple[LatencySegment, np.ndarray]:
    latency_seconds = _metadata_samples(history, "root_time", fallback_key="upstream_write_time")
    processing_seconds = _metadata_samples(history, "upstream_consume_time")
    segment = LatencySegment(
        source_shm=source_shm,
        target_shm=target_shm,
        frame_shift=0,
        count_offset=0,
        count_delta_min=0.0,
        count_delta_max=0.0,
        statistics=LatencyStatistics.from_samples(latency_seconds),
        processing_statistics=LatencyStatistics.from_samples(processing_seconds) if processing_seconds.size > 0 else None,
    )
    return segment, latency_seconds


def plot_latency_histogram(latency_seconds: np.ndarray, *, title: str, bins: int, xrange: Sequence[float]) -> plt.Figure:
    """Render a log-scaled histogram that highlights high-percentile latency."""

    low, high = (float(xrange[0]), float(xrange[1]))
    fig = plt.figure(figsize=(10, 6))
    plt.hist(
        latency_seconds,
        bins=np.logspace(np.log10(low), np.log10(high), bins),
        log=True,
        color="k",
        histtype="step",
        density=False,
    )

    p99 = _safe_percentile(latency_seconds, 99.0)
    p999 = _safe_percentile(latency_seconds, 99.9)
    p9999 = _safe_percentile(latency_seconds, 99.99)

    plt.axvline(x=p99, color="green", label=f"1 in 100 > {1e6 * p99:.0f}us")
    plt.axvline(x=p999, color="orange", label=f"1 in 1,000 > {1e6 * p999:.0f}us")
    plt.axvline(x=p9999, color="red", label=f"1 in 10,000 > {1e6 * p9999:.0f}us")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Latency [s]", size=16)
    plt.ylabel("Counts", size=16)
    plt.title(title, size=18)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    return fig


def _descriptor_transition(descriptor) -> tuple[str, str] | None:
    input_names = [stream.name for stream in descriptor.input_streams if stream.name != "*"]
    if not input_names:
        return None

    preferred_outputs = [stream.name for stream in descriptor.output_streams if not stream.optional and stream.name != "*"]
    fallback_outputs = [stream.name for stream in descriptor.output_streams if stream.name != "*"]
    output_names = preferred_outputs or fallback_outputs
    if not output_names:
        return None

    source_name = input_names[0]
    target_name = next((name for name in output_names if name != source_name), None)
    if target_name is None:
        return None
    return source_name, target_name


def build_stream_transitions(
    section_names: Sequence[str],
    descriptor_resolver: Callable[[str], Any | None],
) -> list[tuple[str, str]]:
    """Build stream-to-stream transitions implied by component descriptors."""

    transitions = []
    for section_name in section_names:
        descriptor = descriptor_resolver(section_name)
        if descriptor is None:
            continue
        transition = _descriptor_transition(descriptor)
        if transition is None:
            continue
        transitions.append(transition)
    return transitions


def _shortest_path(adjacency: Mapping[str, list[str]], source_shm: str, target_shm: str) -> list[str] | None:
    queue = deque([(source_shm, [source_shm])])
    visited = {source_shm}
    while queue:
        current, path = queue.popleft()
        if current == target_shm:
            return path
        for neighbor in adjacency.get(current, []):
            if neighbor in visited:
                continue
            visited.add(neighbor)
            queue.append((neighbor, path + [neighbor]))
    return None


def _longest_path_from(adjacency: Mapping[str, list[str]], node: str, visited: set[str]) -> list[str]:
    best_path = [node]
    for neighbor in adjacency.get(node, []):
        if neighbor in visited:
            continue
        candidate = [node] + _longest_path_from(adjacency, neighbor, visited | {neighbor})
        if len(candidate) > len(best_path):
            best_path = candidate
    return best_path


def infer_stream_path(
    *,
    section_names: Sequence[str],
    descriptor_resolver: Callable[[str], Any | None],
    source_shm: str | None = None,
    target_shm: str | None = None,
) -> tuple[list[str], bool]:
    """Infer a sensible stream path for latency breakdowns.

    When both ``source_shm`` and ``target_shm`` are provided, the function first
    tries to find a descriptor-backed path between them. If none exists, the
    caller can still measure the pair directly.
    """

    transitions = build_stream_transitions(section_names, descriptor_resolver)
    adjacency: dict[str, list[str]] = {}
    indegree: dict[str, int] = {}
    for source_name, target_name in transitions:
        adjacency.setdefault(source_name, []).append(target_name)
        adjacency.setdefault(target_name, [])
        indegree.setdefault(source_name, 0)
        indegree[target_name] = indegree.get(target_name, 0) + 1

    if source_shm is not None and target_shm is not None:
        inferred = _shortest_path(adjacency, source_shm, target_shm)
        if inferred is not None:
            return inferred, True
        return [source_shm, target_shm], False

    if not transitions:
        raise ValueError("No descriptor-backed stream path could be inferred for this system")

    candidate_sources = [stream_name for stream_name, degree in indegree.items() if degree == 0]
    if not candidate_sources:
        candidate_sources = [transitions[0][0]]

    best_path: list[str] = []
    for stream_name in candidate_sources:
        candidate = _longest_path_from(adjacency, stream_name, {stream_name})
        if len(candidate) > len(best_path):
            best_path = candidate

    if len(best_path) < 2:
        raise ValueError("Unable to infer a latency path with at least two streams")
    return best_path, True


def measure_stream_path_latency(
    stream_path: Sequence[str],
    *,
    samples: int = 2048,
    show_progress: bool = False,
    shm_opener: Callable[[str], tuple[Any, Any, Any]] | None = None,
    include_total_samples: bool = False,
    timeout_seconds: float | None = None,
) -> tuple[LatencyReport, np.ndarray | None]:
    """Measure latency across a stream path and return a structured report."""

    if samples < 2:
        raise ValueError("samples must be at least 2")

    normalized_path = [str(stream_name) for stream_name in stream_path]
    if len(normalized_path) < 2:
        raise ValueError("stream_path must contain at least two stream names")

    unique_stream_names = list(dict.fromkeys(normalized_path))
    opener = shm_opener or initExistingShm
    streams = {stream_name: opener(stream_name)[0] for stream_name in unique_stream_names}
    if all(_has_lineage_metadata(streams[stream_name]) for stream_name in normalized_path[1:]):
        histories = {
            stream_name: collect_stream_metadata_history(
                streams[stream_name],
                samples=samples,
                show_progress=show_progress,
                timeout_seconds=timeout_seconds,
            )
            for stream_name in normalized_path[1:]
        }

        segments = []
        for source_name, target_name in zip(normalized_path[:-1], normalized_path[1:]):
            segment, _ = _build_metadata_segment(source_name, target_name, histories[target_name])
            segments.append(segment)

        total_segment, total_samples = _build_metadata_total_segment(
            normalized_path[0],
            normalized_path[-1],
            histories[normalized_path[-1]],
        )

        report = LatencyReport(
            source_shm=normalized_path[0],
            target_shm=normalized_path[-1],
            stream_path=tuple(normalized_path),
            inferred_path=False,
            sample_count=int(samples),
            total=total_segment,
            segments=tuple(segments),
        )
        if include_total_samples:
            return report, total_samples
        return report, None

    counts, write_times = collect_stream_event_history(
        streams,
        samples=samples,
        show_progress=show_progress,
        timeout_seconds=timeout_seconds,
    )

    segments = []
    for source_name, target_name in zip(normalized_path[:-1], normalized_path[1:]):
        segment, _ = _build_latency_segment(
            source_name,
            target_name,
            counts[source_name],
            write_times[source_name],
            counts[target_name],
            write_times[target_name],
        )
        segments.append(segment)

    total_segment, total_samples = _build_latency_segment(
        normalized_path[0],
        normalized_path[-1],
        counts[normalized_path[0]],
        write_times[normalized_path[0]],
        counts[normalized_path[-1]],
        write_times[normalized_path[-1]],
    )

    report = LatencyReport(
        source_shm=normalized_path[0],
        target_shm=normalized_path[-1],
        stream_path=tuple(normalized_path),
        inferred_path=False,
        sample_count=int(samples),
        total=total_segment,
        segments=tuple(segments),
    )
    if include_total_samples:
        return report, total_samples
    return report, None


def format_latency_report(report: LatencyReport | Mapping[str, Any]) -> str:
    """Return a readable text summary for a latency report."""

    payload = report.to_dict() if hasattr(report, "to_dict") else dict(report)
    total = payload["total"]
    total_stats = total["statistics"]
    path_label = " -> ".join(payload["stream_path"])

    lines = [
        f"Latency report: {payload['source_shm']} -> {payload['target_shm']}",
        f"Path: {path_label}{' (inferred)' if payload.get('inferred_path') else ''}",
        f"Samples: {payload['sample_count']}",
        "",
        "Total",
        f"  Mean: {_format_seconds(total_stats['mean_seconds'])}",
        f"  Jitter (std): {_format_seconds(total_stats['jitter_seconds'])}",
        f"  Min / Max: {_format_seconds(total_stats['min_seconds'])} / {_format_seconds(total_stats['max_seconds'])}",
        f"  P95 / P99 / P99.9: {_format_seconds(total_stats['p95_seconds'])} / {_format_seconds(total_stats['p99_seconds'])} / {_format_seconds(total_stats['p999_seconds'])}",
        f"  Max speed (from full-loop P99): {_format_seconds_with_rate(total_stats['p99_seconds'])}",
        f"  Count offset: {total.get('count_offset', 0)}",
        f"  Residual count delta range: {total['count_delta_min']:.0f} to {total['count_delta_max']:.0f}",
    ]
    processing_stats = total.get("processing_statistics")
    if processing_stats is not None:
        lines.append(f"  Processing latency: {_format_seconds(processing_stats['mean_seconds'])}")

    segments = payload.get("segments", [])
    if segments:
        lines.append("")
        lines.append("Breakdown")
        for segment in segments:
            stats = segment["statistics"]
            lines.append(
                "  "
                + f"{segment['source_shm']} -> {segment['target_shm']}: "
                + f"mean={_format_seconds(stats['mean_seconds'])}, "
                + f"jitter={_format_seconds(stats['jitter_seconds'])}, "
                + f"p99={_format_seconds(stats['p99_seconds'])}"
            )
            processing_stats = segment.get("processing_statistics")
            if processing_stats is not None:
                lines.append(
                    "    "
                    + f"processing={_format_seconds(processing_stats['mean_seconds'])}"
                )

    return "\n".join(lines)