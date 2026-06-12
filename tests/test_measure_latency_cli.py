import numpy as np
import pytest

from pyrtc import latency
from pyrtc.scripts import measure_latency


def test_compute_latency_applies_frame_shift():
    source_times = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    target_times = np.array([0.2, 1.2, 2.2, 3.2], dtype=np.float64)

    latency, shift = measure_latency.compute_latency_seconds(source_times, target_times)

    assert shift == 1
    assert np.allclose(latency, np.array([0.2, 0.2, 0.2], dtype=np.float64))


def test_count_aligned_latency_removes_startup_count_offset():
    source_counts = np.array([10, 11, 12, 13], dtype=np.float64)
    source_times = np.array([1.000, 1.010, 1.020, 1.030], dtype=np.float64)
    target_counts = np.array([110, 111, 112, 113], dtype=np.float64)
    target_times = np.array([1.002, 1.012, 1.022, 1.032], dtype=np.float64)

    latency_values, count_offset, residual = latency.compute_count_aligned_latency_seconds(
        source_counts,
        source_times,
        target_counts,
        target_times,
    )

    assert count_offset == 100
    assert np.allclose(latency_values, np.array([0.002, 0.002, 0.002, 0.002], dtype=np.float64))
    assert np.array_equal(residual, np.array([0, 0, 0, 0], dtype=np.int64))


def test_measure_stream_path_latency_uses_shared_event_history(monkeypatch):
    def _fake_open(name, gpu_device=None):
        return object()

    def _fake_collect(streams, samples, **kwargs):
        assert set(streams) == {"wfs", "signal", "wfc"}
        counts = {
            "wfs": np.array([10, 11, 12, 13], dtype=np.float64),
            "signal": np.array([20, 21, 22, 23], dtype=np.float64),
            "wfc": np.array([30, 31, 32, 33], dtype=np.float64),
        }
        write_times = {
            "wfs": np.array([1.000, 1.005, 1.010, 1.015], dtype=np.float64),
            "signal": np.array([1.001, 1.006, 1.011, 1.016], dtype=np.float64),
            "wfc": np.array([1.004, 1.009, 1.014, 1.019], dtype=np.float64),
        }
        return counts, write_times

    monkeypatch.setattr(latency, "collect_stream_event_history", _fake_collect)

    report, total_samples = latency.measure_stream_path_latency(
        ["wfs", "signal", "wfc"],
        samples=4,
        shm_opener=_fake_open,
        include_total_samples=True,
    )

    assert report.stream_path == ("wfs", "signal", "wfc")
    assert np.allclose(total_samples, np.array([0.004, 0.004, 0.004, 0.004], dtype=np.float64))
    assert report.total.statistics.mean_seconds == pytest.approx(0.004)
    assert report.segments[0].statistics.mean_seconds == pytest.approx(0.001)
    assert report.segments[1].statistics.mean_seconds == pytest.approx(0.003)


def test_collect_timestamps_reads_metadata(monkeypatch):
    class FakeShm:
        def __init__(self, events):
            self._events = list(events)
            self._index = -1

        def advance(self):
            self._index = min(self._index + 1, len(self._events) - 1)
            return self._events[self._index][0]

        @property
        def count(self):
            return self._events[max(self._index, 0)][0]

        @property
        def write_time(self):
            return self._events[max(self._index, 0)][1]

    import pyrtc.latency as latency_helpers

    monkeypatch.setattr(
        latency_helpers,
        "_wait_for_new_write",
        lambda stream, poll_interval_seconds=1e-5: stream.advance(),
    )

    source = FakeShm([(1, 0.10), (2, 0.20), (3, 0.30)])
    target = FakeShm([(5, 0.15), (6, 0.25), (7, 0.35)])

    counts, write_times = measure_latency.collect_timestamps(
        {"source": source, "target": target}, samples=3, show_progress=False
    )

    assert np.array_equal(counts["source"], np.array([1, 2, 3], dtype=np.float64))
    assert np.array_equal(counts["target"], np.array([5, 6, 7], dtype=np.float64))
    assert np.allclose(write_times["source"], np.array([0.10, 0.20, 0.30], dtype=np.float64))
    assert np.allclose(write_times["target"], np.array([0.15, 0.25, 0.35], dtype=np.float64))


def test_main_no_show(monkeypatch, tmp_path):
    class FakeShm:
        def __init__(self):
            self._count = 0

        @property
        def count(self):
            self._count += 1
            return self._count

        @property
        def write_time(self):
            return self._count * 1e-3

    monkeypatch.setattr(latency, "open_stream", lambda name, gpu_device=None: FakeShm())
    monkeypatch.setattr(measure_latency, "plot_latency_histogram", lambda *args, **kwargs: None)

    def _fake_savefig(path):
        with open(path, "wb") as f:
            f.write(b"%PDF-FAKE")

    from matplotlib import pyplot as plt

    monkeypatch.setattr(plt, "savefig", _fake_savefig)

    out = tmp_path / "lat.pdf"
    code = measure_latency.main(
        [
            "wfs_raw",
            "wfc_2d",
            "--samples",
            "20",
            "--no-progress",
            "--output",
            str(out),
        ]
    )

    assert code == 0
    assert out.exists()


def test_main_config_json_output(monkeypatch):
    monkeypatch.setattr(
        measure_latency.RTCManager,
        "from_config_file",
        classmethod(lambda cls, path: _FakeManager()),
    )
    emitted = {}

    def _capture_emit(report_payload, output_format):
        emitted["payload"] = report_payload
        emitted["format"] = output_format

    monkeypatch.setattr(measure_latency, "_emit_report", _capture_emit)

    code = measure_latency.main(
        [
            "--config",
            "examples/synthetic_shwfs/config.yaml",
            "--format",
            "json",
            "--samples",
            "16",
        ]
    )

    assert code == 0
    assert emitted["format"] == "json"
    assert emitted["payload"]["stream_path"] == ["wfs", "signal", "wfc"]
    assert emitted["payload"]["inferred_path"] is True


def test_format_latency_report_includes_max_speed_in_khz():
    report_text = latency.format_latency_report(
        {
            "source_shm": "wfs",
            "target_shm": "wfc",
            "stream_path": ["wfs", "signal", "wfc"],
            "inferred_path": True,
            "sample_count": 16,
            "total": {
                "source_shm": "wfs",
                "target_shm": "wfc",
                "frame_shift": 0,
                "count_offset": 0,
                "count_delta_min": 0.0,
                "count_delta_max": 0.0,
                "statistics": {
                    "sample_count": 16,
                    "mean_seconds": 1e-3,
                    "std_seconds": 1e-4,
                    "jitter_seconds": 1e-4,
                    "min_seconds": 9e-4,
                    "max_seconds": 1.2e-3,
                    "p50_seconds": 1e-3,
                    "p95_seconds": 1.1e-3,
                    "p99_seconds": 1.15e-3,
                    "p999_seconds": 1.19e-3,
                },
            },
            "segments": [
                {
                    "source_shm": "wfs",
                    "target_shm": "signal",
                    "frame_shift": 0,
                    "count_offset": 0,
                    "count_delta_min": 0.0,
                    "count_delta_max": 0.0,
                    "statistics": {
                        "sample_count": 16,
                        "mean_seconds": 5e-4,
                        "std_seconds": 2e-5,
                        "jitter_seconds": 2e-5,
                        "min_seconds": 4.5e-4,
                        "max_seconds": 5.2e-4,
                        "p50_seconds": 5e-4,
                        "p95_seconds": 5.1e-4,
                        "p99_seconds": 5.15e-4,
                        "p999_seconds": 5.19e-4,
                    },
                },
            ],
        }
    )

    assert "Mean: 1.000 ms" in report_text
    assert "Max speed (from full-loop P99): 1.150 ms (max speed 0.870 kHz)" in report_text
    assert "mean=500.000 us" in report_text
    assert "p99=515.000 us" in report_text


class _FakeManager:
    def latency(self, **kwargs):
        return {
            "source_shm": "wfs",
            "target_shm": "wfc",
            "stream_path": ["wfs", "signal", "wfc"],
            "inferred_path": True,
            "sample_count": kwargs["samples"],
            "total": {
                "source_shm": "wfs",
                "target_shm": "wfc",
                "frame_shift": 0,
                "count_offset": 0,
                "count_delta_min": 0.0,
                "count_delta_max": 0.0,
                "statistics": {
                    "sample_count": kwargs["samples"],
                    "mean_seconds": 1e-3,
                    "std_seconds": 1e-4,
                    "jitter_seconds": 1e-4,
                    "min_seconds": 9e-4,
                    "max_seconds": 1.2e-3,
                    "p50_seconds": 1e-3,
                    "p95_seconds": 1.1e-3,
                    "p99_seconds": 1.15e-3,
                    "p999_seconds": 1.19e-3,
                },
            },
            "segments": [],
        }
