import numpy as np
import pytest

from pyrtc import latency
from pyrtc.scripts import measure_latency
from testsupport import publishing_chain


def test_compute_latency_applies_frame_shift():
    source_times = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    target_times = np.array([0.2, 1.2, 2.2, 3.2], dtype=np.float64)

    latency_values, shift = latency.compute_latency_seconds(source_times, target_times)

    assert shift == 1
    assert np.allclose(latency_values, np.array([0.2, 0.2, 0.2], dtype=np.float64))


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
    def _fake_open(name):
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
        # No producer stamped frame ids, so segments fall back to count alignment.
        frame_ids = {name: np.zeros(4, dtype=np.uint64) for name in counts}
        return counts, write_times, frame_ids

    monkeypatch.setattr(latency, "collect_stream_event_history", _fake_collect)

    report, total_samples = latency.measure_stream_path_latency(
        ["wfs", "signal", "wfc"],
        samples=4,
        shm_opener=_fake_open,
        include_total_samples=True,
        wait_for_live=False,
    )

    assert report.stream_path == ("wfs", "signal", "wfc")
    assert np.allclose(total_samples, np.array([0.004, 0.004, 0.004, 0.004], dtype=np.float64))
    assert report.total.statistics.mean_seconds == pytest.approx(0.004)
    assert report.segments[0].statistics.mean_seconds == pytest.approx(0.001)
    assert report.segments[1].statistics.mean_seconds == pytest.approx(0.003)
    assert report.total.alignment == "count"


def test_event_history_records_frame_ids_for_exact_alignment():
    with publishing_chain(["src", "dst"], step_seconds=2e-3) as opener:
        streams = {name: opener(name) for name in ("src", "dst")}
        try:
            counts, write_times, frame_ids = latency.collect_stream_event_history(
                streams, samples=6, timeout_seconds=10.0
            )
        finally:
            for stream in streams.values():
                stream.close()

    assert np.all(np.diff(counts["src"]) > 0)
    assert frame_ids["src"].all() and frame_ids["dst"].all()
    matched = latency.compute_frame_matched_latency_seconds(
        frame_ids["src"], write_times["src"], frame_ids["dst"], write_times["dst"]
    )
    assert matched is not None and matched.size >= 4
    # dst is written one step after src for every frame.
    assert np.all(matched > 0)
    assert np.all(matched < 0.1)


def test_latency_waits_for_a_late_pipeline_and_aligns_by_frame_id(caplog):
    with publishing_chain(
        ["src", "mid", "dst"], step_seconds=1e-3, downstream_start_seconds=0.3
    ) as opener:
        report, _ = latency.measure_stream_path_latency(
            ["src", "mid", "dst"], samples=16, shm_opener=opener, timeout_seconds=10.0
        )

    assert report.total.alignment == "frame_id"
    assert all(segment.alignment == "frame_id" for segment in report.segments)
    assert report.total.matched_samples >= 8
    assert "falling back to count alignment" not in caplog.text
    text = latency.format_latency_report(report)
    assert "exact (frame id" in text and "matched frames" in text


def test_latency_warns_when_windows_share_no_frame_ids(caplog):
    # Without the liveness wait, the source window ends before downstream starts.
    with publishing_chain(
        ["src", "dst"], step_seconds=1e-3, downstream_start_seconds=0.3
    ) as opener:
        report, _ = latency.measure_stream_path_latency(
            ["src", "dst"],
            samples=8,
            shm_opener=opener,
            timeout_seconds=10.0,
            wait_for_live=False,
        )

    assert report.total.alignment == "count"
    assert "falling back to count alignment" in caplog.text
    assert "Alignment: count (heuristic" in latency.format_latency_report(report)


def test_wait_for_path_live_times_out_when_downstream_never_publishes():
    with publishing_chain(["src", "dst"], downstream_start_seconds=60.0) as opener:
        streams = {name: opener(name) for name in ("src", "dst")}
        try:
            with pytest.raises(TimeoutError, match="'dst'"):
                latency.wait_for_path_live(streams, ["src", "dst"], timeout_seconds=0.2)
        finally:
            for stream in streams.values():
                stream.close()


def test_wait_for_path_live_returns_without_frame_ids():
    with publishing_chain(["src", "dst"], stamp_frame_ids=False) as opener:
        streams = {name: opener(name) for name in ("src", "dst")}
        try:
            assert latency.wait_for_path_live(streams, ["src", "dst"], timeout_seconds=5.0) < 5.0
        finally:
            for stream in streams.values():
                stream.close()


def test_frame_matched_latency_requires_stamped_frames():
    zeros = np.zeros(3, dtype=np.uint64)
    times = np.array([1.0, 2.0, 3.0])
    assert latency.compute_frame_matched_latency_seconds(zeros, times, zeros, times) is None


def test_frame_matched_latency_pairs_by_frame_not_position():
    source_ids = np.array([5, 6, 7], dtype=np.uint64)
    source_times = np.array([1.0, 2.0, 3.0])
    # The target missed frame 5 and reports frame 6 twice.
    target_ids = np.array([6, 6, 7], dtype=np.uint64)
    target_times = np.array([2.25, 2.5, 3.25])

    matched = latency.compute_frame_matched_latency_seconds(
        source_ids, source_times, target_ids, target_times
    )

    assert np.allclose(matched, [0.25, 0.25])


def test_main_no_show(monkeypatch, tmp_path):
    with publishing_chain(["wfs_raw", "wfc_2d"]) as opener:
        monkeypatch.setattr(latency, "open_stream", opener)
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


def test_observer_closes_while_a_writer_thread_holds_the_lock():
    # pyshmem >= 1.3.5: the lock is shared per name, but closing a handle the
    # writer is not using must not fail while the writer is mid-write.
    import threading

    from testsupport import private_stream

    writer = private_stream("lockwrt", (2,), "float32")
    observer = latency.open_stream(writer.name)
    holding, release = threading.Event(), threading.Event()

    def _write_under_lock():
        with writer.locked():
            holding.set()
            release.wait(5.0)

    worker = threading.Thread(target=_write_under_lock, daemon=True)
    worker.start()
    assert holding.wait(5.0)
    try:
        observer.close()
    finally:
        release.set()
        worker.join(5.0)
