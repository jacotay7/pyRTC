import numpy as np

from pyRTC.scripts import measure_latency


def test_compute_latency_applies_frame_shift():
    source_times = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    target_times = np.array([0.2, 1.2, 2.2, 3.2], dtype=np.float64)

    latency, shift = measure_latency.compute_latency_seconds(source_times, target_times)

    assert shift == 1
    assert np.allclose(latency, np.array([0.2, 0.2, 0.2], dtype=np.float64))


def test_collect_timestamps_reads_metadata():
    class FakeShm:
        def __init__(self, metadata_values):
            self._values = metadata_values
            self._index = 0
            self.metadata = self._values[self._index]

        def hold(self):
            self.metadata = self._values[self._index]
            self._index += 1

    source = FakeShm([
        np.array([1, 0.10]),
        np.array([2, 0.20]),
        np.array([3, 0.30]),
    ])
    target = FakeShm([
        np.array([5, 0.15]),
        np.array([6, 0.25]),
        np.array([7, 0.35]),
    ])

    source_counts, source_times, target_counts, target_times = measure_latency.collect_timestamps(
        source, target, samples=3, show_progress=False
    )

    assert np.array_equal(source_counts, np.array([1, 2, 3], dtype=np.float64))
    assert np.array_equal(target_counts, np.array([5, 6, 7], dtype=np.float64))
    assert np.allclose(source_times, np.array([0.10, 0.20, 0.30], dtype=np.float64))
    assert np.allclose(target_times, np.array([0.15, 0.25, 0.35], dtype=np.float64))


def test_main_no_show(monkeypatch, tmp_path):
    class FakeShm:
        def __init__(self):
            self._count = 0
            self.metadata = np.array([0, 0.0], dtype=np.float64)

        def hold(self):
            self._count += 1
            self.metadata = np.array([self._count, self._count * 1e-3], dtype=np.float64)

    monkeypatch.setattr(measure_latency, "initExistingShm", lambda name: (FakeShm(), None, None))
    monkeypatch.setattr(measure_latency.plt, "show", lambda: None)
    monkeypatch.setattr(measure_latency, "plot_latency_histogram", lambda sys_latency, args: None)

    def _fake_savefig(path):
        with open(path, "wb") as f:
            f.write(b"%PDF-FAKE")

    monkeypatch.setattr(measure_latency.plt, "savefig", _fake_savefig)

    out = tmp_path / "lat.pdf"
    code = measure_latency.main(
        [
            "wfsRaw",
            "wfc2D",
            "--samples",
            "20",
            "--no-progress",
            "--no-show",
            "--output",
            str(out),
        ]
    )

    assert code == 0
    assert out.exists()
