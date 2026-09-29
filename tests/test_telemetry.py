import numpy as np
import importlib
import time
from pathlib import Path

import pyshmem
import pytest

from testsupport import StaticStream, private_stream, publishing_chain

tele_mod = importlib.import_module("pyrtc.telemetry")


def test_telemetry_save_and_read(monkeypatch, tmp_path):
    monkeypatch.setattr(
        tele_mod,
        "open_stream",
        lambda name, **kw: StaticStream(np.array([1.0, 2.0], dtype=np.float32), 123.0, frame_id=9),
    )

    t = tele_mod.Telemetry({"data_dir": str(tmp_path), "functions": []})
    session_path = t.save("signal", 3, unique_str="u")
    arr = t.read()
    assert arr.shape == (3, 2)
    assert Path(session_path).is_dir()

    data = t.read_last_save()
    assert data["signal"]["frames"].shape == (3, 2)
    assert data["signal"]["timestamps"].shape == (3,)
    assert np.all(data["signal"]["timestamps"] == 123.0)
    assert np.all(data["signal"]["frame_ids"] == 9)
    assert data["signal"]["metadata"]["missed_frames"] == 0
    assert data["signal"]["metadata"]["dtype"] == "float32"
    assert t.list_sessions() == [str(Path(session_path).resolve())]

    reopened = t.read(session_path)
    assert reopened["signal"]["frames"].shape == (3, 2)

    other_file = tmp_path / "raw.bin"
    np.array([1, 2, 3], dtype=np.int16).tofile(other_file)
    out = t.read(filename=str(other_file), dtype=np.int16)
    assert out.dtype == np.int16


def test_telemetry_save_session_supports_multi_stream_grouped_capture(monkeypatch, tmp_path):
    streams = {
        "signal": StaticStream(np.array([1.0, 2.0], dtype=np.float32), 456.0),
        "wfc": StaticStream(np.array([[3, 4], [5, 6]], dtype=np.int16), 456.0),
    }
    monkeypatch.setattr(tele_mod, "open_stream", lambda name, **kw: streams[name])

    telemetry = tele_mod.Telemetry(
        {"data_dir": str(tmp_path), "functions": [], "streams": ["signal", "wfc"]}
    )
    session_path = telemetry.save(
        ["signal", "wfc"],
        {"signal": 2, "wfc": 1},
        unique_str="group",
        semantic_tags={"signal": ["signal"], "wfc": ["wfc", "control"]},
        sampling={"signal": {"mode": "every_frame"}},
        config={"metadata": {"name": "synthetic"}},
        config_path=tmp_path / "config.yaml",
        metadata={"operator": "pytest"},
    )

    loaded = tele_mod.load_telemetry_session(session_path)
    manifest = loaded["_session"]

    assert manifest["schema_version"] == tele_mod.TELEMETRY_SESSION_SCHEMA_VERSION
    assert manifest["metadata"]["operator"] == "pytest"
    assert manifest["config_path"] == str((tmp_path / "config.yaml").resolve())
    assert len(manifest["streams"]) == 2
    assert loaded["signal"]["frames"].shape == (2, 2)
    assert loaded["wfc"]["frames"].shape == (1, 2, 2)
    assert manifest["streams"][1]["semantic_tags"] == ["wfc", "control"]
    assert np.all(loaded["signal"]["timestamps"] == 456.0)


def test_telemetry_save_configured_streams_uses_component_config(monkeypatch, tmp_path):
    monkeypatch.setattr(
        tele_mod,
        "open_stream",
        lambda name, **kw: StaticStream(np.array([7, 8, 9], dtype=np.float32), 789.0),
    )
    telemetry = tele_mod.Telemetry(
        {"data_dir": str(tmp_path), "functions": [], "streams": ["signal"]}
    )

    session_path = telemetry.save_configured_streams(2, unique_str="cfg")
    loaded = telemetry.read_last_save()

    assert session_path == telemetry.most_recent_save
    assert loaded["signal"]["frames"].shape == (2, 3)


def test_telemetry_error_paths(monkeypatch, tmp_path):
    telemetry_module = importlib.import_module("pyrtc.telemetry")

    def bad_component_init(self, conf):
        raise RuntimeError("telemetry init failed")

    with monkeypatch.context() as mp:
        mp.setattr(telemetry_module.Component, "__init__", bad_component_init)
        with pytest.raises(RuntimeError, match="telemetry init failed"):
            telemetry_module.Telemetry({"functions": []})

    t = telemetry_module.Telemetry({"data_dir": str(tmp_path), "functions": []})

    monkeypatch.setattr(
        telemetry_module,
        "open_stream",
        lambda name, **kw: (_ for _ in ()).throw(RuntimeError("missing shm")),
    )
    with pytest.raises(RuntimeError, match="missing shm"):
        t.save("signal", 1)

    unmanaged_file = tmp_path / "unmanaged.bin"
    np.array([1, 2, 3], dtype=np.int16).tofile(unmanaged_file)
    with pytest.raises(ValueError, match="please provide a dtype"):
        t.read(filename=str(unmanaged_file))

    broken_manifest = tmp_path / "broken_manifest.json"
    broken_manifest.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(ValueError, match="Failed to read telemetry manifest"):
        telemetry_module.load_telemetry_manifest(broken_manifest)

    monkeypatch.setattr(
        telemetry_module,
        "open_stream",
        lambda name, **kw: StaticStream(np.array([1.0, 2.0], dtype=np.float32), 10.0),
    )
    session_path = t.save("signal", 1)
    manifest = telemetry_module.load_telemetry_manifest(session_path)
    capture_path = Path(session_path) / manifest["streams"][0]["frames_file"]
    capture_path.unlink()

    with pytest.raises(FileNotFoundError, match="Telemetry capture file not found"):
        telemetry_module.load_telemetry_session(session_path)

    empty_telemetry = telemetry_module.Telemetry(
        {"data_dir": str(tmp_path / "empty"), "functions": []}
    )
    with pytest.raises(ValueError, match="no configured streams"):
        empty_telemetry.save_configured_streams(1)

    with pytest.raises(ValueError, match="No telemetry save is available"):
        empty_telemetry.read_last_save()


def test_multi_stream_capture_covers_one_time_window(monkeypatch, tmp_path):
    with publishing_chain(["wfs", "signal", "wfc"], step_seconds=2e-3) as opener:
        monkeypatch.setattr(tele_mod, "open_stream", opener)
        telemetry = tele_mod.Telemetry({"data_dir": str(tmp_path), "functions": []})
        session_path = telemetry.save(["wfs", "signal", "wfc"], 15)

    loaded = tele_mod.load_telemetry_session(session_path)
    windows = [loaded[name]["timestamps"] for name in ("wfs", "signal", "wfc")]
    # Captured concurrently: every stream's window overlaps every other's.
    assert max(t.min() for t in windows) < min(t.max() for t in windows)
    # The chain stamps one frame id per iteration on all three streams, so
    # concurrent captures share most of their frame ids.
    ids = [set(loaded[name]["frame_ids"].tolist()) for name in ("wfs", "signal", "wfc")]
    assert len(ids[0] & ids[1] & ids[2]) >= 10


# ----------------------------------------------------------------------
# Continuous recording (ring buffer, #61)
# ----------------------------------------------------------------------


def _wait_until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached before timeout")
        time.sleep(1e-3)


def _private_opener(monkeypatch, streams):
    """Route telemetry's open_stream to read-only handles on private streams."""
    monkeypatch.setattr(
        tele_mod,
        "open_stream",
        lambda name, **kw: pyshmem.open(streams[name].name, readonly=True),
    )


def _recorded(telemetry, name):
    return telemetry.ring_buffer_status()[name]["recorded"]


def test_ring_buffer_wraps_and_keeps_newest_frames(monkeypatch, tmp_path):
    stream = private_stream("rb_wrap", (3,), np.float32)
    _private_opener(monkeypatch, {"sig": stream})
    telemetry = tele_mod.Telemetry({"data_dir": str(tmp_path), "functions": []})
    try:
        assert telemetry.start_ring_buffer("sig", frames=10) == {"sig": 10}
        for frame_id in range(1, 26):
            stream.write(np.full(3, frame_id, dtype=np.float32), frame_id=frame_id)
            # Lock-step so every publication is recorded (no misses).
            _wait_until(lambda: _recorded(telemetry, "sig") == frame_id)
        status = telemetry.ring_buffer_status()["sig"]
        assert status["size"] == 10 and status["missed"] == 0
        session_path = telemetry.dump_ring_buffer("wrap")
    finally:
        telemetry.stop_ring_buffer()

    loaded = tele_mod.load_telemetry_session(session_path)
    sig = loaded["sig"]
    assert sig["frames"].shape == (10, 3)
    assert sig["frame_ids"].tolist() == list(range(16, 26))
    assert np.all(sig["frames"][:, 0] == sig["frame_ids"])
    assert np.all(np.diff(sig["counts"].astype(np.int64)) == 1)
    assert np.all(np.diff(sig["timestamps"]) >= 0)
    assert sig["metadata"]["missed_frames"] == 0
    assert sig["metadata"]["capture_label"] == "wrap"
    assert sig["metadata"]["ring_buffer"]["capacity"] == 10
    assert sig["metadata"]["ring_buffer"]["recorded_total"] == 25
    assert loaded["_session"]["streams"][0]["frame_count"] == 10
    assert telemetry.read_last_save()["sig"]["frames"].shape == (10, 3)


def test_ring_buffer_counts_missed_publications(monkeypatch, tmp_path):
    stream = private_stream("rb_miss", (1,), np.float32)
    _private_opener(monkeypatch, {"sig": stream})
    telemetry = tele_mod.Telemetry({"data_dir": str(tmp_path), "functions": []})
    try:
        telemetry.start_ring_buffer("sig", frames=100)
        ring = telemetry._ring_streams[0]
        # While the ring lock is held the reader can take at most one
        # publication; the rest are overwritten in shared memory.
        with ring.lock:
            for frame_id in range(1, 11):
                stream.write(np.full(1, frame_id, dtype=np.float32), frame_id=frame_id)
        _wait_until(
            lambda: (
                telemetry.ring_buffer_status()["sig"]["recorded"]
                + telemetry.ring_buffer_status()["sig"]["missed"]
                == 10
            )
        )
        status = telemetry.ring_buffer_status()["sig"]
        assert status["missed"] >= 8
        session_path = telemetry.dump_ring_buffer()
    finally:
        telemetry.stop_ring_buffer()
    sig = tele_mod.load_telemetry_session(session_path)["sig"]
    assert sig["frame_ids"][-1] == 10
    counts = sig["counts"].astype(np.int64)
    assert sig["metadata"]["missed_frames"] == counts[-1] - counts[0] + 1 - len(counts)


def test_ring_buffer_seconds_window_orders_dump_by_time(monkeypatch, tmp_path):
    with publishing_chain(["rb_a"], step_seconds=2e-3) as opener:
        monkeypatch.setattr(tele_mod, "open_stream", opener)
        telemetry = tele_mod.Telemetry({"data_dir": str(tmp_path), "functions": []})
        try:
            # A 1 s window: sleep granularity on some CI hosts (macOS) makes the
            # producer publish far slower than the requested 2 ms period.
            telemetry.start_ring_buffer("rb_a", seconds=1.0, frames=5000)
            time.sleep(1.5)
            before = time.time()
            session_path = telemetry.dump_ring_buffer()
        finally:
            telemetry.stop_ring_buffer()

    stamps = tele_mod.load_telemetry_session(session_path)["rb_a"]["timestamps"]
    assert len(stamps) > 10
    assert np.all(np.diff(stamps) >= 0)
    assert stamps.min() >= before - 1.0 - 0.05
    assert stamps.max() - stamps.min() <= 1.0


def test_ring_buffer_estimates_capacity_from_rate(monkeypatch, tmp_path):
    with publishing_chain(["rb_rate"], step_seconds=2e-3) as opener:
        monkeypatch.setattr(tele_mod, "open_stream", opener)
        telemetry = tele_mod.Telemetry({"data_dir": str(tmp_path), "functions": []})
        try:
            capacities = telemetry.start_ring_buffer("rb_rate", seconds=0.5, probe_seconds=0.2)
        finally:
            telemetry.stop_ring_buffer()
    # ~500 Hz at most; sleep jitter only slows the producer down.
    assert 10 <= capacities["rb_rate"] <= int(0.5 * 500 * tele_mod.RING_BUFFER_RATE_HEADROOM) + 1

    silent = private_stream("rb_quiet", (1,), np.float32)
    _private_opener(monkeypatch, {"quiet": silent})
    with pytest.raises(ValueError, match="pass frames="):
        telemetry.start_ring_buffer("quiet", seconds=1.0, probe_seconds=0.05)
    assert not telemetry.ring_buffer_running


def test_ring_buffer_multi_stream_dump_covers_one_time_window(monkeypatch, tmp_path):
    names = ["rb_wfs", "rb_sig", "rb_wfc"]
    with publishing_chain(names, step_seconds=1e-3) as opener:
        monkeypatch.setattr(tele_mod, "open_stream", opener)
        telemetry = tele_mod.Telemetry({"data_dir": str(tmp_path), "functions": []})
        try:
            telemetry.start_ring_buffer(names, frames=40)
            _wait_until(
                lambda: all(s["size"] == 40 for s in telemetry.ring_buffer_status().values())
            )
            session_path = telemetry.dump_ring_buffer(
                semantic_tags={"rb_wfs": ["wfs"], "rb_wfc": ["wfc", "control"]}
            )
        finally:
            telemetry.stop_ring_buffer()

    loaded = tele_mod.load_telemetry_session(session_path)
    windows = [loaded[name]["timestamps"] for name in names]
    assert max(t.min() for t in windows) < min(t.max() for t in windows)
    ids = [set(loaded[name]["frame_ids"].tolist()) for name in names]
    assert len(ids[0] & ids[1] & ids[2]) >= 20
    assert loaded["_session"]["streams"][2]["semantic_tags"] == ["wfc", "control"]


def test_ring_buffer_dump_while_recording_is_consistent(monkeypatch, tmp_path):
    names = ["rb_x", "rb_y"]
    with publishing_chain(names, step_seconds=5e-4) as opener:
        monkeypatch.setattr(tele_mod, "open_stream", opener)
        telemetry = tele_mod.Telemetry({"data_dir": str(tmp_path), "functions": []})
        try:
            telemetry.start_ring_buffer(names, frames=64)
            paths = []
            for _ in range(5):
                start = _recorded(telemetry, "rb_x")
                _wait_until(lambda: _recorded(telemetry, "rb_x") >= start + 20)
                paths.append(telemetry.dump_ring_buffer())
            # Recording keeps going after dumps.
            after = _recorded(telemetry, "rb_x")
            _wait_until(lambda: _recorded(telemetry, "rb_x") > after)
            rings = list(telemetry._ring_streams)
        finally:
            telemetry.stop_ring_buffer()

    assert len(set(paths)) == 5
    assert not telemetry.ring_buffer_running
    assert all(not ring.thread.is_alive() for ring in rings)
    previous_last = -1
    for path in paths:
        loaded = tele_mod.load_telemetry_session(path)
        for name in names:
            stream = loaded[name]
            # The chain writes value == frame_id: payloads match metadata.
            assert np.all(stream["frames"][:, 0] == stream["frame_ids"])
            assert np.all(np.diff(stream["counts"].astype(np.int64)) > 0)
            assert np.all(np.diff(stream["timestamps"]) >= 0)
        last = int(loaded["rb_x"]["frame_ids"][-1])
        assert last > previous_last
        previous_last = last


def test_ring_buffer_config_driven_start_and_stop(monkeypatch, tmp_path):
    stream = private_stream("rb_cfg", (2,), np.int16)
    _private_opener(monkeypatch, {"sig": stream})
    telemetry = tele_mod.Telemetry(
        {
            "data_dir": str(tmp_path),
            "functions": [],
            "streams": ["sig"],
            "ring_buffer": {"frames": 5},
        }
    )
    assert not telemetry.ring_buffer_running
    telemetry.start()
    try:
        assert telemetry.ring_buffer_running
        with pytest.raises(RuntimeError, match="already running"):
            telemetry.start_ring_buffer("sig", frames=5)
        stream.write(np.array([4, 5], dtype=np.int16), frame_id=1)
        _wait_until(lambda: _recorded(telemetry, "sig") == 1)
        loaded = tele_mod.load_telemetry_session(telemetry.dump_ring_buffer())
        assert loaded["sig"]["frames"].tolist() == [[4, 5]]
    finally:
        telemetry.stop()
    assert not telemetry.ring_buffer_running
    assert telemetry.ring_buffer_status() == {}
    with pytest.raises(ValueError, match="not running"):
        telemetry.dump_ring_buffer()
    telemetry.stop_ring_buffer()  # idempotent


def test_ring_buffer_argument_and_config_validation(tmp_path):
    from pyrtc.config_schema import ConfigValidationError, _validate_telemetry_config

    telemetry = tele_mod.Telemetry({"data_dir": str(tmp_path), "functions": []})
    with pytest.raises(ValueError, match="no streams"):
        telemetry.start_ring_buffer(frames=5)
    with pytest.raises(ValueError, match="seconds=, frames="):
        telemetry.start_ring_buffer("sig")

    _validate_telemetry_config({"ring_buffer": {"streams": ["wfs"], "seconds": 5}})
    for bad in (
        {"ring_buffer": []},
        {"ring_buffer": {"streams": ["wfs"]}},
        {"ring_buffer": {"frames": 0}},
        {"ring_buffer": {"seconds": -1}},
        {"ring_buffer": {"frames": 5, "bogus": 1}},
        {"ring_buffer": {"frames": 5, "autostart": "yes"}},
    ):
        with pytest.raises(ConfigValidationError):
            _validate_telemetry_config(bad)
