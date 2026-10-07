import contextlib
import sys
import threading
import time
import types
import uuid

import pytest


def _np():
    import numpy as np

    return np


class _FakeHDU:
    def __init__(self, data):
        self.data = data


class _FakeHDUList(list):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakePrimaryHDU:
    def __init__(self, data):
        self.data = _np().asarray(data)

    def writeto(self, filename):
        with open(filename, "wb") as f:
            _np().save(f, self.data)


def _fake_fits_open(filename):
    with open(filename, "rb") as f:
        data = _np().load(f, allow_pickle=False)
    return _FakeHDUList([_FakeHDU(data)])


try:
    import astropy.io.fits  # noqa: F401
except Exception:
    fake_fits = types.SimpleNamespace(PrimaryHDU=_FakePrimaryHDU, open=_fake_fits_open)
    fake_io = types.SimpleNamespace(fits=fake_fits)
    fake_astropy = types.SimpleNamespace(io=fake_io)

    sys.modules.setdefault("astropy", fake_astropy)
    sys.modules.setdefault("astropy.io", fake_io)
    sys.modules.setdefault("astropy.io.fits", fake_fits)


_PRIVATE_STREAMS = []


def private_stream(name, shape, dtype, gpu_device=None):
    """Drop-in ``create_stream`` replacement backed by a real pyshmem stream.

    The stream gets a unique name so tests never collide with a running RTC
    or with each other; :func:`unlink_private_streams` destroys it after the
    test. ``gpu_device`` is accepted for signature parity and ignored.
    """
    import pyshmem

    unique = f"{str(name)[:8]}_{uuid.uuid4().hex[:8]}"
    stream = pyshmem.create(unique, shape=tuple(int(axis) for axis in shape), dtype=dtype)
    _PRIVATE_STREAMS.append(stream)
    return stream


def bare_component(cls, conf=None, **streams):
    """Build ``cls`` without its ``__init__`` but with real stream state.

    Runs :meth:`pyrtc.component.Component._init_runtime_state`, so the
    stream helpers (``read_stream``, ``write_stream``, aliases, ``close``)
    behave exactly as on a fully constructed component, while the test sets
    only the attributes the method under test needs. No worker threads are
    started. Keyword arguments register streams: ``inputs={"signal": shm}``
    and ``outputs={"wfc": shm}``.
    """
    from pyrtc.component import Component

    unknown = set(streams) - {"inputs", "outputs"}
    if unknown:
        raise TypeError(f"unexpected keyword arguments: {sorted(unknown)}")
    obj = cls.__new__(cls)
    Component._init_runtime_state(obj, dict(conf or {}))
    for name, shm in (streams.get("inputs") or {}).items():
        obj.register_input_stream(name, shm)
    for name, shm in (streams.get("outputs") or {}).items():
        obj.register_output_stream(name, shm)
    return obj


def private_synthetic_config(workdir, *, prefix=None, include_psf=True):
    """Write the synthetic SHWFS example config with private stream names.

    Every stream is renamed to ``<prefix>_<name>`` so a test can run the full
    system next to other tests (or a real RTC) on the same host. Class files
    become absolute paths and the loop gets a generated interaction matrix.
    Returns ``(config_path, names)`` where ``names`` maps each canonical
    stream name to its private one.
    """
    import copy
    from pathlib import Path

    import yaml

    from pyrtc.hardware.synthetic_systems import (
        _default_wfc_layout,
        build_synthetic_shwfs_response_matrix,
    )

    example_dir = Path(__file__).resolve().parents[1] / "examples" / "synthetic_shwfs"
    config = copy.deepcopy(yaml.safe_load((example_dir / "config.yaml").read_text("utf-8")))
    prefix = prefix or f"t{uuid.uuid4().hex[:8]}"
    workdir = Path(workdir)
    if not include_psf:
        config.pop("psf", None)
        for key in ("component_classes", "component_files"):
            config.get("manager", {}).get(key, {}).pop("psf", None)
    config.get("manager", {}).pop("graph_layout", None)

    names = {}

    def _private(stream):
        return names.setdefault(stream, f"{prefix}_{stream}")

    def _absolute(path_value):
        path = Path(path_value)
        return str(path if path.is_absolute() else (example_dir / path).resolve())

    for section in config.values():
        if not isinstance(section, dict) or "class_name" not in section:
            continue
        for direction in ("input_streams", "output_streams"):
            aliases = section.get(direction) or {}
            section[direction] = {key: _private(value) for key, value in aliases.items()}
        if section.get("class_file"):
            section["class_file"] = _absolute(section["class_file"])
    # The synthetic WFS and science camera read these streams by their
    # canonical names unless an input alias says otherwise.
    config["wfs"]["input_streams"] = {"wfc": _private("wfc")}
    if "psf" in config:
        config["psf"]["input_streams"] = {"signal": _private("signal")}
    manager_conf = config.setdefault("manager", {})
    manager_conf["component_files"] = {
        key: _absolute(value) for key, value in manager_conf.get("component_files", {}).items()
    }

    side = min(int(config["wfs"]["width"]), int(config["wfs"]["height"]))
    num_regions = side // int(config["slopes"]["sub_ap_spacing"])
    layout = _default_wfc_layout(int(config["wfc"]["num_actuators"]))
    response = build_synthetic_shwfs_response_matrix(
        num_regions, int(config["wfc"]["num_modes"]), layout
    )
    im_path = workdir / f"{prefix}_im.npy"
    from pyrtc.calibration import save_calibration

    save_calibration(im_path, response.astype(_np().float32), "interaction_matrix")
    config["loop"]["im_file"] = str(im_path)

    config_path = workdir / f"{prefix}_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path, names


@contextlib.contextmanager
def publishing_chain(
    names, *, step_seconds=1e-3, stamp_frame_ids=True, downstream_start_seconds=0.0
):
    """Run a background producer writing frames through ``names`` in order.

    Each frame is written to every stream in turn, ``step_seconds`` apart,
    stamped with the same frame id (as pyrtc components do), so stream ``i+1``
    lags stream ``i`` by one step. Yields an opener mapping a logical name to a
    fresh read-only handle, suitable for ``latency.open_stream``.

    With ``downstream_start_seconds``, only the first stream is written for
    that long, like a pipeline whose downstream workers are still compiling.
    """
    import pyshmem

    streams = {name: private_stream(name, (1,), "float32") for name in names}
    stop = threading.Event()

    def _run():
        frame_id = 0
        downstream_from = time.monotonic() + downstream_start_seconds
        while not stop.is_set():
            frame_id += 1
            live = time.monotonic() >= downstream_from
            for index, stream in enumerate(streams.values()):
                if index and not live:
                    continue
                stream.write(
                    _np().full(1, frame_id, dtype=_np().float32),
                    frame_id=frame_id if stamp_frame_ids else None,
                )
                time.sleep(step_seconds)

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()
    try:
        yield lambda name, **_kwargs: pyshmem.open(streams[name].name, readonly=True)
    finally:
        stop.set()
        worker.join()


def prefix_system_streams(config, prefix):
    """Rename every component stream in a normalized system config.

    Gives a whole RTC private stream names so a test can run alongside other
    tests or live systems. Returns the sorted list of new names (for cleanup).
    """
    names = set()
    for conf in config.values():
        if not isinstance(conf, dict):
            continue
        for key in ("input_streams", "output_streams"):
            aliases = conf.get(key)
            if not isinstance(aliases, dict):
                continue
            conf[key] = {logical: f"{prefix}{target}" for logical, target in aliases.items()}
            names.update(conf[key].values())
    return sorted(names)


class StaticStream:
    """Stream stand-in that republishes one frame on every read.

    Telemetry capture blocks for new publications; this double lets tests
    capture any number of frames without a producer thread.
    """

    def __init__(self, data, write_time=1.0, frame_id=0):
        self.data = _np().asarray(data)
        self.shape = self.data.shape
        self.dtype = self.data.dtype
        self.write_time = float(write_time)
        self.frame_id = int(frame_id)
        self.count = 0

    def read(self, **_kwargs):
        return _np().copy(self.data)

    def read_new_publication(self, **_kwargs):
        import pyshmem

        self.count += 1
        return pyshmem.Publication(
            payload=self.read(),
            count=self.count,
            frame_id=self.frame_id,
            write_time=self.write_time,
            missed_publications=0,
        )

    def close(self):
        pass


@pytest.fixture(autouse=True)
def unlink_private_streams():
    yield
    import pyshmem

    while _PRIVATE_STREAMS:
        pyshmem.unlink_quiet(_PRIVATE_STREAMS.pop().name)


@pytest.fixture(autouse=True)
def pyrtc_logs_propagate():
    """Let pytest's ``caplog`` see pyrtc log records in every test.

    ``configure_logging`` (run once, by whichever test first builds a
    component) sets ``propagate = False`` on the ``pyrtc`` logger so CLI
    output isn't duplicated. That silently disabled ``caplog`` for all later
    tests, making log assertions depend on test order.
    """
    import logging

    from pyrtc.logging_utils import ensure_logging_configured

    pyrtc_logger = ensure_logging_configured()
    previous = pyrtc_logger.propagate
    pyrtc_logger.propagate = True
    yield
    logging.getLogger("pyrtc").propagate = previous


@pytest.fixture
def unique_name():
    def _make(prefix="test_shm"):
        short = uuid.uuid4().hex[:8]
        return f"{prefix[:8]}_{short}"

    return _make


# -- frame orientation (#162) -------------------------------------------------

#: A non-square camera frame (rows, columns) and an asymmetric feature in it.
ORIENTATION_FRAME_SHAPE = (24, 32)
ORIENTATION_HOT_PIXEL = (5, 21)  # (row, column) = (y, x)
ORIENTATION_SUBAP = 8  # 3 x 3 sub-apertures of 8 x 8 pixels in a 24 x 32 frame


def hot_pixel_frame(shape=ORIENTATION_FRAME_SHAPE, hot=ORIENTATION_HOT_PIXEL, dtype="uint16"):
    """A camera frame ``(height, width)`` that is dark except one pixel at ``hot`` (row, col)."""

    np = _np()
    frame = np.full(shape, 10, dtype=dtype)
    frame[hot] = 1000
    return frame


def shwfs_spot_frame(shift_x, shift_y, shape=ORIENTATION_FRAME_SHAPE, spacing=ORIENTATION_SUBAP):
    """A Shack-Hartmann camera frame ``[y, x]`` with every spot moved by ``(shift_x, shift_y)`` px.

    x is the column and y the row. The spots are Gaussians centred on their
    sub-aperture's centre ``(spacing - 1) / 2`` plus the shift, so an
    unshifted frame reads zero slopes (CONVENTIONS 1.2).
    """

    np = _np()
    height, width = shape
    rows, cols = np.indices(shape, dtype=np.float64)
    centre = (spacing - 1) / 2.0
    dy = (rows % spacing) - centre - shift_y
    dx = (cols % spacing) - centre - shift_x
    frame = 50.0 + 5000.0 * np.exp(-(dx**2 + dy**2) / (2.0 * 0.9**2))
    return np.rint(frame).astype(np.uint16)


def shwfs_slopes_from_stream(wfs_stream_name, spacing=ORIENTATION_SUBAP, **slopes_conf):
    """``(sx, sy)`` per sub-aperture from a real SHWFS ``SlopesProcess`` on ``wfs_stream_name``.

    Zero reference slopes, so the slopes are raw spot positions in pixels.
    """

    np = _np()
    from pyrtc.slopes_process import SlopesProcess
    from pyrtc.streams import clear_shms

    suffix = uuid.uuid4().hex[:8]
    outputs = {"signal": f"sig_{suffix}", "signal_2d": f"sig2d_{suffix}"}
    conf = {
        "type": "SHWFS",
        "signal_type": "slopes",
        "sub_ap_spacing": spacing,
        "sub_ap_offset_x": 0,
        "sub_ap_offset_y": 0,
        "image_noise": 1.0,
        "contrast": 100.0,  # threshold 100 counts: background off, spots on
        "functions": [],
        "input_streams": {"wfs": wfs_stream_name},
        "output_streams": outputs,
        **slopes_conf,
    }
    proc = SlopesProcess(conf)
    try:
        proc.compute_signal()
        signal = np.array(proc.read(block=False), dtype=np.float64)
    finally:
        proc.close()
        clear_shms(list(outputs.values()))
    half = signal.size // 2
    return signal[:half], signal[half:]


def assert_wfs_follows_camera_axes(wfs, set_camera_frame):
    """Check a WFS adapter publishes camera frames as ``[y, x]`` with the right slope signs.

    ``set_camera_frame(frame)`` makes the fake SDK return ``frame``, a
    ``(height, width)`` array, on the next ``wfs.expose()``.
    """

    np = _np()
    frame = hot_pixel_frame()
    set_camera_frame(frame)
    wfs.expose()
    raw = np.asarray(wfs._stream_object("wfs_raw").read())
    assert raw.shape == ORIENTATION_FRAME_SHAPE  # (height, width)
    assert np.unravel_index(np.argmax(raw), raw.shape) == ORIENTATION_HOT_PIXEL
    processed = np.asarray(wfs._stream_object("wfs").read())
    assert np.unravel_index(np.argmax(processed), processed.shape) == ORIENTATION_HOT_PIXEL

    wfs_name = wfs._stream_object("wfs").name
    for shift_x, shift_y in ((1.0, 0.0), (0.0, 1.0)):
        set_camera_frame(shwfs_spot_frame(shift_x, shift_y))
        wfs.expose()
        sx, sy = shwfs_slopes_from_stream(wfs_name)
        assert sx.size == sy.size == 9  # 3 x 3 sub-apertures
        # A spot moved along +x (columns) gives +sx and no sy, and vice versa.
        np.testing.assert_allclose(sx, shift_x, atol=0.02)
        np.testing.assert_allclose(sy, shift_y, atol=0.02)
