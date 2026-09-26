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


@contextlib.contextmanager
def publishing_chain(names, *, step_seconds=1e-3, stamp_frame_ids=True):
    """Run a background producer writing frames through ``names`` in order.

    Each frame is written to every stream in turn, ``step_seconds`` apart,
    stamped with the same frame id (as pyrtc components do), so stream ``i+1``
    lags stream ``i`` by one step. Yields an opener mapping a logical name to a
    fresh read-only handle, suitable for ``latency.open_stream``.
    """
    import pyshmem

    streams = {name: private_stream(name, (1,), "float32") for name in names}
    stop = threading.Event()

    def _run():
        frame_id = 0
        while not stop.is_set():
            frame_id += 1
            for stream in streams.values():
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


@pytest.fixture
def unique_name():
    def _make(prefix="test_shm"):
        short = uuid.uuid4().hex[:8]
        return f"{prefix[:8]}_{short}"

    return _make
