import sys
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


class DummySHM:
    """In-memory stand-in mimicking the pyshmem.SharedMemory API."""

    def __init__(self, name, shape, dtype, gpu_device=None):
        self.name = name
        self.shape = tuple(shape)
        self.dtype = _np().dtype(dtype)
        self.gpu_device = gpu_device
        self.arr = _np().zeros(self.shape, dtype=self.dtype)
        self._count = 0
        self._write_time = 0.0

    @property
    def count(self):
        return self._count

    @property
    def write_time(self):
        return self._write_time

    def write(self, arr):
        np = _np()
        arr = np.asarray(arr, dtype=self.dtype)
        np.copyto(self.arr, arr.reshape(self.shape))
        self._count += 1
        self._write_time = float(self._count)

    def read(self, out=None):
        if out is not None:
            _np().copyto(out, self.arr)
            return out
        return _np().copy(self.arr)

    def read_new(self, timeout=None, out=None):
        return self.read(out=out)

    def close(self):
        pass

    def unlink(self):
        pass


class FakeStream:
    """Minimal pyshmem-like stream double recording writes."""

    def __init__(self, arr):
        self.arr = _np().asarray(arr)
        self.writes = []
        self._count = 1
        self._write_time = 1.0

    @property
    def count(self):
        return self._count

    @property
    def write_time(self):
        return self._write_time

    def read(self, out=None):
        if out is not None:
            _np().copyto(out, self.arr)
            return out
        return _np().copy(self.arr)

    def read_new(self, timeout=None, out=None):
        return self.read(out=out)

    def write(self, arr):
        self.writes.append(_np().asarray(arr))
        self._count += 1
        self._write_time = float(self._count)


@pytest.fixture
def unique_name():
    def _make(prefix="test_shm"):
        short = uuid.uuid4().hex[:8]
        return f"{prefix[:8]}_{short}"

    return _make
