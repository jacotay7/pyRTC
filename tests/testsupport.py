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


fake_fits = types.SimpleNamespace(PrimaryHDU=_FakePrimaryHDU, open=_fake_fits_open)
fake_io = types.SimpleNamespace(fits=fake_fits)
fake_astropy = types.SimpleNamespace(io=fake_io)

sys.modules.setdefault("astropy", fake_astropy)
sys.modules.setdefault("astropy.io", fake_io)
sys.modules.setdefault("astropy.io.fits", fake_fits)


class DummySHM:
    def __init__(self, name, shape, dtype, gpuDevice=None, consumer=True):
        self.name = name
        self.shape = tuple(shape)
        self.dtype = _np().dtype(dtype)
        self.gpuDevice = gpuDevice
        self.arr = _np().zeros(self.shape, dtype=self.dtype)

    def write(self, arr):
        np = _np()
        arr = np.asarray(arr, dtype=self.dtype)
        np.copyto(self.arr, arr.reshape(self.shape))
        return 1

    def read(self, SAFE=True, GPU=False, RELEASE_GIL=True):
        if SAFE:
            return _np().copy(self.arr)
        return self.arr

    def read_noblock(self, SAFE=True, GPU=False):
        if SAFE:
            return _np().copy(self.arr)
        return self.arr


class FakeStream:
    def __init__(self, arr):
        self.arr = _np().asarray(arr)
        self.writes = []

    def read(self, SAFE=True, RELEASE_GIL=True):
        return _np().copy(self.arr)

    def read_noblock(self, SAFE=True):
        return _np().copy(self.arr)

    def write(self, arr):
        self.writes.append(_np().asarray(arr))


@pytest.fixture
def unique_name():
    def _make(prefix="test_shm"):
        short = uuid.uuid4().hex[:8]
        return f"{prefix[:8]}_{short}"

    return _make