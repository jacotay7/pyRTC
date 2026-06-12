"""Tests for the pyshmem-backed stream policy (pyrtc.streams)."""

import numpy as np

import pyrtc.streams as streams
from pyrtc.streams import _existing_shm_spec, clear_shms, create_stream


def test_create_stream_falls_back_when_torch_missing(monkeypatch, unique_name):
    name = unique_name("notorch")
    monkeypatch.setattr(streams, "TORCH_AVAILABLE", False)
    shm = create_stream(name, (4,), np.float32, gpu_device="cuda:0")
    try:
        assert not shm.gpu_enabled
    finally:
        shm.unlink()


def test_create_stream_falls_back_when_cuda_unavailable(monkeypatch, unique_name):
    name = unique_name("nocuda")
    monkeypatch.setattr(streams, "TORCH_AVAILABLE", True)
    monkeypatch.setattr(streams.pyshmem, "gpu_available", lambda: False)
    shm = create_stream(name, (4,), np.float32, gpu_device="cuda:0")
    try:
        assert not shm.gpu_enabled
    finally:
        shm.unlink()


def test_create_stream_falls_back_for_unsupported_gpu_dtype(monkeypatch, unique_name):
    name = unique_name("u16")
    monkeypatch.setattr(streams, "TORCH_AVAILABLE", True)
    monkeypatch.setattr(streams.pyshmem, "gpu_available", lambda: True)
    # uint16 has no torch equivalent, so the dtype check fires before any
    # CUDA work happens — safe to exercise without a GPU.
    shm = create_stream(name, (4,), np.uint16, gpu_device="cuda:0")
    try:
        assert not shm.gpu_enabled
        shm.write(np.arange(4, dtype=np.uint16))
        assert np.array_equal(shm.read(), np.arange(4, dtype=np.uint16))
    finally:
        shm.unlink()


def test_existing_shm_spec_reports_shape_and_dtype(unique_name):
    name = unique_name("spec")
    shm = create_stream(name, (3, 2), np.int32)
    try:
        assert _existing_shm_spec(name) == ((3, 2), np.dtype(np.int32))
    finally:
        shm.unlink()


def test_existing_shm_spec_missing_stream_returns_none(unique_name):
    assert _existing_shm_spec(unique_name("missing")) is None


def test_clear_shms_tolerates_missing_streams(unique_name):
    clear_shms([unique_name("ghost-a"), unique_name("ghost-b")])
