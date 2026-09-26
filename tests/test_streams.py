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
    # The dtype check fires before any CUDA work happens, so this is safe to
    # exercise without a GPU.
    monkeypatch.setattr(streams.pyshmem, "GPU_SUPPORTED_DTYPES", frozenset({np.dtype(np.float32)}))
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


def _cleanup_shm(shm):
    try:
        shm.unlink()
    except Exception:
        pass


def test_normalize_gpu_device_falls_back(monkeypatch):
    monkeypatch.setattr(streams, "TORCH_AVAILABLE", False)
    assert streams.normalize_gpu_device("cuda:0", "ctx") is None


def test_create_stream_cpu_read_write(unique_name):
    name = unique_name("img")
    shm = streams.create_stream(name, (4, 3), np.float32)
    try:
        arr = np.arange(12, dtype=np.float32).reshape(4, 3)
        shm.write(arr)
        assert np.array_equal(shm.read(), arr)
        assert shm.count == 1
        assert shm.write_time > 0
    finally:
        _cleanup_shm(shm)


def test_create_stream_reuses_matching_stream(unique_name):
    name = unique_name("reuse")
    first = streams.create_stream(name, (2, 2), np.int32)
    try:
        first.write(np.array([[1, 2], [3, 4]], dtype=np.int32))
        second = streams.create_stream(name, (2, 2), np.int32)
        assert np.array_equal(second.read(), np.array([[1, 2], [3, 4]], dtype=np.int32))
        second.close()
    finally:
        _cleanup_shm(first)


def test_create_stream_rebuilds_on_mismatch(unique_name):
    name = unique_name("rebuild")
    first = streams.create_stream(name, (2, 2), np.int32)
    first.close()
    second = streams.create_stream(name, (3,), np.float32)
    try:
        assert tuple(second.shape) == (3,)
        assert second.dtype == np.float32
    finally:
        _cleanup_shm(second)


def test_open_stream_attaches_to_existing(unique_name):
    name = unique_name("existing")
    prod = streams.create_stream(name, (2, 2), np.int32)
    try:
        prod.write(np.array([[1, 2], [3, 4]], dtype=np.int32))
        cons = streams.open_stream(name)
        assert tuple(cons.shape) == (2, 2)
        assert np.dtype(cons.dtype) == np.dtype(np.int32)
        assert np.array_equal(cons.read(), np.array([[1, 2], [3, 4]], dtype=np.int32))
        cons.close()
    finally:
        _cleanup_shm(prod)


def test_open_stream_missing_raises(unique_name):
    name = unique_name("missing")
    try:
        streams.open_stream(name)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("expected FileNotFoundError")


def test_clear_shms(unique_name):
    import pyshmem

    name = unique_name("clear")
    shm = streams.create_stream(name, (1,), np.uint8)
    shm.close()
    streams.clear_shms([name])
    assert name not in pyshmem.list_streams()
    # clearing again must not raise
    streams.clear_shms([name])


def test_expected_output_specs_follow_component_output_aliases():
    config = {
        "wfs": {
            "class_name": "WavefrontSensor",
            "width": 16,
            "height": 16,
            "dark_count": 1,
            "output_streams": {"wfs_raw": "raw_custom", "wfs": "wfs_custom"},
        },
        "slopes": {
            "class_name": "SlopesProcess",
            "type": "SHWFS",
            "signal_type": "slopes",
            "sub_ap_spacing": 4,
            "sub_ap_offset_x": 0,
            "sub_ap_offset_y": 0,
            "output_streams": {"signal": "signal_custom", "signal_2d": "signal2d_custom"},
        },
        "wfc": {
            "class_name": "WavefrontCorrector",
            "name": "dm",
            "num_actuators": 8,
            "num_modes": 8,
            "output_streams": {"wfc": "wfc_custom", "wfc_2d": "wfc2d_custom"},
        },
        "psf": {
            "class_name": "ScienceCamera",
            "name": "cam",
            "width": 8,
            "height": 8,
            "dark_count": 1,
            "integration": 1,
            "output_streams": {
                "psf_short": "short_custom",
                "psf_long": "long_custom",
                "strehl": "strehl_custom",
                "tiptilt": "tiptilt_custom",
            },
        },
    }

    specs = streams.expected_output_shm_specs_for_config(config)

    assert "raw_custom" in specs
    assert "wfs_custom" in specs
    assert "signal_custom" in specs
    assert "signal2d_custom" in specs
    assert "wfc_custom" in specs
    assert "wfc2d_custom" in specs
    assert "short_custom" in specs
