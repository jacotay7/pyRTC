import importlib

import numpy as np

from testsupport import DummySHM


synthetic_mod = importlib.import_module("pyrtc.hardware.synthetic_systems")
wfs_mod = importlib.import_module("pyrtc.wavefront_sensor")
science_mod = importlib.import_module("pyrtc.science_camera")


def test_synthetic_shwfs_generates_frame_and_responds_to_correction(monkeypatch):
    streams = {}

    def _make_stream(name, shape, dtype, gpu_device=None):
        stream = DummySHM(name, shape, dtype, gpu_device=gpu_device)
        streams[name] = stream
        return stream

    def _open_existing(name, gpu_device=None):
        return streams[name]

    monkeypatch.setattr(wfs_mod, "create_stream", _make_stream)
    monkeypatch.setattr(science_mod, "create_stream", _make_stream)
    monkeypatch.setattr(synthetic_mod, "open_stream", _open_existing)

    streams["wfc"] = DummySHM("wfc", (32,), np.float32)

    sensor = synthetic_mod.SyntheticSHWFS(
        {
            "name": "synthetic-wfs",
            "width": 32,
            "height": 32,
            "dark_count": 1,
            "sub_ap_spacing": 8,
            "sub_ap_offset_x": 0,
            "sub_ap_offset_y": 0,
            "num_modes": 32,
            "functions": [],
        }
    )

    sensor.expose()
    nominal_image = streams["wfs"].read()

    streams["wfc"].write(np.full((32,), 0.35, dtype=np.float32))
    sensor.expose()
    corrected_image = streams["wfs"].read()

    assert nominal_image.shape == (32, 32)
    assert corrected_image.shape == (32, 32)
    assert sensor.frame_counter == 2
    assert np.max(nominal_image) > 0
    assert not np.array_equal(nominal_image, corrected_image)


def test_synthetic_science_camera_updates_strehl_from_signal(monkeypatch):
    streams = {}

    def _make_stream(name, shape, dtype, gpu_device=None):
        stream = DummySHM(name, shape, dtype, gpu_device=gpu_device)
        streams[name] = stream
        return stream

    def _open_existing(name, gpu_device=None):
        return streams[name]

    monkeypatch.setattr(wfs_mod, "create_stream", _make_stream)
    monkeypatch.setattr(science_mod, "create_stream", _make_stream)
    monkeypatch.setattr(synthetic_mod, "open_stream", _open_existing)

    streams["signal"] = DummySHM("signal", (32,), np.float32)
    camera = synthetic_mod.SyntheticScienceCamera(
        {
            "name": "synthetic-psf",
            "width": 48,
            "height": 48,
            "dark_count": 1,
            "integration": 2,
            "functions": [],
        }
    )

    streams["signal"].write(np.zeros((32,), dtype=np.float32))
    camera.expose()
    high_strehl = streams["strehl"].read()[0]
    sharp_peak = float(np.max(np.asarray(streams["psf_short"].read(), dtype=np.float32)))

    streams["signal"].write(np.full((32,), 0.5, dtype=np.float32))
    camera.expose()
    low_strehl = streams["strehl"].read()[0]
    blurred_peak = float(np.max(np.asarray(streams["psf_short"].read(), dtype=np.float32)))

    assert camera.frame_counter == 2
    assert high_strehl > low_strehl
    assert sharp_peak > blurred_peak
