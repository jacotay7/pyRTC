import importlib

import numpy as np

from testsupport import DummySHM


synthetic_mod = importlib.import_module("pyRTC.hardware.SyntheticSystems")
wfs_mod = importlib.import_module("pyRTC.WavefrontSensor")
science_mod = importlib.import_module("pyRTC.ScienceCamera")


def test_synthetic_shwfs_generates_frame_and_responds_to_correction(monkeypatch):
    streams = {}

    def _make_stream(name, shape, dtype, gpuDevice=None, consumer=True):
        stream = DummySHM(name, shape, dtype, gpuDevice=gpuDevice, consumer=consumer)
        streams[name] = stream
        return stream

    def _init_existing(name, gpuDevice=None):
        stream = streams[name]
        return stream, list(stream.shape), stream.dtype

    monkeypatch.setattr(wfs_mod, "ImageSHM", _make_stream)
    monkeypatch.setattr(science_mod, "ImageSHM", _make_stream)
    monkeypatch.setattr(synthetic_mod, "initExistingShm", _init_existing)

    streams["wfc"] = DummySHM("wfc", (32,), np.float32, consumer=False)

    sensor = synthetic_mod.SyntheticSHWFS(
        {
            "name": "synthetic-wfs",
            "width": 32,
            "height": 32,
            "darkCount": 1,
            "subApSpacing": 8,
            "subApOffsetX": 0,
            "subApOffsetY": 0,
            "numModes": 32,
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
    assert sensor.frameCounter == 2
    assert np.max(nominal_image) > 0
    assert not np.array_equal(nominal_image, corrected_image)


def test_synthetic_science_camera_updates_strehl_from_signal(monkeypatch):
    streams = {}

    def _make_stream(name, shape, dtype, gpuDevice=None, consumer=True):
        stream = DummySHM(name, shape, dtype, gpuDevice=gpuDevice, consumer=consumer)
        streams[name] = stream
        return stream

    def _init_existing(name, gpuDevice=None):
        stream = streams[name]
        return stream, list(stream.shape), stream.dtype

    monkeypatch.setattr(wfs_mod, "ImageSHM", _make_stream)
    monkeypatch.setattr(science_mod, "ImageSHM", _make_stream)
    monkeypatch.setattr(synthetic_mod, "initExistingShm", _init_existing)

    streams["signal"] = DummySHM("signal", (32,), np.float32, consumer=False)
    camera = synthetic_mod.SyntheticScienceCamera(
        {
            "name": "synthetic-psf",
            "width": 48,
            "height": 48,
            "darkCount": 1,
            "integration": 2,
            "functions": [],
        }
    )

    streams["signal"].write(np.zeros((32,), dtype=np.float32))
    camera.expose()
    high_strehl = streams["strehl"].read()[0]
    sharp_peak = float(np.max(np.asarray(streams["psfShort"].read(), dtype=np.float32)))

    streams["signal"].write(np.full((32,), 0.5, dtype=np.float32))
    camera.expose()
    low_strehl = streams["strehl"].read()[0]
    blurred_peak = float(np.max(np.asarray(streams["psfShort"].read(), dtype=np.float32)))

    assert camera.frameCounter == 2
    assert high_strehl > low_strehl
    assert sharp_peak > blurred_peak