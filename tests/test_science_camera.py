import numpy as np
import importlib

import pytest

sci_mod = importlib.import_module("pyRTC.ScienceCamera")


def test_science_camera_core(monkeypatch, tmp_path):
    from testsupport import DummySHM

    monkeypatch.setattr(sci_mod, "ImageSHM", DummySHM)

    conf = {
        "name": "psf",
        "width": 8,
        "height": 8,
        "darkCount": 2,
        "integration": 2,
        "functions": [],
    }
    cam = sci_mod.ScienceCamera(conf)

    cam.setRoi([2, 3, 4, 5])
    cam.setExposure(10)
    cam.setBinning(2)
    cam.setGain(1)
    cam.setGamma(2.2)
    cam.setBitDepth(16)
    cam.setIntegrationLength(2)

    cam.data = np.ones((8, 8), dtype=np.uint16) * 5
    cam.dark = np.ones((8, 8), dtype=np.int32) * 2
    cam.expose()
    assert np.all(cam.read(block=False) == 3)

    # Integrate from mocked reads
    frames = [np.ones((8, 8), dtype=np.int32) * 3, np.ones((8, 8), dtype=np.int32) * 5]
    cam.read = lambda block=True: frames.pop(0)
    cam.integrate()
    assert np.allclose(cam.readLong(), 4)

    dark_file = tmp_path / "dark.npy"
    model_file = tmp_path / "model.npy"
    cam.setDark(np.ones((8, 8), dtype=np.int32) * 7)
    cam.saveDark(str(dark_file))
    cam.loadDark(str(dark_file))
    assert np.all(cam.dark == 7)

    cam.setModelPSF(np.ones((8, 8), dtype=np.float64) * 10)
    cam.saveModelPSF(str(model_file))
    cam.loadModelPSF(str(model_file))
    assert np.all(cam.model == 10)

    cam.psfLong.write(np.ones((8, 8), dtype=np.float64) * 10)
    sr = cam.computeStrehl()
    assert np.isclose(sr, 1.0)


def test_science_camera_default_files_plot_and_error_paths(monkeypatch, tmp_path):
    from testsupport import DummySHM

    monkeypatch.setattr(sci_mod, "ImageSHM", DummySHM)

    conf = {
        "name": "psf",
        "width": 4,
        "height": 4,
        "darkCount": 2,
        "integration": 2,
        "functions": [],
    }
    cam = sci_mod.ScienceCamera(conf)

    assert np.array_equal(cam.dark, np.zeros((4, 4), dtype=np.int32))
    assert np.array_equal(cam.model, np.zeros((4, 4), dtype=np.float64))

    with pytest.raises(ValueError, match="No dark frame filename provided"):
        cam.saveDark()
    with pytest.raises(ValueError, match="No model PSF filename provided"):
        cam.saveModelPSF()

    frame = np.arange(16, dtype=np.int32).reshape(4, 4)
    cam.psfShort.write(frame)
    monkeypatch.setattr(sci_mod.plt, "imshow", lambda *args, **kwargs: None)
    monkeypatch.setattr(sci_mod.plt, "colorbar", lambda *args, **kwargs: None)
    monkeypatch.setattr(sci_mod.plt, "show", lambda: None)
    cam.plot()

    cam.psfLong.write(np.ones((4, 4), dtype=np.float64) * 5)
    cam.takeModelPSF()
    assert np.array_equal(cam.model, np.ones((4, 4), dtype=np.float64) * 5)

    frames = [np.ones((4, 4), dtype=np.int32) * 2, np.ones((4, 4), dtype=np.int32) * 4]
    cam.read = lambda block=True: frames.pop(0)
    cam.takeDark()
    assert np.all(cam.dark == 3)


def test_science_camera_setter_and_load_error_paths(monkeypatch):
    from testsupport import DummySHM

    monkeypatch.setattr(sci_mod, "ImageSHM", DummySHM)
    cam = sci_mod.ScienceCamera(
        {
            "name": "psf",
            "width": 4,
            "height": 4,
            "darkCount": 1,
            "integration": 1,
            "functions": [],
        }
    )

    class BadLogger:
        def info(self, *args, **kwargs):
            raise RuntimeError("log failed")

        def exception(self, *args, **kwargs):
            return None

    cam.logger = BadLogger()

    with pytest.raises(RuntimeError, match="log failed"):
        cam.setRoi([1, 2, 3, 4])
    with pytest.raises(RuntimeError, match="log failed"):
        cam.setExposure(1)
    with pytest.raises(RuntimeError, match="log failed"):
        cam.setBinning(1)
    with pytest.raises(RuntimeError, match="log failed"):
        cam.setGain(1)
    with pytest.raises(RuntimeError, match="log failed"):
        cam.setGamma(1.0)
    with pytest.raises(RuntimeError, match="log failed"):
        cam.setBitDepth(16)
    with pytest.raises(RuntimeError, match="log failed"):
        cam.setIntegrationLength(1)
    with pytest.raises(RuntimeError, match="log failed"):
        cam.setDark(np.ones((4, 4), dtype=np.int32))
    with pytest.raises(RuntimeError, match="log failed"):
        cam.setModelPSF(np.ones((4, 4), dtype=np.float64))

    cam = sci_mod.ScienceCamera(
        {
            "name": "psf",
            "width": 4,
            "height": 4,
            "darkCount": 0,
            "integration": 1,
            "functions": [],
        }
    )
    with pytest.raises(ValueError, match="darkCount must be at least 1"):
        cam.takeDark()

    monkeypatch.setattr(sci_mod.np, "load", lambda filename: (_ for _ in ()).throw(RuntimeError("load failed")))
    with pytest.raises(RuntimeError, match="load failed"):
        cam.loadDark("missing-dark.npy")
    with pytest.raises(RuntimeError, match="load failed"):
        cam.loadModelPSF("missing-model.npy")
