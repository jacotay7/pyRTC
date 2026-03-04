import numpy as np
import importlib

sci_mod = importlib.import_module("pyRTC.ScienceCamera")


def test_science_camera_core(monkeypatch, tmp_path):
    from conftest import DummySHM

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
