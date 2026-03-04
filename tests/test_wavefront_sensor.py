import numpy as np
import importlib

wfs_mod = importlib.import_module("pyRTC.WavefrontSensor")


def test_downsample_and_rotate_helpers():
    img = np.arange(16, dtype=np.int32).reshape(4, 4)
    ds = wfs_mod.downsample_int32_image_jit(img, 2)
    assert ds.shape == (2, 2)
    rot = wfs_mod.rotate_image_jit(img, 0.0)
    assert rot.shape == img.shape


def test_wavefront_sensor_basic(monkeypatch, tmp_path):
    from conftest import DummySHM

    monkeypatch.setattr(wfs_mod, "ImageSHM", DummySHM)

    conf = {
        "name": "w",
        "width": 8,
        "height": 8,
        "darkCount": 2,
        "darkFile": "",
        "functions": [],
    }
    wfs = wfs_mod.WavefrontSensor(conf)
    assert wfs.name == "w"

    wfs.setRoi([2, 3, 4, 5])
    wfs.setExposure(1.2)
    wfs.setBinning(2)
    wfs.setGain(3.4)
    wfs.setBitDepth(12)

    wfs.data = np.ones((8, 8), dtype=np.uint16) * 4
    wfs.setDark(np.ones((8, 8), dtype=np.int32))
    wfs.expose()
    out = wfs.read(block=False)
    assert out.shape == (8, 8)
    assert np.all(out == 3)

    dark_file = tmp_path / "dark.npy"
    wfs.saveDark(str(dark_file))
    wfs.setDark(np.zeros((8, 8), dtype=np.int32))
    wfs.loadDark(str(dark_file))
    assert np.all(wfs.dark == 1)

    # dark-taking path
    frames = [np.ones((8, 8), dtype=np.int32) * 2, np.ones((8, 8), dtype=np.int32) * 4]

    def fake_read(block=True):
        return frames.pop(0)

    wfs.read = fake_read
    wfs.takeDark()
    assert np.all(wfs.dark == 3)

    wfs.read = lambda block=False: np.ones((8, 8), dtype=np.int32)
    rot = wfs.rotateImage(10.0)
    assert rot.shape == (8, 8)
