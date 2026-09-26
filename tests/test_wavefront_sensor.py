import importlib
import threading
import time

import numpy as np
import pytest

wfs_mod = importlib.import_module("pyrtc.wavefront_sensor")


def test_downsample_and_rotate_helpers():
    img = np.arange(16, dtype=np.int32).reshape(4, 4)
    ds = wfs_mod.downsample_int32_image_jit(img, 2)
    assert ds.shape == (2, 2)
    rot = wfs_mod.rotate_image_jit(img, 0.0)
    assert rot.shape == img.shape


def test_wavefront_sensor_basic(monkeypatch, tmp_path):
    from testsupport import private_stream

    monkeypatch.setattr(wfs_mod, "create_stream", private_stream)

    conf = {
        "name": "w",
        "width": 8,
        "height": 8,
        "dark_count": 2,
        "dark_file": "",
        "functions": [],
    }
    wfs = wfs_mod.WavefrontSensor(conf)
    assert wfs.name == "w"

    wfs.set_roi([2, 3, 4, 5])
    wfs.set_exposure(1.2)
    wfs.set_binning(2)
    wfs.set_gain(3.4)
    wfs.set_bit_depth(12)

    wfs.data = np.ones((8, 8), dtype=np.uint16) * 4
    wfs.set_dark(np.ones((8, 8), dtype=np.int32))
    wfs.expose()
    out = wfs.read(block=False)
    assert out.shape == (8, 8)
    assert np.all(out == 3)

    dark_file = tmp_path / "dark.npy"
    wfs.save_dark(str(dark_file))
    wfs.set_dark(np.zeros((8, 8), dtype=np.int32))
    wfs.load_dark(str(dark_file))
    assert np.all(wfs.dark == 1)

    # dark-taking path: the first raw frame is discarded, then two averaged.
    frames = [np.full((8, 8), value, dtype=np.uint16) for value in (100, 2, 5)]
    wfs.read_stream = lambda name, **_kwargs: frames.pop(0)
    wfs.take_dark()
    assert np.all(wfs.dark == 4)  # rint(3.5)
    del wfs.read_stream

    wfs.read = lambda block=False: np.ones((8, 8), dtype=np.int32)
    rot = wfs.rotate_image(10.0)
    assert rot.shape == (8, 8)


@pytest.mark.parametrize("extra", [{"downsample_factor": 2}, {"rotation_angle": 30.0}])
def test_take_dark_matches_the_raw_frame_it_is_subtracted_from(monkeypatch, extra):
    from testsupport import private_stream

    monkeypatch.setattr(wfs_mod, "create_stream", private_stream)
    conf = {"name": "w", "width": 8, "height": 8, "dark_count": 3, "functions": [], **extra}
    wfs = wfs_mod.WavefrontSensor(conf)
    wfs.data = np.full((8, 8), 7, dtype=np.uint16)

    stop = threading.Event()

    def _camera():
        while not stop.is_set():
            wfs.expose()
            time.sleep(1e-3)

    camera = threading.Thread(target=_camera, daemon=True)
    camera.start()
    try:
        wfs.take_dark()
    finally:
        stop.set()
        camera.join()

    assert wfs.dark.shape == (8, 8)
    assert np.all(wfs.dark == 7)
    wfs.expose()  # must not fail to broadcast
    assert np.all(wfs.read(block=False) == 0)
