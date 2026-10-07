"""Micro-Manager camera adapters against a fake pymmcore-plus (#133)."""

import sys
import types

import numpy as np
import pytest

from testsupport import (
    ORIENTATION_FRAME_SHAPE,
    ORIENTATION_HOT_PIXEL,
    assert_wfs_follows_camera_axes,
    hot_pixel_frame,
    private_stream,
)


class _Core:
    instances = []

    def __init__(self):
        self.calls, self.queue, self.running = [], [], False
        self.camera = "Camera"
        _Core.instances.append(self)

    def setDeviceAdapterSearchPaths(self, paths):
        self.calls.append(("paths", tuple(paths)))

    def loadSystemConfiguration(self, path):
        self.calls.append(("config", path))

    def setCameraDevice(self, label):
        self.camera = label
        self.calls.append(("camera", label))

    def getCameraDevice(self):
        return self.camera

    def setExposure(self, ms):
        self.calls.append(("exposure", ms))

    def setProperty(self, device, name, value):
        self.calls.append(("property", device, name, value))

    def setROI(self, x, y, w, h):
        self.calls.append(("roi", x, y, w, h))

    def startContinuousSequenceAcquisition(self, interval):
        self.running = True

    def isSequenceRunning(self):
        return self.running

    def stopSequenceAcquisition(self):
        self.running = False

    def reset(self):
        self.calls.append(("reset",))

    def getRemainingImageCount(self):
        return len(self.queue)

    def popNextImage(self):
        return self.queue.pop(0)


@pytest.fixture
def mm(monkeypatch):
    _Core.instances.clear()
    fake = types.ModuleType("pymmcore_plus")
    fake.CMMCorePlus = _Core
    fake.find_micromanager = lambda: "/opt/micro-manager"
    monkeypatch.setitem(sys.modules, "pymmcore_plus", fake)
    import pyrtc.hardware.micromanager_camera as module
    import pyrtc.science_camera as science_camera
    import pyrtc.wavefront_sensor as wavefront_sensor

    monkeypatch.setattr(wavefront_sensor, "create_stream", private_stream)
    monkeypatch.setattr(science_camera, "create_stream", private_stream)
    return module


def _conf(**extra):
    return {
        "name": "wfs",
        "width": 4,
        "height": 3,
        "functions": [],
        "mm_config": "/lab/rig.cfg",
        **extra,
    }


def test_wfs_loads_config_applies_settings_and_streams_the_newest_frame(mm):
    wfs = mm.MicroManagerWFS(
        _conf(
            camera="Andor",
            exposure=2.5,
            binning=2,
            left=10,
            top=20,
            properties={"Andor": {"Gain": 100, "ReadoutMode": "10MHz"}},
            frame_timeout=0.1,
        )
    )
    core = _Core.instances[-1]
    try:
        assert core.calls[:3] == [
            ("paths", ("/opt/micro-manager",)),
            ("config", "/lab/rig.cfg"),
            ("camera", "Andor"),
        ]
        assert ("property", "Andor", "Binning", "2") in core.calls
        assert ("roi", 10, 20, 4, 3) in core.calls
        assert ("exposure", 2.5) in core.calls
        assert core.calls[-2:] == [
            ("property", "Andor", "Gain", 100),
            ("property", "Andor", "ReadoutMode", "10MHz"),
        ]
        assert core.running

        old = np.zeros((3, 4), dtype=np.uint16)
        new = np.arange(12, dtype=np.uint16).reshape(3, 4)
        core.queue = [old, new]
        wfs.expose()
        np.testing.assert_array_equal(wfs.data, new)  # newest frame, as (height, width)
        np.testing.assert_array_equal(wfs._stream_object("wfs_raw").read(), new)
        assert core.queue == []

        with pytest.raises(TimeoutError, match="no frame"):
            wfs.expose()

        wfs.set_exposure(1.0)
        assert core.calls[-1] == ("exposure", 1.0)
    finally:
        wfs.close()
    assert not core.running and core.calls[-1] == ("reset",)


def test_science_camera_and_default_camera(mm):
    camera = mm.MicroManagerScienceCamera(
        {
            "name": "psf",
            "width": 4,
            "height": 3,
            "dark_count": 1,
            "integration": 2,
            "functions": [],
            "mm_config": "/lab/rig.cfg",
            "mm_path": "/custom/mm",
        }
    )
    core = _Core.instances[-1]
    try:
        assert core.calls[0] == ("paths", ("/custom/mm",))
        assert camera.camera_label == "Camera"
        core.queue = [np.ones((3, 4), dtype=np.uint16)]
        camera.expose()
        assert camera.data.shape == (3, 4)
    finally:
        camera.close()


def test_wfs_frames_follow_camera_axes(mm):
    """#162: a camera frame lands at [y, x], and a spot moved along +x gives +sx."""

    height, width = ORIENTATION_FRAME_SHAPE
    wfs = mm.MicroManagerWFS(_conf(width=width, height=height))
    core = _Core.instances[-1]
    try:
        assert_wfs_follows_camera_axes(wfs, lambda frame: core.queue.append(frame))
    finally:
        wfs.close()


def test_science_camera_frames_follow_camera_axes(mm):
    height, width = ORIENTATION_FRAME_SHAPE
    camera = mm.MicroManagerScienceCamera(
        {
            "name": "psf",
            "width": width,
            "height": height,
            "dark_count": 1,
            "integration": 1,
            "functions": [],
            "mm_config": "/lab/rig.cfg",
        }
    )
    try:
        _Core.instances[-1].queue.append(hot_pixel_frame())
        camera.expose()
        short = np.asarray(camera._stream_object("psf_short").read())
        assert short.shape == ORIENTATION_FRAME_SHAPE
        assert np.unravel_index(np.argmax(short), short.shape) == ORIENTATION_HOT_PIXEL
    finally:
        camera.close()


def test_config_is_required(mm):
    with pytest.raises(ValueError, match="mm_config"):
        mm.MicroManagerWFS({"name": "wfs", "width": 4, "height": 3, "functions": []})


def test_missing_pymmcore_plus_names_the_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "pymmcore_plus", None)
    import pyrtc.hardware.micromanager_camera as module
    import pyrtc.wavefront_sensor as wavefront_sensor

    monkeypatch.setattr(wavefront_sensor, "create_stream", private_stream)
    with pytest.raises(ImportError, match=r"pyrtcao\[micromanager\]"):
        module.MicroManagerWFS(_conf())
