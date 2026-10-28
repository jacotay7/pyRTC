"""Vimba X (Allied Vision) adapters against a fake vmbpy API."""

import sys
import types

import numpy as np
import pytest

from testsupport import (
    ORIENTATION_FRAME_SHAPE,
    ORIENTATION_HOT_PIXEL,
    assert_wfs_follows_camera_axes,
    hot_pixel_frame,
)
from testsupport import private_stream


class _FrameStatus:
    Complete = "complete"
    Incomplete = "incomplete"


class _Frame:
    def __init__(self, image, status=_FrameStatus.Complete):
        self.image, self.status = image, status

    def get_status(self):
        return self.status

    def as_numpy_ndarray(self):
        return self.image[..., np.newaxis]


class _Feature:
    def __init__(self, camera, name, value):
        self._camera, self._name, self._value = camera, name, value

    def get(self):
        return self._value

    def set(self, value):
        if (
            self._name == "TriggerMode"
            and self._camera.features["AcquisitionFrameRateEnable"]._value
        ):
            raise RuntimeError("TriggerMode is locked while the frame-rate limiter is on")
        if self._name in ("PixelFormat", "Width", "Height") and self._camera.streaming:
            raise RuntimeError(f"{self._name} cannot change while streaming")
        self._value = value
        self._camera.history.append((self._name, value))
        # Like a camera: binning n divides the sensor's width / height by n.
        if self._name == "BinningHorizontal":
            self._camera.features["Width"]._value = self._camera.sensor[1] // int(value)
        if self._name == "BinningVertical":
            self._camera.features["Height"]._value = self._camera.sensor[0] // int(value)


class _Camera:
    def __init__(self, serial, height=3, width=4):
        self.serial, self.history, self.queued = serial, [], []
        self.sensor = (height, width)
        self.entered = self.streaming = False
        self.buffer_count = None
        initial = {
            "ExposureAuto": "Continuous",
            "GainAuto": "Off",
            "TriggerMode": "On",
            "AcquisitionFrameRateEnable": True,
            "AcquisitionFrameRate": 10.0,
            "PixelFormat": "Mono8",
            "ExposureTime": 1000.0,
            "Gain": 0.0,
            "Width": width,
            "Height": height,
            "OffsetX": 0,
            "OffsetY": 0,
            "BinningHorizontal": 1,
            "BinningVertical": 1,
            "DeviceLinkThroughputLimit": 450000000,
        }
        self.features = {name: _Feature(self, name, value) for name, value in initial.items()}

    def get_serial(self):
        return self.serial

    def get_model(self):
        return "Alvium fake"

    def get_feature_by_name(self, name):
        if name not in self.features:
            raise LookupError(name)
        return self.features[name]

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, *exc):
        self.entered = False

    def start_streaming(self, handler, buffer_count):
        self.handler, self.buffer_count, self.streaming = handler, buffer_count, True

    def stop_streaming(self):
        self.streaming = False

    def queue_frame(self, frame):
        self.queued.append(frame)

    def deliver(self, image, status=_FrameStatus.Complete):
        assert self.streaming
        self.handler(self, None, _Frame(np.asarray(image), status))


class _VmbSystem:
    instance = None

    def __init__(self):
        self.cameras, self.paths, self.depth = [], None, 0

    @classmethod
    def get_instance(cls):
        return cls.instance

    def set_path_configuration(self, *paths):
        if self.depth:  # like vmbpy: only before the system is entered
            raise RuntimeError("set_path_configuration while the system is entered")
        self.paths = paths

    def __enter__(self):
        self.depth += 1
        return self

    def __exit__(self, *exc):
        self.depth -= 1

    def get_all_cameras(self):
        return tuple(self.cameras)


@pytest.fixture
def vimba(monkeypatch):
    _VmbSystem.instance = _VmbSystem()
    fake = types.SimpleNamespace(VmbSystem=_VmbSystem, FrameStatus=_FrameStatus)
    monkeypatch.setitem(sys.modules, "vmbpy", fake)

    import pyrtc.hardware.vimba_camera as module
    import pyrtc.science_camera as science_camera
    import pyrtc.wavefront_sensor as wavefront_sensor

    monkeypatch.setattr(wavefront_sensor, "create_stream", private_stream)
    monkeypatch.setattr(science_camera, "create_stream", private_stream)
    return module


def _wfs_conf(**extra):
    return {"name": "wfs", "width": 4, "height": 3, "functions": [], **extra}


def test_wfs_opens_by_serial_configures_and_streams(vimba):
    system = _VmbSystem.instance
    other, camera = _Camera("AAAA"), _Camera("CAM42")
    system.cameras = [other, camera]
    wfs = vimba.VimbaWFS(
        _wfs_conf(
            serial="CAM42",
            cti_path="/opt/VimbaX/cti/VimbaUSBTL.cti",
            exposure=250.0,
            gain=0.0,
            bit_depth=12,
            frame_rate=3.0,
            node_settings={"DeviceLinkThroughputLimit": 100000000},
            buffer_count=4,
            fetch_timeout=0.2,
        )
    )
    try:
        assert system.paths == ("/opt/VimbaX/cti/VimbaUSBTL.cti",) and system.depth == 1
        assert camera.entered and not other.entered
        assert camera.streaming and camera.buffer_count == 4
        names = [name for name, _ in camera.history]
        # Auto exposure off, then the limiter off so TriggerMode can be freed.
        assert names[:3] == ["ExposureAuto", "AcquisitionFrameRateEnable", "TriggerMode"]
        assert ("PixelFormat", "Mono12") in camera.history
        assert ("ExposureTime", 250.0) in camera.history
        assert ("AcquisitionFrameRate", 3.0) in camera.history
        assert camera.history[-1] == ("DeviceLinkThroughputLimit", 100000000)

        # Only the newest complete frame is published; every frame is requeued.
        camera.deliver(np.full((3, 4), 1, dtype=np.uint8))
        camera.deliver(np.full((3, 4), 2, dtype=np.uint8))
        camera.deliver(np.full((3, 4), 9, dtype=np.uint8), _FrameStatus.Incomplete)
        assert len(camera.queued) == 3
        wfs.expose()
        np.testing.assert_array_equal(wfs._stream_object("wfs_raw").read(), 2)
        assert wfs.data.dtype == np.uint16
        with pytest.raises(TimeoutError):
            wfs.expose()

        wfs.set_exposure(100.0)
        assert camera.history[-1] == ("ExposureTime", 100.0)
        # A payload change stops streaming, applies, and restarts.
        wfs.set_bit_depth(8)
        assert camera.history[-1] == ("PixelFormat", "Mono8") and camera.streaming
    finally:
        wfs.close()
    assert not camera.entered and not camera.streaming and system.depth == 0
    # Every feature is back at its value at open.
    values = {name: feature.get() for name, feature in camera.features.items()}
    assert values["ExposureAuto"] == "Continuous" and values["TriggerMode"] == "On"
    assert values["AcquisitionFrameRateEnable"] is True
    assert values["AcquisitionFrameRate"] == 10.0 and values["ExposureTime"] == 1000.0
    assert values["DeviceLinkThroughputLimit"] == 450000000


def test_restore_on_close_can_be_disabled(vimba):
    camera = _Camera("1")
    _VmbSystem.instance.cameras = [camera]
    wfs = vimba.VimbaWFS(_wfs_conf(device_index=0, exposure=50.0, restore_on_close=False))
    wfs.close()
    assert camera.features["ExposureTime"].get() == 50.0


def test_open_errors_release_the_camera(vimba):
    system = _VmbSystem.instance
    camera = _Camera("1", height=5, width=6)
    system.cameras = [camera]
    with pytest.raises(RuntimeError, match=r"serial '2' \(found: 1\)"):
        vimba.VimbaWFS(_wfs_conf(serial="2"))
    assert system.depth == 0
    with pytest.raises(ValueError, match=r"\(height, width\) = \(5, 6\)"):
        vimba.VimbaWFS(_wfs_conf(serial="1", exposure=10.0))
    assert system.depth == 0 and not camera.entered and not camera.streaming
    assert camera.features["ExposureTime"].get() == 1000.0
    with pytest.raises(AttributeError, match="no feature 'NotAFeature'"):
        vimba.VimbaWFS(_wfs_conf(width=6, height=5, node_settings={"NotAFeature": 1}))


def test_wfs_frames_follow_camera_axes(vimba):
    height, width = ORIENTATION_FRAME_SHAPE
    camera = _Camera("1", height=height, width=width)
    _VmbSystem.instance.cameras = [camera]
    wfs = vimba.VimbaWFS(_wfs_conf(width=width, height=height))
    try:
        assert_wfs_follows_camera_axes(wfs, camera.deliver)
    finally:
        wfs.close()


def test_science_camera_frames_follow_camera_axes(vimba):
    height, width = ORIENTATION_FRAME_SHAPE
    camera = _Camera("1", height=height, width=width)
    _VmbSystem.instance.cameras = [camera]
    psf = vimba.VimbaScienceCamera(
        {
            "name": "psf",
            "width": width,
            "height": height,
            "dark_count": 1,
            "integration": 1,
            "functions": [],
        }
    )
    try:
        camera.deliver(hot_pixel_frame())
        psf.expose()
        short = np.asarray(psf._stream_object("psf_short").read())
        assert np.unravel_index(np.argmax(short), short.shape) == ORIENTATION_HOT_PIXEL
    finally:
        psf.close()


def test_two_cameras_share_the_vimba_system(vimba):
    first, second = _Camera("1"), _Camera("2")
    _VmbSystem.instance.cameras = [first, second]
    cti = "/opt/VimbaX/cti/VimbaUSBTL.cti"
    a = vimba.VimbaWFS(_wfs_conf(serial="1", cti_path=cti))
    try:
        b = vimba.VimbaScienceCamera(
            {
                "name": "psf",
                "width": 4,
                "height": 3,
                "dark_count": 1,
                "integration": 1,
                "functions": [],
                "serial": "2",
                "cti_path": cti,
            }
        )
        assert _VmbSystem.instance.depth == 2
        with pytest.raises(ValueError, match="one transport configuration per process"):
            vimba.VimbaWFS(_wfs_conf(serial="1", cti_path="/other/producer.cti"))
        b.close()
    finally:
        a.close()
    assert _VmbSystem.instance.depth == 0


def test_a_runtime_change_of_frame_shape_is_refused_and_undone(vimba):
    camera = _Camera("1")
    _VmbSystem.instance.cameras = [camera]
    wfs = vimba.VimbaWFS(_wfs_conf(serial="1"))
    try:
        with pytest.raises(ValueError, match="frame is"):
            wfs.set_binning(2)  # would halve the frame; the streams cannot follow
        assert camera.features["BinningHorizontal"].get() == 1
        assert camera.streaming
        wfs.set_roi([4, 3, 0, 0])  # same shape: accepted
        assert camera.features["Width"].get() == 4
    finally:
        wfs.close()
