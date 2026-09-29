"""GenICam adapters against a fake Harvesters API (#72)."""

import sys
import types

import numpy as np
import pytest

from testsupport import private_stream


class _Node:
    def __init__(self, value=None):
        self.value = value


class _NodeMap:
    NAMES = (
        "PixelFormat",
        "ExposureTime",
        "Gain",
        "BinningHorizontal",
        "BinningVertical",
        "Width",
        "Height",
        "OffsetX",
        "OffsetY",
        "AcquisitionFrameRate",
        "DeviceModelName",
    )

    def __init__(self):
        self.history = []
        for name in self.NAMES:
            object.__setattr__(self, name, _Recorder(self, name))
        self.DeviceModelName.value = "FakeCam"


class _Recorder(_Node):
    def __init__(self, node_map, name):
        self._map, self._name = node_map, name
        self._value = None

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value
        if self._name != "DeviceModelName":
            self._map.history.append((self._name, value))


class _Buffer:
    def __init__(self, frame):
        component = types.SimpleNamespace(
            data=frame.ravel(), width=frame.shape[1], height=frame.shape[0]
        )
        self.payload = types.SimpleNamespace(components=[component])

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Acquirer:
    def __init__(self, key):
        self.key = key
        self.remote_device = types.SimpleNamespace(node_map=_NodeMap())
        self.started = self.destroyed = False
        self.frame = np.arange(12, dtype=np.uint16).reshape(3, 4)

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def destroy(self):
        self.destroyed = True

    def fetch(self, timeout):
        self.last_timeout = timeout
        return _Buffer(self.frame)


class _Harvester:
    instances = []

    def __init__(self):
        self.files, self.updated, self.acquirers = [], False, []
        _Harvester.instances.append(self)

    def add_file(self, path):
        self.files.append(path)

    def update(self):
        self.updated = True

    def create(self, key):
        acquirer = _Acquirer(key)
        self.acquirers.append(acquirer)
        return acquirer

    def reset(self):
        self.was_reset = True


@pytest.fixture
def genicam(monkeypatch):
    harvesters = types.ModuleType("harvesters")
    core = types.ModuleType("harvesters.core")
    core.Harvester = _Harvester
    harvesters.core = core
    monkeypatch.setitem(sys.modules, "harvesters", harvesters)
    monkeypatch.setitem(sys.modules, "harvesters.core", core)
    _Harvester.instances.clear()

    import pyrtc.hardware.genicam_camera as module
    import pyrtc.science_camera as science_camera
    import pyrtc.wavefront_sensor as wavefront_sensor

    monkeypatch.setattr(wavefront_sensor, "create_stream", private_stream)
    monkeypatch.setattr(science_camera, "create_stream", private_stream)
    return module


def _wfs_conf(**extra):
    return {
        "name": "wfs",
        "width": 4,
        "height": 3,
        "functions": [],
        "cti_file": "/opt/vendor/producer.cti",
        **extra,
    }


def test_wfs_opens_configures_and_grabs(genicam):
    wfs = genicam.GenICamWFS(
        _wfs_conf(
            serial="1234",
            exposure=250.0,
            gain=2.0,
            bit_depth=12,
            binning=2,
            left=8,
            top=4,
            node_settings={"AcquisitionFrameRate": 500.0},
            fetch_timeout=0.2,
        )
    )
    try:
        harvester = _Harvester.instances[-1]
        acquirer = harvester.acquirers[-1]
        assert harvester.files == ["/opt/vendor/producer.cti"] and harvester.updated
        assert acquirer.key == {"serial_number": "1234"} and acquirer.started
        history = acquirer.remote_device.node_map.history
        assert ("PixelFormat", "Mono12") in history
        assert ("ExposureTime", 250.0) in history and ("Gain", 2.0) in history
        assert ("BinningHorizontal", 2) in history and ("BinningVertical", 2) in history
        # ROI: offsets reset first, then size, then the requested offsets.
        roi = [entry for entry in history if entry[0] in ("Width", "Height", "OffsetX", "OffsetY")]
        assert roi == [
            ("OffsetX", 0),
            ("OffsetY", 0),
            ("Width", 4),
            ("Height", 3),
            ("OffsetX", 8),
            ("OffsetY", 4),
        ]
        assert history[-1] == ("AcquisitionFrameRate", 500.0)  # node_settings last
        assert wfs.exposure == 250.0 and wfs.gain == 2.0

        wfs.expose()
        assert acquirer.last_timeout == 0.2
        # Camera frames are (Height, Width); pyrtc streams are (width, height).
        np.testing.assert_array_equal(wfs.data, acquirer.frame.T)
        np.testing.assert_array_equal(wfs._stream_object("wfs_raw").read(), acquirer.frame.T)

        wfs.set_exposure(100.0)
        assert history[-1] == ("ExposureTime", 100.0)
    finally:
        wfs.close()
    assert acquirer.destroyed and harvester.was_reset


def test_device_index_and_unknown_nodes(genicam):
    wfs = genicam.GenICamWFS(_wfs_conf(device_index=1))
    try:
        assert _Harvester.instances[-1].acquirers[-1].key == 1
        with pytest.raises(AttributeError, match="no node 'NotANode'"):
            wfs._set_node("NotANode", 1)
    finally:
        wfs.close()


def test_science_camera_grabs_frames(genicam):
    camera = genicam.GenICamScienceCamera(
        {
            "name": "psf",
            "width": 4,
            "height": 3,
            "dark_count": 1,
            "integration": 2,
            "functions": [],
            "cti_file": ["/a.cti", "/b.cti"],
        }
    )
    try:
        assert _Harvester.instances[-1].files == ["/a.cti", "/b.cti"]
        camera.expose()
        assert camera.data.shape == (4, 3) and camera.data.dtype == np.uint16
    finally:
        camera.close()


def test_producers_from_the_environment(genicam, tmp_path, monkeypatch):
    (tmp_path / "vendor.cti").write_text("")
    (tmp_path / "readme.txt").write_text("")
    monkeypatch.setenv("GENICAM_GENTL64_PATH", str(tmp_path))
    assert genicam._producer_files(None) == [str(tmp_path / "vendor.cti")]
    monkeypatch.setenv("GENICAM_GENTL64_PATH", "")
    with pytest.raises(RuntimeError, match="GenTL producer"):
        genicam._producer_files(None)


def test_missing_harvesters_names_the_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "harvesters", None)
    monkeypatch.setitem(sys.modules, "harvesters.core", None)
    import pyrtc.hardware.genicam_camera as module
    import pyrtc.wavefront_sensor as wavefront_sensor

    monkeypatch.setattr(wavefront_sensor, "create_stream", private_stream)
    with pytest.raises(ImportError, match=r"pyrtcao\[genicam\]"):
        module.GenICamWFS(_wfs_conf())
