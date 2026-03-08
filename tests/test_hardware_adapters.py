import importlib
import sys
import types

import numpy as np


def test_ximea_wfs_init_and_controls(monkeypatch):
    from conftest import DummySHM

    fake_wfs_module = importlib.import_module("pyRTC.WavefrontSensor")
    monkeypatch.setattr(fake_wfs_module, "ImageSHM", DummySHM)

    class _Camera:
        def __init__(self):
            self.params = {}
            self.opened = None
            self.started = False
            self.stopped = False
            self.closed = False

        def open_device_by(self, mode, serial):
            self.opened = (mode, serial)

        def set_param(self, key, value):
            self.params[key] = value

        def start_acquisition(self):
            self.started = True

        def stop_acquisition(self):
            self.stopped = True

        def close_device(self):
            self.closed = True

        def get_image(self, _img):
            return None

    class _Image:
        def get_image_data_numpy(self):
            return np.ones((8, 8), dtype=np.uint16)

    fake_xiapi = types.SimpleNamespace(Camera=_Camera, Image=_Image)
    fake_ximea = types.SimpleNamespace(xiapi=fake_xiapi)
    monkeypatch.setitem(sys.modules, "ximea", fake_ximea)

    sys.modules.pop("pyRTC.hardware.ximeaWFS", None)
    module = importlib.import_module("pyRTC.hardware.ximeaWFS")

    conf = {
        "name": "wfs",
        "serial": "123",
        "width": 8,
        "height": 8,
        "darkCount": 1,
        "functions": [],
        "bitDepth": 16,
        "binning": 2,
        "exposure": 100,
        "left": 1,
        "top": 2,
        "gain": 3,
    }

    cam = module.XIMEA_WFS(conf)
    assert cam.cam.opened == ("XI_OPEN_BY_SN", "123")
    assert cam.cam.started is True
    assert cam.cam.params["width"] == 8
    assert cam.cam.params["height"] == 8
    assert cam.cam.params["offsetX"] == 1
    assert cam.cam.params["offsetY"] == 2
    assert cam.cam.params["gain"] == 3
    cam.__del__()
    assert cam.cam.stopped is True
    assert cam.cam.closed is True


def test_spinnaker_science_camera_init_and_controls(monkeypatch):
    from conftest import DummySHM

    fake_science_module = importlib.import_module("pyRTC.ScienceCamera")
    monkeypatch.setattr(fake_science_module, "ImageSHM", DummySHM)

    class _Node:
        def __init__(self):
            self.value = None

        def set_node_value(self, value, verify=False):
            self.value = value

        def set_node_value_from_str(self, value, verify=False):
            self.value = value

    class _Camera:
        def __init__(self):
            self.camera_nodes = types.SimpleNamespace(
                ExposureAuto=_Node(),
                GainAuto=_Node(),
                OffsetX=_Node(),
                OffsetY=_Node(),
                Height=_Node(),
                Width=_Node(),
                ExposureTime=_Node(),
                Gain=_Node(),
                Gamma=_Node(),
                PixelFormat=_Node(),
            )
            self.started = False
            self.ended = False
            self.deinited = False
            self.released = False

        def init_cam(self):
            return None

        def begin_acquisition(self):
            self.started = True

        def end_acquisition(self):
            self.ended = True

        def deinit_cam(self):
            self.deinited = True

        def release(self):
            self.released = True

        def get_next_image(self, timeout=5):
            class _Image:
                def get_image_data(self_inner):
                    return np.ones((8, 8), dtype=np.uint16).tobytes()

            return _Image()

    class _CameraList:
        @staticmethod
        def create_from_system(system, update_cams=True, update_interfaces=True):
            class _List:
                def create_camera_by_index(self, index):
                    return _Camera()

            return _List()

    fake_rotpy_camera = types.SimpleNamespace(CameraList=_CameraList)
    fake_rotpy_system = types.SimpleNamespace(SpinSystem=lambda: object())
    monkeypatch.setitem(sys.modules, "rotpy.camera", fake_rotpy_camera)
    monkeypatch.setitem(sys.modules, "rotpy.system", fake_rotpy_system)

    sys.modules.pop("pyRTC.hardware.SpinnakerScienceCam", None)
    module = importlib.import_module("pyRTC.hardware.SpinnakerScienceCam")

    conf = {
        "name": "psf",
        "index": 0,
        "width": 8,
        "height": 8,
        "darkCount": 1,
        "integration": 1,
        "functions": [],
        "bitDepth": 16,
        "exposure": 10,
        "left": 1,
        "top": 2,
        "gain": 3,
        "gamma": 2.0,
    }

    cam = module.spinCam(conf)
    assert cam.camera.started is True
    assert cam.camera.camera_nodes.ExposureAuto.value == "Off"
    assert cam.camera.camera_nodes.GainAuto.value == "Off"
    assert cam.camera.camera_nodes.OffsetX.value == 1
    assert cam.camera.camera_nodes.OffsetY.value == 2
    assert cam.camera.camera_nodes.Gain.value == 3
    cam.__del__()
    assert cam.camera.ended is True
    assert cam.camera.deinited is True
    assert cam.camera.released is True


def test_alpao_dm_init_and_layout(monkeypatch, tmp_path):
    from conftest import DummySHM

    fake_wfc_module = importlib.import_module("pyRTC.WavefrontCorrector")
    monkeypatch.setattr(fake_wfc_module, "ImageSHM", DummySHM)

    class _DM:
        def __init__(self, serial):
            self.serial = serial
            self.sent = []
            self.reset = False

        def Get(self, key):
            assert key == "NBOfActuator"
            return 97

        def Send(self, shape):
            self.sent.append(np.asarray(shape))

        def Reset(self):
            self.reset = True

    monkeypatch.setitem(sys.modules, "Lib64.asdk", types.SimpleNamespace(DM=_DM))

    sys.modules.pop("pyRTC.hardware.ALPAODM", None)
    module = importlib.import_module("pyRTC.hardware.ALPAODM")

    floating_file = tmp_path / "floating.npy"
    np.save(floating_file, np.array([0, 2], dtype=np.int32))

    conf = {
        "name": "wfc",
        "serial": "BAX123",
        "numActuators": 97,
        "numModes": 4,
        "commandCap": 0.5,
        "floatingActuatorsFile": str(floating_file),
        "functions": [],
    }

    dm = module.ALPAODM(conf)
    assert dm.layout.shape == (11, 11)
    assert dm.dm.serial == "BAX123"
    assert dm.CAP == 0.5
    dm.__del__()
    assert dm.dm.reset is True