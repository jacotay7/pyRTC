import importlib
import sys
import types

import numpy as np
import pytest


def test_ximea_wfs_init_and_controls(monkeypatch):
    from testsupport import DummySHM

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
    from testsupport import DummySHM

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
    from testsupport import DummySHM

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


@pytest.mark.parametrize(
    ("symbol_name", "module_name", "missing_dependency"),
    [
        ("XIMEA_WFS", ".ximeaWFS", "ximea"),
        ("spinCam", ".SpinnakerScienceCam", "rotpy"),
        ("ALPAODM", ".ALPAODM", "Lib64.asdk"),
        ("PIModulator", ".PIModulator", "pipython"),
    ],
)
def test_hardware_lazy_import_reports_missing_dependency(monkeypatch, symbol_name, module_name, missing_dependency):
    hardware = importlib.import_module("pyRTC.hardware")
    hardware.__dict__.pop(symbol_name, None)

    original_import_module = hardware.import_module

    def _fake_import_module(requested_module_name, package=None):
        if requested_module_name == module_name and package == hardware.__name__:
            raise ModuleNotFoundError(f"No module named '{missing_dependency}'")
        return original_import_module(requested_module_name, package)

    monkeypatch.setattr(hardware, "import_module", _fake_import_module)

    with pytest.raises(ImportError) as excinfo:
        getattr(hardware, symbol_name)

    message = str(excinfo.value)
    assert f"Unable to import pyRTC.hardware.{symbol_name}" in message
    assert missing_dependency in message


def _oopao_conf():
    return {
        "wfs": {"name": "wfs", "width": 2, "height": 2, "darkCount": 1, "functions": []},
        "wfc": {"name": "wfc", "numActuators": 4, "numModes": 4, "functions": []},
        "psf": {"name": "psf", "width": 2, "height": 2, "darkCount": 1, "integration": 1, "functions": []},
    }


def _install_fake_oopao(monkeypatch):
    from testsupport import DummySHM

    fake_wfs_module = importlib.import_module("pyRTC.WavefrontSensor")
    fake_science_module = importlib.import_module("pyRTC.ScienceCamera")
    fake_wfc_module = importlib.import_module("pyRTC.WavefrontCorrector")
    monkeypatch.setattr(fake_wfs_module, "ImageSHM", DummySHM)
    monkeypatch.setattr(fake_science_module, "ImageSHM", DummySHM)
    monkeypatch.setattr(fake_wfc_module, "ImageSHM", DummySHM)

    class _FakeSource:
        def __init__(self, optBand="I", magnitude=0, **kwargs):
            self.optBand = optBand
            self.magnitude = magnitude
            self.kwargs = kwargs
            self.mask = 1
            self.OPD = None
            self.OPD_no_pupil = None
            self.optical_path = []

        def __mul__(self, obj):
            obj.relay(self)
            return self

        def __pow__(self, obj):
            if hasattr(obj, "src"):
                obj.src = self
            self.optical_path = []
            self.reset()
            self * obj
            return self

        def reset(self):
            self.mask = 1
            self.OPD = None
            self.OPD_no_pupil = None

    class _FakeTelescope:
        def __init__(self, resolution, diameter, samplingTime, centralObstruction=None, **kwargs):
            self.kwargs = {
                "resolution": resolution,
                "diameter": diameter,
                "samplingTime": samplingTime,
                "centralObstruction": centralObstruction,
                **kwargs,
            }
            self.tag = "telescope"
            self.isPaired = False
            self.pupil = np.ones((2, 2), dtype=bool)
            self.src = None
            self.PSF = np.ones((2, 2), dtype=np.float64)

        def relay(self, src):
            self.src = src
            src.mask = self.pupil.copy()
            if src.OPD_no_pupil is None:
                src.OPD_no_pupil = np.zeros(self.pupil.shape, dtype=np.float64)
            src.OPD = src.OPD_no_pupil * src.mask

        def computePSF(self, zeroPaddingFactor=5):
            level = 1.0
            if self.src is not None and getattr(self.src, "OPD_no_pupil", None) is not None:
                level += float(np.sum(self.src.OPD_no_pupil))
            self.PSF = np.full((2, 2), level, dtype=np.float64)

        def __add__(self, atm):
            self.isPaired = True
            return self

        def __sub__(self, atm):
            self.isPaired = False
            return self

    class _FakeDM:
        def __init__(self, telescope, nSubap, mechCoupling, altitude=None, **kwargs):
            self.telescope = telescope
            self.kwargs = {"nSubap": nSubap, "mechCoupling": mechCoupling, "altitude": altitude, **kwargs}
            self.tag = "dm"
            self.validAct = np.ones((4,), dtype=bool)
            self.OPD = np.zeros((2, 2), dtype=np.float64)
            self.coefs = 0
            self.flat = None

        def relay(self, src):
            src.OPD_no_pupil = src.OPD_no_pupil + self.OPD
            src.OPD = src.OPD_no_pupil * src.mask

    class _FakeAtmosphere:
        def __init__(self, telescope, r0, L0, windSpeed, fractionalR0, windDirection, altitude, **kwargs):
            self.telescope = telescope
            self.kwargs = {
                "r0": r0,
                "L0": L0,
                "windSpeed": windSpeed,
                "fractionalR0": fractionalR0,
                "windDirection": windDirection,
                "altitude": altitude,
                **kwargs,
            }
            self.tag = "atmosphere"
            self.OPD_no_pupil = np.zeros((2, 2), dtype=np.float64)
            self.initialized_with = None

        def initializeAtmosphere(self, telescope):
            self.initialized_with = telescope

        def update(self):
            return None

    class _FakePyramid:
        def __init__(self, nSubap, telescope, modulation, lightRatio, n_pix_separation, psfCentering, postProcessing, **kwargs):
            self.telescope = telescope
            self.kwargs = {
                "nSubap": nSubap,
                "modulation": modulation,
                "lightRatio": lightRatio,
                "n_pix_separation": n_pix_separation,
                "psfCentering": psfCentering,
                "postProcessing": postProcessing,
                **kwargs,
            }
            self.tag = "wfs"
            self.cam = types.SimpleNamespace(frame=np.zeros((2, 2), dtype=np.float32))

        def relay(self, src):
            level = float(np.sum(src.OPD_no_pupil))
            self.cam.frame = np.full((2, 2), level, dtype=np.float32)

    fake_oopao_pkg = types.ModuleType("OOPAO")
    monkeypatch.setitem(sys.modules, "OOPAO", fake_oopao_pkg)
    monkeypatch.setitem(sys.modules, "OOPAO.Atmosphere", types.SimpleNamespace(Atmosphere=_FakeAtmosphere))
    monkeypatch.setitem(sys.modules, "OOPAO.DeformableMirror", types.SimpleNamespace(DeformableMirror=_FakeDM))
    monkeypatch.setitem(sys.modules, "OOPAO.Pyramid", types.SimpleNamespace(Pyramid=_FakePyramid))
    monkeypatch.setitem(sys.modules, "OOPAO.Source", types.SimpleNamespace(Source=_FakeSource))
    monkeypatch.setitem(sys.modules, "OOPAO.Telescope", types.SimpleNamespace(Telescope=_FakeTelescope))

    sys.modules.pop("pyRTC.hardware.OOPAOInterface", None)
    module = importlib.import_module("pyRTC.hardware.OOPAOInterface")
    return module, _FakeTelescope, _FakeSource, _FakeAtmosphere, _FakeDM, _FakePyramid


def test_oopao_interface_builds_from_param_dict_and_keeps_source_overrides(monkeypatch):
    module, _, _, _, _, _ = _install_fake_oopao(monkeypatch)

    param = {
        "resolution": 40,
        "diameter": 8,
        "samplingTime": 0.001,
        "centralObstruction": 0.112,
        "ngs_band": "R",
        "ngs_magnitude": 9,
        "science_band": "K",
        "science_magnitude": 5,
        "r0": 0.3,
        "L0": 30,
        "fractionalR0": [0.45, 0.1, 0.1, 0.25, 0.1],
        "windSpeed": [10, 12, 11, 15, 20],
        "windDirection": [0, 72, 144, 216, 288],
        "altitude": [0, 1000, 5000, 10000, 12000],
        "nSubap": 10,
        "mechCoupling": 0.45,
        "modulation": 5,
        "lightRatio": 0.1,
        "n_pix_separation": 4,
        "psfCentering": False,
        "postProcessing": "slopesMaps",
    }

    sim = module.OOPAOInterface(_oopao_conf(), param=param)

    assert sim.ngs.optBand == "R"
    assert sim.ngs.magnitude == 9
    assert sim.src.optBand == "K"
    assert sim.src.magnitude == 5
    assert sim.tel is not sim.tel_psf
    assert sim.atm.initialized_with is sim.tel
    assert sim.dm.kwargs["nSubap"] == 10
    assert sim.dm.kwargs["altitude"] is None
    assert sim.wfs.kwargs["lightRatio"] == 0.1
    assert sim.tel.kwargs["centralObstruction"] == 0.112
    assert sim.tel_psf.kwargs["centralObstruction"] == 0.112


def test_oopao_interface_only_passes_optional_object_kwargs_when_present(monkeypatch):
    module, _, _, _, _, _ = _install_fake_oopao(monkeypatch)

    param = {
        "resolution": 40,
        "diameter": 8,
        "samplingTime": 0.001,
        "ngs_band": "R",
        "ngs_magnitude": 9,
        "science_band": "K",
        "science_magnitude": 5,
        "r0": 0.3,
        "L0": 30,
        "fractionalR0": [0.45, 0.1, 0.1, 0.25, 0.1],
        "windSpeed": [10, 12, 11, 15, 20],
        "windDirection": [0, 72, 144, 216, 288],
        "altitude": [0, 1000, 5000, 10000, 12000],
        "nSubap": 10,
        "mechCoupling": 0.45,
        "modulation": 5,
        "lightRatio": 0.1,
        "n_pix_separation": 4,
        "psfCentering": False,
        "postProcessing": "slopesMaps",
    }

    sim = module.OOPAOInterface(_oopao_conf(), param=param)

    assert sim.tel.kwargs["centralObstruction"] is None
    assert sim.tel_psf.kwargs["centralObstruction"] is None


def test_oopao_interface_accepts_prebuilt_objects(monkeypatch):
    module, FakeTelescope, FakeSource, FakeAtmosphere, FakeDM, FakePyramid = _install_fake_oopao(monkeypatch)

    tel = FakeTelescope(resolution=40, diameter=8, samplingTime=0.001, centralObstruction=0.112)
    tel_psf = FakeTelescope(resolution=40, diameter=8, samplingTime=0.001, centralObstruction=0.112)
    ngs = FakeSource(optBand="I", magnitude=8)
    src = FakeSource(optBand="K", magnitude=4)
    atm = FakeAtmosphere(
        telescope=tel,
        r0=0.3,
        L0=30,
        windSpeed=[10, 12, 11, 15, 20],
        fractionalR0=[0.45, 0.1, 0.1, 0.25, 0.1],
        windDirection=[0, 72, 144, 216, 288],
        altitude=[0, 1000, 5000, 10000, 12000],
    )
    dm = FakeDM(telescope=tel, nSubap=10, mechCoupling=0.45)
    wfs = FakePyramid(
        nSubap=10,
        telescope=tel,
        modulation=5,
        lightRatio=0.1,
        n_pix_separation=4,
        psfCentering=False,
        postProcessing="slopesMaps",
    )

    sim = module.OOPAOInterface(
        _oopao_conf(),
        tel=tel,
        tel_psf=tel_psf,
        ngs=ngs,
        src=src,
        atm=atm,
        dm=dm,
        wfs=wfs,
    )

    assert sim.tel is tel
    assert sim.tel_psf is tel_psf
    assert sim.ngs is ngs
    assert sim.src is src
    assert sim.atm is atm
    assert sim.dm is dm
    assert sim.wfs is wfs


def test_oopao_interface_requires_param_or_objects(monkeypatch):
    module, _, _, _, _, _ = _install_fake_oopao(monkeypatch)

    with pytest.raises(ValueError, match="embedded default parameter payload"):
        module.OOPAOInterface(_oopao_conf())


def test_oopao_wfs_static_dm_does_not_accumulate_without_atmosphere(monkeypatch):
    from testsupport import DummySHM

    fake_wfs_module = importlib.import_module("pyRTC.WavefrontSensor")
    monkeypatch.setattr(fake_wfs_module, "ImageSHM", DummySHM)

    class _FakeSource:
        def __init__(self):
            self.tag = "source"
            self.mask = 1
            self.OPD = None
            self.OPD_no_pupil = None
            self.optical_path = []

        def __mul__(self, obj):
            obj.relay(self)
            return self

        def __pow__(self, obj):
            if hasattr(obj, "src"):
                obj.src = self
            self.optical_path = []
            self.reset()
            self * obj
            return self

        def reset(self):
            self.mask = 1
            self.OPD = None
            self.OPD_no_pupil = None

    class _FakeTelescope:
        def __init__(self, *args, **kwargs):
            self.tag = "telescope"
            self.isPaired = False
            self.pupil = np.ones((2, 2), dtype=bool)
            self.src = None

        def relay(self, src):
            self.src = src
            src.mask = self.pupil.copy()
            if src.OPD is None:
                src.OPD_no_pupil = np.zeros(self.pupil.shape, dtype=np.float64)
            src.OPD = src.OPD_no_pupil * src.mask

    class _FakeDM:
        def __init__(self, *args, **kwargs):
            self.tag = "dm"
            self.validAct = np.ones((4,), dtype=bool)
            self.OPD = np.ones((2, 2), dtype=np.float64) * 2.0
            self.coefs = 0

        def relay(self, src):
            src.OPD_no_pupil = src.OPD_no_pupil + self.OPD
            src.OPD = src.OPD_no_pupil * src.mask

    class _FakeAtmosphere:
        def __init__(self, *args, **kwargs):
            self.tag = "atmosphere"

        def initializeAtmosphere(self, telescope):
            return None

        def update(self):
            return None

    class _FakePyramid:
        def __init__(self, *args, **kwargs):
            self.tag = "wfs"
            self.cam = types.SimpleNamespace(frame=np.zeros((2, 2), dtype=np.float32))

        def relay(self, src):
            level = float(np.sum(src.OPD_no_pupil))
            self.cam.frame = np.full((2, 2), level, dtype=np.float32)

    fake_oopao_pkg = types.ModuleType("OOPAO")
    monkeypatch.setitem(sys.modules, "OOPAO", fake_oopao_pkg)
    monkeypatch.setitem(sys.modules, "OOPAO.Atmosphere", types.SimpleNamespace(Atmosphere=_FakeAtmosphere))
    monkeypatch.setitem(sys.modules, "OOPAO.DeformableMirror", types.SimpleNamespace(DeformableMirror=_FakeDM))
    monkeypatch.setitem(sys.modules, "OOPAO.Pyramid", types.SimpleNamespace(Pyramid=_FakePyramid))
    monkeypatch.setitem(sys.modules, "OOPAO.Source", types.SimpleNamespace(Source=_FakeSource))
    monkeypatch.setitem(sys.modules, "OOPAO.Telescope", types.SimpleNamespace(Telescope=_FakeTelescope))

    sys.modules.pop("pyRTC.hardware.OOPAOInterface", None)
    module = importlib.import_module("pyRTC.hardware.OOPAOInterface")

    sensor = module._OOPAOWFSensor(
        {"name": "wfs", "width": 2, "height": 2, "darkCount": 1, "functions": []},
        _FakeTelescope(),
        _FakeSource(),
        _FakeAtmosphere(),
        _FakeDM(),
        _FakePyramid(),
    )

    sensor.expose()
    first = sensor.data.copy()
    sensor.expose()
    second = sensor.data.copy()

    assert np.array_equal(first, second)
    assert np.all(first == 8)