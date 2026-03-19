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


def _specula_conf():
    return {
        "wfs": {"name": "wfs", "width": 8, "height": 8, "darkCount": 1, "functions": []},
        "wfc": {"name": "wfc", "numActuators": 4, "numModes": 4, "functions": []},
    }


def _specula_param():
    return {
        "speculaInit": {"device_idx": -1, "precision": 1},
        "main": {"pixel_pupil": 16, "pixel_pitch": 0.05, "time_step": 0.001},
        "source": {"polar_coordinates": [0.0, 0.0], "magnitude": 8, "wavelengthInNm": 750},
        "signals": {"seeing": 0.8, "wind_speed": [0.0], "wind_direction": [0.0]},
        "pupilstop": {},
        "atmo": {"L0": [25.0], "heights": [0.0], "Cn2": [1.0], "pixel_phasescreens": 32},
        "propagation": {"wavelengthInNm": 750},
        "pyramid": {"wavelengthInNm": 750, "fov": 2.0, "pup_diam": 4, "output_resolution": 8, "mod_amp": 0.0},
        "detector": {"size": [8, 8], "dt": 0.001, "bandw": 300},
        "dm": {"height": 0.0, "type_str": "zonal", "geom": "square", "n_act": 2, "obsratio": 0.0},
        "basis": {"type_str": "zernike", "obsratio": 0.0},
    }


def _install_fake_specula(monkeypatch):
    from testsupport import DummySHM

    fake_wfs_module = importlib.import_module("pyRTC.WavefrontSensor")
    fake_wfc_module = importlib.import_module("pyRTC.WavefrontCorrector")
    monkeypatch.setattr(fake_wfs_module, "ImageSHM", DummySHM)
    monkeypatch.setattr(fake_wfc_module, "ImageSHM", DummySHM)

    init_calls = []

    class _Input:
        def __init__(self):
            self.value = None

        def set(self, value):
            self.value = value

        def get(self, target_device_idx=None):
            return self.value

    class _BaseValue:
        def __init__(self, value=None, target_device_idx=None, precision=None):
            self.value = None if value is None else np.asarray(value, dtype=np.float32)
            self.generation_time = -1

        def set_value(self, value):
            self.value = np.asarray(value, dtype=np.float32)

    class _SimulParams:
        def __init__(self, pixel_pupil, pixel_pitch, root_dir=".", total_time=0.01, time_step=0.001):
            self.pixel_pupil = pixel_pupil
            self.pixel_pitch = pixel_pitch
            self.root_dir = root_dir
            self.total_time = total_time
            self.time_step = time_step

    class _Source:
        def __init__(self, polar_coordinates, magnitude, wavelengthInNm, target_device_idx=None, precision=None):
            self.polar_coordinates = polar_coordinates
            self.magnitude = magnitude
            self.wavelengthInNm = wavelengthInNm

    class _Layer:
        def __init__(self):
            self.generation_time = 0
            self.command = np.zeros(4, dtype=np.float32)
            self.shiftXYinPixel = (0.0, 0.0)
            self.magnification = 1.0
            self.size = (4, 4)

    class _Pupilstop(_Layer):
        def __init__(self, simul_params, target_device_idx=None, precision=None, **kwargs):
            super().__init__()
            self.A = np.ones((simul_params.pixel_pupil, simul_params.pixel_pupil), dtype=np.float32)

    class _AtmoEvolution:
        def __init__(self, simul_params, target_device_idx=None, precision=None, **kwargs):
            self.inputs = {"seeing": _Input(), "wind_speed": _Input(), "wind_direction": _Input()}
            self.outputs = {"layer_list": [_Layer()]}
            self.inputs_changed = False

        def setup(self):
            return None

        def check_ready(self, t):
            self.inputs_changed = True
            self.outputs["layer_list"][0].generation_time = t
            return True

        def trigger(self):
            return None

        def post_trigger(self):
            return None

    class _DM:
        def __init__(self, simul_params, n_act=2, type_str="zonal", target_device_idx=None, precision=None, **kwargs):
            ifunc = kwargs.get("ifunc")
            if ifunc is not None and hasattr(ifunc, "influence_function"):
                influence_function = np.asarray(ifunc.influence_function, dtype=np.float32)
                mask_inf_func = np.asarray(ifunc.mask_inf_func, dtype=np.float32)
                self.nmodes = int(influence_function.shape[0])
            else:
                self.nmodes = n_act * n_act
                influence_function = np.eye(self.nmodes, dtype=np.float32)
                mask_inf_func = np.ones((n_act, n_act), dtype=np.float32)
            self.type_str = type_str
            self.inputs = {"in_command": _Input()}
            self.outputs = {"out_layer": _Layer()}
            self.inputs_changed = False
            self.ifunc_obj = types.SimpleNamespace(
                influence_function=influence_function,
                mask_inf_func=mask_inf_func,
            )

        def setup(self):
            return None

        def check_ready(self, t):
            self.inputs_changed = True
            return True

        def trigger(self):
            command = self.inputs["in_command"].get().value
            self.outputs["out_layer"].command = np.asarray(command, dtype=np.float32)
            self.outputs["out_layer"].generation_time = self.inputs["in_command"].get().generation_time

        def post_trigger(self):
            return None

    class _ElectricField:
        def __init__(self):
            self.generation_time = 0
            self.command_sum = 0.0

    class _AtmoPropagation:
        def __init__(self, simul_params, source_dict, target_device_idx=None, precision=None, **kwargs):
            self.inputs = {"atmo_layer_list": _Input(), "common_layer_list": _Input()}
            self.outputs = {"out_on_axis_source_ef": _ElectricField()}
            self.inputs_changed = False
            self.local_inputs = {}
            self.doFresnel = False

        def setup(self):
            return None

        def get_all_inputs(self):
            self.local_inputs = {
                "atmo_layer_list": self.inputs["atmo_layer_list"].get(),
                "common_layer_list": self.inputs["common_layer_list"].get(),
            }

        def setup_interpolators(self):
            return None

        def doFresnel_setup(self):
            return None

        def check_ready(self, t):
            self.inputs_changed = True
            return True

        def trigger(self):
            common_layers = self.inputs["common_layer_list"].get()
            dm_layer = common_layers[-1]
            self.outputs["out_on_axis_source_ef"].command_sum = float(np.sum(dm_layer.command))
            self.outputs["out_on_axis_source_ef"].generation_time = dm_layer.generation_time

        def post_trigger(self):
            return None

    class _Intensity:
        def __init__(self, size):
            self.i = np.zeros(size, dtype=np.float32)
            self.generation_time = 0

    class _ModulatedPyramid:
        def __init__(self, simul_params, output_resolution=8, target_device_idx=None, precision=None, **kwargs):
            self.inputs = {"in_ef": _Input()}
            self.outputs = {"out_i": _Intensity((output_resolution, output_resolution))}
            self.inputs_changed = False

        def setup(self):
            return None

        def check_ready(self, t):
            self.inputs_changed = True
            return True

        def trigger(self):
            ef = self.inputs["in_ef"].get()
            self.outputs["out_i"].i[:] = ef.command_sum
            self.outputs["out_i"].generation_time = ef.generation_time

        def post_trigger(self):
            return None

    class _Pixels:
        def __init__(self, size):
            self.pixels = np.zeros(size, dtype=np.float32)
            self.generation_time = 0

    class _CCD:
        def __init__(self, simul_params, size, dt, bandw, target_device_idx=None, precision=None, **kwargs):
            self.inputs = {"in_i": _Input()}
            self.outputs = {"out_pixels": _Pixels(tuple(size))}
            self.inputs_changed = False

        def setup(self):
            return None

        def check_ready(self, t):
            self.inputs_changed = True
            return True

        def trigger(self):
            intensity = self.inputs["in_i"].get()
            self.outputs["out_pixels"].pixels[:] = intensity.i
            self.outputs["out_pixels"].generation_time = intensity.generation_time

        def post_trigger(self):
            return None

    class _PSF:
        def __init__(self, simul_params, target_device_idx=None, precision=None, **kwargs):
            self.inputs = {"in_ef": _Input()}
            self.outputs = {
                "out_psf": types.SimpleNamespace(value=np.ones((2, 2), dtype=np.float32)),
                "out_sr": types.SimpleNamespace(value=np.array([1.0], dtype=np.float32)),
            }
            self.ref = types.SimpleNamespace(i=np.ones((2, 2), dtype=np.float32))
            self.inputs_changed = False
            self.verbose = kwargs.get("verbose", True)

        def setup(self):
            return None

        def check_ready(self, t):
            self.inputs_changed = True
            return True

        def trigger(self):
            return None

        def post_trigger(self):
            return None

    fake_specula = types.ModuleType("specula")
    fake_specula.init = lambda device_idx=-1, precision=1: init_calls.append((device_idx, precision))
    fake_specula.cpuArray = lambda arr: np.asarray(arr)

    class _IFunc:
        def __init__(self, type_str, mask, npixels, nmodes=None, n_act=None, obsratio=0.0, diaratio=1.0, target_device_idx=None, precision=None, **kwargs):
            self.type_str = type_str
            self.mask_inf_func = np.asarray(mask, dtype=np.float32)
            pixel_count = int(np.count_nonzero(self.mask_inf_func))
            if nmodes is None:
                if n_act is None:
                    raise TypeError("fake IFunc requires nmodes or n_act")
                nmodes = int(n_act) * int(n_act)
            self.influence_function = np.zeros((nmodes, pixel_count), dtype=np.float32)
            for index in range(nmodes):
                self.influence_function[index, index % pixel_count] = 1.0

    monkeypatch.setitem(sys.modules, "specula", fake_specula)
    monkeypatch.setitem(sys.modules, "specula.base_value", types.SimpleNamespace(BaseValue=_BaseValue))
    monkeypatch.setitem(sys.modules, "specula.data_objects.ifunc", types.SimpleNamespace(IFunc=_IFunc))
    monkeypatch.setitem(sys.modules, "specula.data_objects.simul_params", types.SimpleNamespace(SimulParams=_SimulParams))
    monkeypatch.setitem(sys.modules, "specula.data_objects.source", types.SimpleNamespace(Source=_Source))
    monkeypatch.setitem(sys.modules, "specula.data_objects.pupilstop", types.SimpleNamespace(Pupilstop=_Pupilstop))
    monkeypatch.setitem(sys.modules, "specula.processing_objects.atmo_evolution", types.SimpleNamespace(AtmoEvolution=_AtmoEvolution))
    monkeypatch.setitem(sys.modules, "specula.processing_objects.atmo_propagation", types.SimpleNamespace(AtmoPropagation=_AtmoPropagation))
    monkeypatch.setitem(sys.modules, "specula.processing_objects.modulated_pyramid", types.SimpleNamespace(ModulatedPyramid=_ModulatedPyramid))
    monkeypatch.setitem(sys.modules, "specula.processing_objects.ccd", types.SimpleNamespace(CCD=_CCD))
    monkeypatch.setitem(sys.modules, "specula.processing_objects.dm", types.SimpleNamespace(DM=_DM))
    monkeypatch.setitem(sys.modules, "specula.processing_objects.psf", types.SimpleNamespace(PSF=_PSF))

    return init_calls


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


def test_specula_interface_requires_optional_dependency(monkeypatch):
    sys.modules.pop("specula", None)
    monkeypatch.setitem(sys.modules, "specula", None)
    sys.modules.pop("pyRTC.hardware.SPECULAInterface", None)
    module = importlib.import_module("pyRTC.hardware.SPECULAInterface")

    with pytest.raises(ImportError, match="SPECULA support requires the optional 'specula' package"):
        module.SPECULAInterface(_specula_conf(), param=_specula_param())


def test_expected_output_specs_sync_specula_pywfs_geometry():
    pipeline = importlib.import_module("pyRTC.Pipeline")

    system_conf = {
        "specula": {
            "className": "pyRTC.hardware.SPECULAInterface.SPECULAInterface",
            "param": {
                "main": {"pixel_pupil": 120, "pixel_pitch": 0.05},
                "dm": {"type_str": "zonal", "geom": "square", "n_act": 4, "obsratio": 0.0},
                "pyramid": {
                    "wavelengthInNm": 750,
                    "fov": 2.0,
                    "pup_diam": 12,
                    "pup_dist": 16,
                    "output_resolution": 40,
                    "mod_amp": 0.0,
                },
                "detector": {"size": [80, 80], "dt": 0.001, "bandw": 300},
                "psf": {"wavelengthInNm": 1650, "nd": 4.0},
            },
        },
        "wfs": {
            "name": "wfs",
            "width": 1,
            "height": 1,
            "darkCount": 1,
            "functions": [],
            "resource": "specula",
            "className": "pyRTC.hardware.SPECULAInterface.SPECULAWFSensor",
        },
        "slopes": {
            "type": "PYWFS",
            "signalType": "slopes",
        },
        "wfc": {"name": "wfc", "numActuators": 4, "numModes": 4, "functions": []},
        "psf": {"name": "psf", "width": 1, "height": 1, "darkCount": 1, "integration": 1, "functions": []},
    }

    specs = pipeline.expected_output_shm_specs_for_config(system_conf)

    assert system_conf["wfs"]["width"] == 80
    assert system_conf["wfs"]["height"] == 80
    assert system_conf["slopes"]["pupils"] == ["24,24", "24,56", "56,24", "56,56"]
    assert system_conf["slopes"]["pupilsRadius"] == 12
    assert system_conf["wfc"]["displayGridSize"] == 4
    assert system_conf["psf"]["width"] == 480
    assert system_conf["psf"]["height"] == 480
    assert specs["wfsRaw"]["shape"] == (80, 80)
    assert specs["wfs"]["shape"] == (80, 80)
    assert specs["signal2D"]["shape"] == (24, 48)
    assert specs["wfc2D"]["shape"] == (4, 4)
    assert specs["psfShort"]["shape"] == (480, 480)


def test_specula_standalone_bridge_syncs_pywfs_geometry(monkeypatch):
    _install_fake_specula(monkeypatch)

    sys.modules.pop("pyRTC.hardware.SPECULAInterface", None)
    module = importlib.import_module("pyRTC.hardware.SPECULAInterface")

    conf = {
        "wfs": {"name": "wfs", "width": 1, "height": 1, "darkCount": 1, "functions": []},
        "slopes": {"type": "PYWFS", "signalType": "slopes"},
        "wfc": {"name": "wfc", "numActuators": 4, "numModes": 4, "functions": []},
        "psf": {"name": "psf", "width": 1, "height": 1, "darkCount": 1, "integration": 1, "functions": []},
    }
    param = _specula_param()
    param["pyramid"].update({"pup_diam": 6, "pup_dist": 8, "output_resolution": 12})
    param["detector"]["size"] = [12, 12]
    param["psf"] = {"wavelengthInNm": 1650, "nd": 1.0}
    param["psf"]["nd"] = 3.0

    sim = module.SPECULAInterface(conf, param=param)
    wfs, dm, _psf = sim.get_hardware()

    assert conf["wfs"]["width"] == 12
    assert conf["wfs"]["height"] == 12
    assert conf["slopes"]["pupils"] == ["2,2", "2,10", "10,2", "10,10"]
    assert conf["slopes"]["pupilsRadius"] == 3
    assert conf["wfc"]["displayGridSize"] == 2
    assert conf["psf"]["width"] == 48
    assert conf["psf"]["height"] == 48
    assert dm.correctionVector2D.arr.shape == (2, 2)
    assert sim.context.psf.verbose is False

    wfs.expose()
    assert wfs.read(block=False).shape == (12, 12)


def test_specula_square_dm_zeroes_m2c_outside_circular_support(monkeypatch):
    _install_fake_specula(monkeypatch)

    sys.modules.pop("pyRTC.hardware.SPECULAInterface", None)
    module = importlib.import_module("pyRTC.hardware.SPECULAInterface")

    conf = {
        "wfs": {"name": "wfs", "width": 1, "height": 1, "darkCount": 1, "functions": []},
        "slopes": {"type": "PYWFS", "signalType": "slopes"},
        "wfc": {"name": "wfc", "numActuators": 25, "numModes": 4, "functions": []},
    }
    param = _specula_param()
    param["dm"].update({"geom": "square", "circ_geom": False, "n_act": 5, "obsratio": 0.0})
    param["basis"]["obsratio"] = 0.0

    sim = module.SPECULAInterface(conf, param=param)
    _wfs, dm, _psf = sim.get_hardware()

    support = module._square_actuator_support_mask(5, 0.0).reshape(-1)
    assert dm.M2C.shape[0] == support.size
    assert np.all(dm.M2C[~support] == 0.0)
    assert np.any(dm.M2C[support] != 0.0)


def test_specula_circular_dm_uses_circular_display_diameter(monkeypatch):
    _install_fake_specula(monkeypatch)

    sys.modules.pop("pyRTC.hardware.SPECULAInterface", None)
    module = importlib.import_module("pyRTC.hardware.SPECULAInterface")

    conf = {
        "wfs": {"name": "wfs", "width": 1, "height": 1, "darkCount": 1, "functions": []},
        "slopes": {"type": "PYWFS", "signalType": "slopes"},
        "wfc": {"name": "wfc", "numActuators": 7, "numModes": 4, "functions": []},
    }
    param = _specula_param()
    param["dm"] = {"height": 0.0, "type_str": "zonal", "n_act": 2, "circ_geom": True, "obsratio": 0.0}

    assert module.derive_specula_wfc_display_geometry(param)["displayGridSize"] == 3
    layout, rows, cols = module._circular_zonal_display_mapping(2, 0.0)
    assert layout.shape == (3, 3)
    assert len(rows) == 7
    assert len(cols) == 7

    larger_layout, larger_rows, larger_cols = module._circular_zonal_display_mapping(8, 0.0)
    assert larger_layout.shape == (12, 12)
    assert len(set(zip(larger_rows.tolist(), larger_cols.tolist()))) == 61


def test_specula_interface_updates_live_atmosphere_controls(monkeypatch):
    _install_fake_specula(monkeypatch)

    sys.modules.pop("pyRTC.hardware.SPECULAInterface", None)
    module = importlib.import_module("pyRTC.hardware.SPECULAInterface")

    sim = module.SPECULAInterface(_specula_conf(), param=_specula_param())

    old_atmo = sim.context.atmo
    sim.seeing = 0.35
    sim.wind_speed = [7.5]
    sim.wind_direction = [45.0]
    sim.atmo_L0 = [30.0]
    sim.useAtmosphere = False

    assert sim.seeing == pytest.approx(0.35)
    assert sim.wind_speed == [7.5]
    assert sim.wind_direction == [45.0]
    assert sim.atmo_L0 == [30.0]
    assert sim.context.atmo is not old_atmo
    assert sim.useAtmosphere is False
    assert sim.context.atmosphere_enabled is False


def test_specula_standalone_bridge_exposes_frames_and_updates_dm(monkeypatch):
    init_calls = _install_fake_specula(monkeypatch)

    sys.modules.pop("pyRTC.hardware.SPECULAInterface", None)
    module = importlib.import_module("pyRTC.hardware.SPECULAInterface")

    sim = module.SPECULAInterface(_specula_conf(), param=_specula_param())
    wfs, dm, _psf = sim.get_hardware()

    wfs.expose()
    initial = wfs.read(block=False)

    dm.write(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    dm.sendToHardware()
    wfs.expose()
    updated = wfs.read(block=False)

    assert init_calls == [(-1, 1)]
    assert initial.shape == (8, 8)
    assert updated.shape == (8, 8)
    assert np.any(updated != initial)
    assert np.isclose(sim.context.command.value[0], 1.0)
    assert dm.correctionVector2D is not None
    assert int(np.count_nonzero(dm.layout)) == dm.numActuators
    assert dm.M2C.shape == (dm.numActuators, dm.numModes)


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

    assert sim.context.ngs.optBand == "R"
    assert sim.context.ngs.magnitude == 9
    assert sim.context.src.optBand == "K"
    assert sim.context.src.magnitude == 5
    assert sim.context.tel is not sim.context.tel_psf
    assert sim.context.atm.initialized_with is sim.context.tel
    assert sim.context.dm.kwargs["nSubap"] == 10
    assert sim.context.dm.kwargs["altitude"] is None
    assert sim.context.wfs.kwargs["lightRatio"] == 0.1
    assert sim.context.tel.kwargs["centralObstruction"] == 0.112
    assert sim.context.tel_psf.kwargs["centralObstruction"] == 0.112


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

    assert sim.context.tel.kwargs["centralObstruction"] is None
    assert sim.context.tel_psf.kwargs["centralObstruction"] is None


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

    assert sim.context.tel is tel
    assert sim.context.tel_psf is tel_psf
    assert sim.context.ngs is ngs
    assert sim.context.src is src
    assert sim.context.atm is atm
    assert sim.context.dm is dm
    assert sim.context.wfs is wfs


def test_oopao_interface_requires_param_or_objects(monkeypatch):
    module, _, _, _, _, _ = _install_fake_oopao(monkeypatch)

    with pytest.raises(ValueError, match="requires either param=<mapping> or paramFile=<YAML path>"):
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

    fake_context = types.SimpleNamespace(
        tel=_FakeTelescope(),
        ngs=_FakeSource(),
        atm=_FakeAtmosphere(),
        dm=_FakeDM(),
        wfs=_FakePyramid(),
        register_component=lambda *args, **kwargs: None,
    )
    sensor = module._OOPAOWFSensor(
        {"name": "wfs", "width": 2, "height": 2, "darkCount": 1, "functions": []},
        fake_context,
    )

    sensor.expose()
    first = sensor.data.copy()
    sensor.expose()
    second = sensor.data.copy()

    assert np.array_equal(first, second)
    assert np.all(first == 8)