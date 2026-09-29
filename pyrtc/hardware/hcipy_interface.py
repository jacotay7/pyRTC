"""Bridge between pyrtc components and an HCIPy optical simulation.

`HCIPy <https://hcipy.org>`_ is a pip-installable AO and high-contrast imaging
simulator (``pip install pyrtcao[hcipy]``). This module builds one shared
HCIPy system (telescope pupil, deformable mirror, Shack-Hartmann or modulated
pyramid WFS, a frozen-flow atmospheric layer and a science camera) from a flat
parameter mapping, and adapts it to the pyrtc ``WavefrontSensor``,
``WavefrontCorrector`` and ``ScienceCamera`` interfaces, like the OOPAO and
SPECULA bridges.

Parameters (``param`` mapping or ``param_file`` YAML; defaults in brackets):

``telescope_diameter`` [2.0 m], ``central_obscuration`` [0.0, fraction of D],
``wfs_type`` [``shwfs``; or ``pywfs``], ``wfs_wavelength`` [700e-9 m],
``num_lenslets`` [10], ``pixels_per_lenslet`` [12] (the pupil grid is
``num_lenslets * pixels_per_lenslet`` pixels across, so each sub-aperture is a
whole number of pixels), ``spot_fwhm_pixels`` [2.5] (the lenslet focal
ratio is chosen so a diffraction-limited spot is this wide), ``pyramid_modulation``
[3.0 lambda/D], ``pyramid_modulation_steps`` [8], ``pyramid_pupil_pixels``
[32], ``wfs_photons`` [1e6 per frame], ``photon_noise`` [false],
``num_actuators_across`` [11], ``actuator_crosstalk`` [0.15], ``r0`` [0.15 m
at 500 nm], ``L0`` [25 m], ``wind_speed`` [10 m/s], ``frame_rate`` [1000 Hz],
``science_wavelength`` [1.6e-6 m], ``science_q`` [4], ``science_num_airy``
[16], ``seed`` [1].

DM commands are actuator surface heights in metres (HCIPy convention).
Keep the sub-aperture size ``telescope_diameter / num_lenslets`` near r0 at
the WFS wavelength, as in a real Shack-Hartmann design: much larger
sub-apertures see speckled spots and the loop cannot close on the atmosphere.
"""

from __future__ import annotations

import argparse
import threading
import time
from typing import Any, Mapping

import numpy as np

from pyrtc.component import Component
from pyrtc.logging_utils import get_logger
from pyrtc.rpc import Listener
from pyrtc.science_camera import ScienceCamera
from pyrtc.utils import read_yaml_file, require_optional, set_affinity_and_priority, set_from_config
from pyrtc.wavefront_corrector import WavefrontCorrector
from pyrtc.wavefront_sensor import WavefrontSensor

logger = get_logger(__name__)

DEFAULT_PARAMS: dict[str, Any] = {
    "telescope_diameter": 2.0,
    "central_obscuration": 0.0,
    "wfs_type": "shwfs",
    "wfs_wavelength": 700e-9,
    "num_lenslets": 10,
    "pixels_per_lenslet": 12,
    "spot_fwhm_pixels": 2.5,
    "pyramid_modulation": 3.0,
    "pyramid_modulation_steps": 8,
    "pyramid_pupil_pixels": 32,
    "wfs_photons": 1.0e6,
    "photon_noise": False,
    "num_actuators_across": 11,
    "actuator_crosstalk": 0.15,
    "r0": 0.15,
    "L0": 25.0,
    "wind_speed": 10.0,
    "frame_rate": 1000.0,
    "science_wavelength": 1.6e-6,
    "science_q": 4,
    "science_num_airy": 16,
    "seed": 1,
}


def _hcipy():
    return require_optional("hcipy", "hcipy", "The HCIPy simulator interface")


def _unwrap_context(resource):
    return getattr(resource, "context", resource)


class HCIPySystemContext:
    """Build and own one shared HCIPy simulation for several pyrtc components.

    HCIPy works in physical units; the Shack-Hartmann microlens array is
    modelled at ``_MLA_DIAMETER`` (5 mm) across, the scale of HCIPy's own
    examples. Only the resulting spot size in pixels matters to pyrtc.

    All optical propagation goes through :meth:`wfs_image` and
    :meth:`psf_image` under one lock, because the WFS and science-camera
    workers run in separate threads and share the DM and atmosphere.
    """

    _MLA_DIAMETER = 5e-3

    def __init__(self, param: Mapping[str, Any] | None = None, param_file: str | None = None):
        if param is None and param_file:
            param = read_yaml_file(param_file)
        unknown = set(param or {}) - set(DEFAULT_PARAMS)
        if unknown:
            raise ValueError(f"unknown HCIPy parameters: {sorted(unknown)}")
        self.param = {**DEFAULT_PARAMS, **dict(param or {})}
        self.lock = threading.RLock()
        self.atmosphere_enabled = False
        self.components: dict[str, Any] = {}
        self._build()

    # -- construction ---------------------------------------------------------

    def _build(self) -> None:
        hp = _hcipy()
        p = self.param
        diameter = float(p["telescope_diameter"])
        self.wfs_type = str(p["wfs_type"]).lower()
        if self.wfs_type not in ("shwfs", "pywfs"):
            raise ValueError(f"wfs_type must be 'shwfs' or 'pywfs', got {p['wfs_type']!r}")
        num_pixels = int(p["num_lenslets"]) * int(p["pixels_per_lenslet"])
        self.pupil_grid = hp.make_pupil_grid(num_pixels, diameter)
        if float(p["central_obscuration"]) > 0:
            aperture = hp.make_obstructed_circular_aperture(
                diameter, float(p["central_obscuration"])
            )
        else:
            aperture = hp.make_circular_aperture(diameter)
        self.aperture = hp.evaluate_supersampled(aperture, self.pupil_grid, 4)

        # Deformable mirror: Gaussian influence functions on a square grid,
        # keeping the actuators that fall inside the pupil.
        num_act = int(p["num_actuators_across"])
        pitch = diameter / (num_act - 1)
        influence = hp.make_gaussian_influence_functions(
            self.pupil_grid, num_act, pitch, crosstalk=float(p["actuator_crosstalk"])
        )
        actuator_grid = hp.make_pupil_grid(num_act, pitch * num_act)
        positions = np.column_stack((actuator_grid.x, actuator_grid.y))
        valid = np.hypot(positions[:, 0], positions[:, 1]) <= 0.5 * diameter + 0.25 * pitch
        self.actuator_layout = valid.reshape(num_act, num_act)
        self.actuator_positions = positions[valid]
        self.dm = hp.DeformableMirror(
            hp.ModeBasis([influence[i] for i in np.flatnonzero(valid)], self.pupil_grid)
        )

        # Wavefront sensor optics and detector.
        wavelength = float(p["wfs_wavelength"])
        if self.wfs_type == "shwfs":
            # HCIPy's Shack-Hartmann optics work at the physical scale of the
            # microlens array: magnify the pupil onto it, and pick the focal
            # ratio that gives spots of spot_fwhm_pixels (FWHM ~ lambda F).
            mla_diameter = self._MLA_DIAMETER
            self.wfs_magnifier = hp.Magnifier(mla_diameter / diameter)
            self.wfs_grid = self.pupil_grid.scaled(mla_diameter / diameter)
            pixel_pitch = mla_diameter / num_pixels
            f_number = float(p["spot_fwhm_pixels"]) * pixel_pitch / wavelength
            self.wfs_optics = hp.SquareShackHartmannWavefrontSensorOptics(
                self.wfs_grid, f_number, int(p["num_lenslets"]), mla_diameter
            )
        else:
            self.wfs_magnifier = None
            pupil_pixels = int(p["pyramid_pupil_pixels"])
            self.wfs_grid = hp.make_pupil_grid(2 * pupil_pixels, 2 * diameter)
            pyramid = hp.PyramidWavefrontSensorOptics(
                self.pupil_grid,
                self.wfs_grid,
                separation=diameter,
                pupil_diameter=diameter,
                wavelength_0=wavelength,
                q=3,
            )
            self.wfs_optics = hp.ModulatedPyramidWavefrontSensorOptics(
                pyramid,
                float(p["pyramid_modulation"]) * wavelength / diameter,
                num_steps=int(p["pyramid_modulation_steps"]),
            )
        self.wfs_camera = hp.NoiselessDetector(self.wfs_grid)
        self.wfs_shape = tuple(int(n) for n in self.wfs_grid.shape)

        # Frozen-flow atmosphere, advanced by one frame per WFS exposure.
        self.frame_time = 1.0 / float(p["frame_rate"])
        np.random.seed(int(p["seed"]))  # InfiniteAtmosphericLayer draws from the global RNG
        self.layer = hp.InfiniteAtmosphericLayer(
            self.pupil_grid,
            hp.Cn_squared_from_fried_parameter(float(p["r0"]), 500e-9),
            float(p["L0"]),
            float(p["wind_speed"]),
        )

        # Science camera.
        science_wavelength = float(p["science_wavelength"])
        self.focal_grid = hp.make_focal_grid(
            int(p["science_q"]),
            int(p["science_num_airy"]),
            spatial_resolution=science_wavelength / diameter,
        )
        self.propagator = hp.FraunhoferPropagator(self.pupil_grid, self.focal_grid)
        self.psf_shape = tuple(int(n) for n in self.focal_grid.shape)
        self.reference_psf = self._psf(atmosphere=False, flat=True)
        self.reference_peak = float(self.reference_psf.max()) or 1.0

    # -- propagation ----------------------------------------------------------

    def _wavefront(self, wavelength, power):
        hp = _hcipy()
        wavefront = hp.Wavefront(self.aperture, wavelength)
        wavefront.total_power = power
        return wavefront

    def wfs_image(self) -> np.ndarray:
        """Advance the atmosphere by one frame and return a WFS detector image."""

        hp = _hcipy()
        with self.lock:
            if self.atmosphere_enabled:
                self.layer.t += self.frame_time
            wavefront = self._wavefront(float(self.param["wfs_wavelength"]), 1.0)
            if self.atmosphere_enabled:
                wavefront = self.layer(wavefront)
            wavefront = self.dm(wavefront)
            if self.wfs_magnifier is not None:
                wavefront = self.wfs_magnifier(wavefront)
            output = self.wfs_optics.forward(wavefront)
            outputs = output if isinstance(output, list) else [output]
            for item in outputs:
                self.wfs_camera.integrate(item, 1.0 / len(outputs))
            image = np.asarray(self.wfs_camera.read_out(), dtype=np.float64)
        image *= float(self.param["wfs_photons"]) / max(image.sum(), 1e-300)
        if self.param["photon_noise"]:
            image = np.asarray(hp.large_poisson(image), dtype=np.float64)
        return image.reshape(self.wfs_shape)

    def _psf(self, *, atmosphere: bool, flat: bool = False) -> np.ndarray:
        wavefront = self._wavefront(float(self.param["science_wavelength"]), 1.0)
        if atmosphere:
            wavefront = self.layer(wavefront)
        if not flat:
            wavefront = self.dm(wavefront)
        return np.asarray(self.propagator(wavefront).power, dtype=np.float64).reshape(
            self.psf_shape
        )

    def psf_image(self) -> np.ndarray:
        """Return the current science PSF, normalized to the unaberrated peak."""

        with self.lock:
            psf = self._psf(atmosphere=self.atmosphere_enabled)
        return psf / self.reference_peak

    def set_actuators(self, heights) -> None:
        with self.lock:
            self.dm.actuators = np.asarray(heights, dtype=np.float64)

    # -- control --------------------------------------------------------------

    def add_atmosphere(self):
        self.atmosphere_enabled = True

    def remove_atmosphere(self):
        self.atmosphere_enabled = False

    def register_component(self, section_name: str, component: Any) -> None:
        self.components[str(section_name)] = component

    def get_component(self, section_name: str) -> Any:
        return self.components.get(str(section_name))


class HCIPyWFSensor(WavefrontSensor):
    """Wavefront sensor fed by the shared HCIPy simulation.

    ``width`` and ``height`` follow the simulated detector; a config that
    disagrees is corrected with a log message.
    """

    EXTRA_CONFIG_KEYS = ()

    def __init__(self, wfs_conf, context) -> None:
        self.context = _unwrap_context(context)
        wfs_conf = dict(wfs_conf)
        height, width = self.context.wfs_shape
        if (wfs_conf.get("width"), wfs_conf.get("height")) != (width, height):
            logger.info(
                "HCIPy WFS detector is %sx%s; overriding the configured size", width, height
            )
            wfs_conf["width"], wfs_conf["height"] = width, height
        super().__init__(wfs_conf)
        if self.section_name:
            self.context.register_component(self.section_name, self)

    def expose(self):
        image = self.context.wfs_image()
        self.data = np.clip(np.rint(image), 0, np.iinfo(np.uint16).max).astype(np.uint16)
        super().expose()

    def add_atmosphere(self):
        self.context.add_atmosphere()

    def remove_atmosphere(self):
        self.context.remove_atmosphere()


class HCIPyWFCorrector(WavefrontCorrector):
    """Deformable mirror of the shared HCIPy simulation.

    The actuator count and 2D layout follow the simulated DM (the actuators
    inside the pupil). ``basis_actuator_positions`` gives aobasis the real
    actuator coordinates in metres, and a ``basis`` section without
    ``pupil_diameter`` uses the telescope diameter.
    """

    EXTRA_CONFIG_KEYS = ()

    def __init__(self, corrector_conf, context) -> None:
        self.context = _unwrap_context(context)
        corrector_conf = dict(corrector_conf)
        num_actuators = int(self.context.dm.num_actuators)
        if corrector_conf.get("num_actuators") != num_actuators:
            logger.info(
                "HCIPy DM has %s actuators in the pupil; overriding the config", num_actuators
            )
            corrector_conf["num_actuators"] = num_actuators
        self._hcipy_geometry_ready = False
        super().__init__(corrector_conf)
        if self.section_name:
            self.context.register_component(self.section_name, self)
        self.set_layout(self.context.actuator_layout)
        self._hcipy_geometry_ready = True
        self.read_m2c()

    def read_m2c(self, filename=""):
        if not getattr(self, "_hcipy_geometry_ready", False):
            self.set_m2c(None)
            return
        if self.basis_conf is not None:
            self.basis_conf = self.basis_conf.with_defaults(
                pupil_diameter=float(self.context.param["telescope_diameter"])
            )
        super().read_m2c(filename)

    def basis_actuator_positions(self):
        return np.asarray(self.context.actuator_positions, dtype=np.float64)

    def send_to_hardware(self):
        super().send_to_hardware()
        self.context.set_actuators(self.current_shape)


class HCIPyScienceCamera(ScienceCamera):
    """Science camera rendering the shared HCIPy system's PSF.

    Frames are scaled so the unaberrated peak fills the detector's integer
    range, and the unaberrated PSF is the model for Strehl estimates.
    """

    EXTRA_CONFIG_KEYS = ()

    def __init__(self, science_conf, context) -> None:
        self.context = _unwrap_context(context)
        science_conf = dict(science_conf)
        height, width = self.context.psf_shape
        if (science_conf.get("width"), science_conf.get("height")) != (width, height):
            logger.info("HCIPy PSF is %sx%s; overriding the configured size", width, height)
            science_conf["width"], science_conf["height"] = width, height
        super().__init__(science_conf)
        if self.section_name:
            self.context.register_component(self.section_name, self)
        reference = self.context.reference_psf / self.context.reference_peak
        self.set_model_psf(self._to_detector(reference).astype(self.psf_long_dtype))

    def _to_detector(self, normalized):
        full_scale = np.iinfo(self.image_raw_dtype).max
        return np.clip(normalized * full_scale, 0, full_scale)

    def expose(self):
        self.data = self._to_detector(self.context.psf_image()).astype(self.image_raw_dtype)
        super().expose()

    def integrate(self):
        super().integrate()
        if np.max(self.model) > 0:
            self.compute_strehl(median_filter_size=1, gaussian_sigma=0)

    def add_atmosphere(self):
        self.context.add_atmosphere()

    def remove_atmosphere(self):
        self.context.remove_atmosphere()


class HCIPyInterface(Component):
    """Provider component owning the shared HCIPy simulation.

    In a manager config, give this section a ``param`` mapping or a
    ``param_file`` (and optionally ``use_atmosphere``) and let the ``wfs``,
    ``wfc`` and ``psf`` sections declare ``resource: <this section>``.
    Standalone (a full system config with ``wfs``, ``wfc`` and ``psf``
    sections, as the examples use), it builds those three components itself;
    get them with :meth:`get_hardware`.
    """

    def __init__(self, conf, param=None) -> None:
        self.conf = conf
        self.logger = get_logger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self._standalone_mode = isinstance(conf, Mapping) and all(
            key in conf for key in ("wfs", "wfc", "psf")
        )
        if self._standalone_mode:
            self.system_conf = dict(conf)
            section = self.system_conf.get("hcipy") or {}
            if param is None:
                param = section.get("param")
                param_file = section.get("param_file")
            else:
                param_file = None
            self.context = HCIPySystemContext(param, param_file)
            self.use_atmosphere = bool(section.get("use_atmosphere", True))
            self.wfs_interface = HCIPyWFSensor(self.system_conf["wfs"], self.context)
            self.dm_interface = HCIPyWFCorrector(self.system_conf["wfc"], self.context)
            self.psf_interface = HCIPyScienceCamera(self.system_conf["psf"], self.context)
            self._apply_atmosphere()
            return

        self.context = HCIPySystemContext(
            param if param is not None else conf.get("param"), conf.get("param_file")
        )
        self.use_atmosphere = bool(set_from_config(conf, "use_atmosphere", False))
        super().__init__(conf)
        if self.section_name:
            self.context.register_component(self.section_name, self)
        self._apply_atmosphere()

    def _apply_atmosphere(self):
        if self.use_atmosphere:
            self.add_atmosphere()
        else:
            self.remove_atmosphere()

    def add_atmosphere(self):
        self.context.add_atmosphere()
        self.logger.info("Enabled HCIPy atmosphere")

    def remove_atmosphere(self):
        self.context.remove_atmosphere()
        self.logger.info("Disabled HCIPy atmosphere")

    def get_hardware(self):
        """Return the WFS, WFC and science-camera components."""

        if self._standalone_mode:
            return self.wfs_interface, self.dm_interface, self.psf_interface
        return (
            self.context.get_component("wfs"),
            self.context.get_component("wfc"),
            self.context.get_component("psf"),
        )

    def close(self, **kwargs):
        if self._standalone_mode:
            for component in (self.wfs_interface, self.dm_interface, self.psf_interface):
                component.close(**kwargs)
            return
        super().close(**kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serve an HCIPy simulation over pyrtc RPC.")
    parser.add_argument("-c", "--config", required=True, help="Path to the pyrtc config file")
    parser.add_argument("--param-file", help="HCIPy parameter YAML file")
    parser.add_argument("-p", "--port", required=True, help="Port for communication")
    args = parser.parse_args()

    conf = read_yaml_file(args.config)
    set_affinity_and_priority("main", conf["wfs"].get("affinity"))
    param = read_yaml_file(args.param_file) if args.param_file else None
    sim = HCIPyInterface(conf=conf, param=param)
    listener = Listener(sim, port=int(args.port))
    while listener.running:
        listener.listen()
        time.sleep(1e-3)
