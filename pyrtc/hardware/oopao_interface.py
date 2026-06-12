"""Bridge between pyrtc components and an OOPAO optical simulation.

This module adapts the OOPAO telescope, atmosphere, deformable-mirror, wavefront
sensor, and PSF-camera objects into the pyrtc component interfaces. It is used
for simulation-backed development and validation where the control stack should
behave as if it were driving real hardware.
"""

from __future__ import annotations

import argparse
import inspect
import os
import time
from typing import Any, Mapping

import numpy as np

from pyrtc.logging_utils import get_logger
from pyrtc.rpc import Listener
from pyrtc.component import Component
from pyrtc.science_camera import ScienceCamera
from pyrtc.wavefront_corrector import WavefrontCorrector
from pyrtc.wavefront_sensor import WavefrontSensor
from pyrtc.utils import decrease_nice, read_yaml_file, set_from_config, set_affinity

from OOPAO.Atmosphere import Atmosphere
from OOPAO.DeformableMirror import DeformableMirror
from OOPAO.Pyramid import Pyramid
from OOPAO.Source import Source
from OOPAO.Telescope import Telescope

try:
    from OOPAO.ShackHartmann import ShackHartmann
except ModuleNotFoundError:  # pragma: no cover - exercised by fake-module tests
    ShackHartmann = None


logger = get_logger(__name__)


def _unwrap_oopao_context(resource):
    return getattr(resource, "context", resource)


def _oopao_slopes_type(system_conf: Mapping[str, Any] | None) -> str:
    if not isinstance(system_conf, Mapping):
        return "pywfs"
    slopes_conf = system_conf.get("slopes")
    if not isinstance(slopes_conf, Mapping):
        return "pywfs"
    slopes_type = slopes_conf.get("type", "PYWFS")
    if not isinstance(slopes_type, str):
        return "pywfs"
    slopes_type = slopes_type.strip().lower()
    return slopes_type or "pywfs"

class OOPAOWFSensor(WavefrontSensor):
    """Wavefront-sensor wrapper around an OOPAO wavefront sensor.

    The wrapper advances the simulated atmosphere when required, propagates the
    guide star through the telescope and deformable mirror, and exposes the
    resulting detector frame through the standard pyrtc ``WavefrontSensor`` API.
    """

    def __init__(self, wfs_conf, context) -> None:
        self.context = _unwrap_oopao_context(context)
        self.tel = self.context.tel
        self.ngs = self.context.ngs
        self.atm = self.context.atm
        self.dm = self.context.dm
        self.wfs = self.context.wfs
        super().__init__(wfs_conf)
        if self.section_name:
            self.context.register_component(self.section_name, self)

    def _propagate_source(self):
        if self.tel.isPaired:
            # Atmosphere.update() rebuilds the source/telescope state from
            # scratch, so only the DM and WFS need to be applied afterwards.
            self.atm.update()
            self.ngs * self.dm * self.wfs
            return

        # Without atmosphere, OOPAO does not reset the source state for us.
        # Rebuilding the source/telescope path each frame prevents the DM OPD
        # from accumulating across repeated exposures of a static command.
        self.ngs ** self.tel
        self.ngs * self.dm * self.wfs
        
    def expose(self):
        self._propagate_source()

        #Generate a new exposure
        self.data = self.wfs.cam.frame.astype(np.uint16)
        super().expose()

        return

    def add_atmosphere(self):
        self.context.add_atmosphere()

    def remove_atmosphere(self):
        self.context.remove_atmosphere()

class OOPAOWFCorrector(WavefrontCorrector):
    """Wavefront-corrector wrapper for an OOPAO deformable mirror.

    This adapter maps pyrtc command vectors onto the OOPAO deformable-mirror
    coefficient array so the simulated optical train responds to control-loop
    updates exactly where a physical mirror would in a deployed system.
    """

    def __init__(self, corrector_conf, context) -> None:
        self.context = _unwrap_oopao_context(context)
        self.tel = self.context.tel
        self.dm = self.context.dm
        self.dm.coefs = 0
        super().__init__(corrector_conf)
        if self.section_name:
            self.context.register_component(self.section_name, self)

        #Set-up additional pyrtc parameters from simulation
        num_actuators = self.dm.validAct.size
        self.set_layout(self.dm.validAct.reshape(int(np.sqrt(num_actuators)),
                                                            int(np.sqrt(num_actuators))))

    def read_m2c(self, filename=''):
        self.set_m2c(None)
    
    def send_to_hardware(self):
        
        super().send_to_hardware()

        self.dm.coefs = self.current_shape.astype(np.float64)

    def set_flat(self, flat):
        super().set_flat(flat)
        self.dm.flat = flat 


class OOPAOScienceCamera(ScienceCamera):
    """Science-camera wrapper around the OOPAO PSF path.

    The class reuses the current atmosphere and deformable-mirror state to
    synthesize a PSF image that can be consumed by pyrtc exactly like a hardware
    science camera. It is intentionally simulation-facing and does not attempt
    to hide OOPAO-specific PSF generation details.
    """

    def __init__(self, science_conf, context) -> None:
        self.context = _unwrap_oopao_context(context)
        self.tel = self.context.tel_psf
        self.src = self.context.src
        self.atm = self.context.atm
        self.dm = self.context.dm
        super().__init__(science_conf)
        if self.section_name:
            self.context.register_component(self.section_name, self)
        self._reference_psf = self._render_reference_psf()
        self._reference_peak = float(np.max(self._reference_psf)) if self._reference_psf.size else 1.0
        if self._reference_peak <= 0:
            self._reference_peak = 1.0
        self.set_model_psf(self._scale_psf_to_detector(self._reference_psf).astype(self.psf_long_dtype))

    def _compute_psf(self, opd_no_pupil):
        self.src ** self.tel
        self.src.OPD_no_pupil = opd_no_pupil
        self.tel.compute_psf(zero_padding_factor=5)
        return np.array(self.tel.PSF, dtype=np.float64, copy=True)

    def _render_reference_psf(self):
        zero_opd = np.zeros(self.tel.pupil.shape, dtype=np.float64)
        reference = self._compute_psf(zero_opd)
        return np.nan_to_num(reference, nan=0.0, posinf=0.0, neginf=0.0)

    def _scale_psf_to_detector(self, psf):
        psf = np.nan_to_num(psf, nan=0.0, posinf=0.0, neginf=0.0)
        if psf.size == 0:
            return np.zeros(self.image_shape, dtype=np.float64)

        scaled = psf / self._reference_peak
        scaled *= np.iinfo(self.image_raw_dtype).max
        return np.clip(scaled, 0, np.iinfo(self.image_raw_dtype).max)

    def _current_opd_no_pupil(self):
        base_opd = np.zeros(self.tel.pupil.shape, dtype=np.float64)
        if self.context.atmosphere_enabled:
            if getattr(self.atm, "OPD_no_pupil", None) is not None:
                base_opd = np.array(self.atm.OPD_no_pupil, dtype=np.float64, copy=True)
            elif getattr(self.atm, "OPD", None) is not None:
                base_opd = np.array(self.atm.OPD, dtype=np.float64, copy=True)

        dm_opd = getattr(self.dm, "OPD", None)
        if dm_opd is None:
            return base_opd

        dm_opd = np.asarray(dm_opd)
        if dm_opd.ndim == 2:
            return base_opd + dm_opd.astype(np.float64, copy=False)

        # Interaction-matrix calibration can temporarily drive the DM with a cube.
        # The science camera only renders one frame at a time, so keep the most
        # recent 2D command if the PSF path is left running during calibration.
        return base_opd + dm_opd[..., -1].astype(np.float64, copy=False)

    def _render_psf_frame(self):
        psf = self._compute_psf(self._current_opd_no_pupil())
        return self._scale_psf_to_detector(psf).astype(self.image_raw_dtype)
        
    def expose(self):
        self.data = self._render_psf_frame()
        
        super().expose()

        return

    def integrate(self):
        super().integrate()
        if np.max(self.model) > 0:
            self.compute_strehl(median_filter_size=1, gaussian_sigma=0)
        return
    
    def add_atmosphere(self):
        self.context.add_atmosphere()

    def remove_atmosphere(self):
        self.context.remove_atmosphere()

class OOPAOSystemContext:
    """Builds and owns the shared OOPAO simulation objects for soft-rtc components.

    This context is meant to be instantiated once by the manager and then
    injected into several pyrtc components that need to share a single OOPAO
    telescope, atmosphere, DM, and sensor graph.
    """

    OBJECT_NAMES = ("tel", "tel_psf", "ngs", "src", "atm", "dm", "wfs")

    def __init__(self, resource_conf, system_conf=None, tel=None, tel_psf=None, ngs=None, src=None, atm=None, dm=None, wfs=None) -> None:
        self.resource_conf = dict(resource_conf)
        self.system_conf = {} if system_conf is None else dict(system_conf)
        self.param = self._load_param_mapping(self.resource_conf.get("param"), self.resource_conf.get("param_file"))
        self.atmosphere_enabled = False
        self.components: dict[str, Any] = {}
        self.tel_input = tel
        self.tel_psf_input = tel_psf
        self.ngs_input = ngs
        self.src_input = src
        self.atm_input = atm
        self.dm_input = dm
        self.wfs_input = wfs
        self.tel = tel
        self.tel_psf = tel_psf
        self.ngs = ngs
        self.src = src
        self.atm = atm
        self.dm = dm
        self.wfs = wfs

        if not self.param and all(getattr(self, name) is None for name in self.OBJECT_NAMES):
            raise ValueError(
                "OOPAOSystemContext requires either param=<mapping> or param_file=<YAML path>, "
                "or explicit pre-built OOPAO objects."
            )

        if self.tel is None:
            self.tel = self._build_object("tel", Telescope)
        if self.tel_psf is None:
            self.tel_psf = self._build_object("tel_psf", Telescope)
        if self.ngs is None:
            self.ngs = self._build_object("ngs", Source)
        if self.src is None:
            self.src = self._build_object("src", Source)

        if hasattr(self.ngs, "__mul__"):
            self.ngs * self.tel
        if hasattr(self.src, "__mul__"):
            self.src * self.tel_psf

        if self.atm is None:
            self.atm = self._build_object("atm", Atmosphere, telescope=self.tel)
        if self.dm is None:
            self.dm = self._build_object("dm", DeformableMirror, telescope=self.tel)
        if self.wfs is None:
            slopes_type = _oopao_slopes_type(self.system_conf)
            if slopes_type == "shwfs":
                if ShackHartmann is None:
                    raise ImportError("OOPAO ShackHartmann support is unavailable in the current environment")
                wfs_factory = ShackHartmann
            else:
                wfs_factory = Pyramid
            self.wfs = self._build_object("wfs", wfs_factory, telescope=self.tel)

        if hasattr(self.atm, "initializeAtmosphere"):
            logger.info("Initializing OOPAO atmosphere against the telescope model")
            self.atm.initializeAtmosphere(self.tel)

    def add_atmosphere(self):
        if getattr(self.tel, "isPaired", False):
            self.atmosphere_enabled = True
            return None
        if hasattr(self.tel, "__add__"):
            self.tel + self.atm
        self.atmosphere_enabled = True
        return None

    def remove_atmosphere(self):
        if not getattr(self.tel, "isPaired", False):
            self.atmosphere_enabled = False
            return None
        if hasattr(self.tel, "__sub__"):
            self.tel - self.atm
        self.atmosphere_enabled = False
        return None

    def register_component(self, section_name: str, component: Any) -> None:
        self.components[str(section_name)] = component

    def get_component(self, section_name: str) -> Any:
        return self.components.get(str(section_name))

    def restart_simulation(self):
        self.__init__(
            self.resource_conf,
            self.system_conf,
            tel=self.tel_input,
            tel_psf=self.tel_psf_input,
            ngs=self.ngs_input,
            src=self.src_input,
            atm=self.atm_input,
            dm=self.dm_input,
            wfs=self.wfs_input,
        )

    def _describe_supported_inputs(self):
        return (
            "Provide param as a flat dict using OOPAO constructor argument names where possible. "
            "Non-source objects are built by forwarding any matching keys from that flat dict. "
            "The two Source objects are the only special case: use ngs_band and ngs_magnitude for the guide star, "
            "and science_band and science_magnitude for the science source. "
            "If a flat key is ambiguous across OOPAO constructors, pass the already-built object explicitly. "
            "You can also pass already-built objects through the matching explicit constructor arguments."
        )

    def _load_param_mapping(self, param: Any, param_file: Any = None) -> dict[str, Any]:
        if param is None and isinstance(param_file, str) and param_file:
            param = read_yaml_file(param_file)

        if param is None:
            return {}
        if not isinstance(param, Mapping):
            raise TypeError(f"OOPAO parameters must be a mapping/dict, got {type(param).__name__}")
        return dict(param)

    def _build_object(self, object_name: str, factory, **extra_kwargs):
        if object_name == "ngs":
            kwargs = {
                "optBand": self.param.get("ngs_band"),
                "magnitude": self.param.get("ngs_magnitude"),
            }
            kwargs.update(extra_kwargs)
            kwargs = {key: value for key, value in kwargs.items() if value is not None}
        elif object_name == "src":
            kwargs = {
                "optBand": self.param.get("science_band"),
                "magnitude": self.param.get("science_magnitude"),
            }
            kwargs.update(extra_kwargs)
            kwargs = {key: value for key, value in kwargs.items() if value is not None}
        else:
            kwargs = {}
            signature = inspect.signature(factory)
            for parameter_name, parameter in signature.parameters.items():
                if parameter_name == "self":
                    continue
                if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                    continue
                if parameter_name in extra_kwargs:
                    kwargs[parameter_name] = extra_kwargs[parameter_name]
                    continue
                if parameter_name in self.param:
                    value = self.param[parameter_name]
                    if object_name == "dm" and parameter_name == "altitude" and not np.isscalar(value):
                        logger.info(
                            "Skipping non-scalar 'altitude' when building OOPAO object 'dm'; treating it as atmosphere layer heights"
                        )
                        continue
                    kwargs[parameter_name] = value

        logger.info("Building OOPAO object '%s' with kwargs keys %s", object_name, sorted(kwargs))

        try:
            return factory(**kwargs)
        except TypeError as exc:
            raise TypeError(
                f"Failed to build OOPAO object '{object_name}' from kwargs {sorted(kwargs)}. "
                f"{self._describe_supported_inputs()}"
            ) from exc


class OOPAOInterface(Component):
    """Manager-visible soft-rtc provider component for shared OOPAO simulation state.

    This component owns one shared :class:`OOPAOSystemContext` and exposes the
    notebook-style control actions that operators need from the manager GUI.
    Downstream OOPAO-backed pyrtc components can declare ``resource: oopao`` to
    reuse this provider instance inside the same soft-rtc manager process.
    """

    def __init__(self, conf, param=None, tel=None, tel_psf=None, ngs=None, src=None, atm=None, dm=None, wfs=None) -> None:
        self.conf = conf
        self.logger = get_logger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self._standalone_mode = isinstance(conf, Mapping) and all(key in conf for key in ("wfs", "wfc", "psf"))
        if self._standalone_mode:
            self.system_conf = dict(conf)
            resource_conf = {"param": param} if param is not None else {}
            self.context = OOPAOSystemContext(
                resource_conf,
                self.system_conf,
                tel=tel,
                tel_psf=tel_psf,
                ngs=ngs,
                src=src,
                atm=atm,
                dm=dm,
                wfs=wfs,
            )
            self.kl_basis = None
            self.use_atmosphere = True
            self.wfs_interface = OOPAOWFSensor(self.system_conf["wfs"], self.context)
            self.dm_interface = OOPAOWFCorrector(self.system_conf["wfc"], self.context)
            self.psf_interface = OOPAOScienceCamera(self.system_conf["psf"], self.context)
            self.wfc_section = "wfc"
            self.add_atmosphere()
            return

        self.system_conf = conf.get("_systemConfig", conf)
        self.context = OOPAOSystemContext(
            conf,
            self.system_conf,
            tel=tel,
            tel_psf=tel_psf,
            ngs=ngs,
            src=src,
            atm=atm,
            dm=dm,
            wfs=wfs,
        )
        self.kl_basis = None
        self.use_atmosphere = bool(set_from_config(conf, "use_atmosphere", False))
        self.wfc_section = str(set_from_config(conf, "wfc_section", "wfc"))
        super().__init__(conf)
        if self.section_name:
            self.context.register_component(self.section_name, self)
        if self.use_atmosphere:
            self.add_atmosphere()
        else:
            self.remove_atmosphere()

    def compute_kl_basis(self):
        """Compute and cache the KL modal basis from the current OOPAO context."""

        from OOPAO.calibration.compute_KL_modal_basis import compute_KL_basis

        self.kl_basis = compute_KL_basis(self.context.tel, self.context.atm, self.context.dm)
        self.logger.info("Computed KL basis shape=%s", getattr(self.kl_basis, "shape", None))
        return self.kl_basis

    def load_kl_basis_to_wfc(self):
        """Apply the cached KL basis to the shared wavefront-corrector component."""

        if self.kl_basis is None:
            self.compute_kl_basis()
        wfc_section = self.wfc_section
        wfc_component = self.context.get_component(wfc_section)
        if wfc_component is None:
            raise RuntimeError(f"OOPAOInterface: wavefront-corrector component '{wfc_section}' is not active")
        num_modes = int(self.context.system_conf.get(wfc_section, {}).get("num_modes", self.kl_basis.shape[1]))
        wfc_component.set_m2c(self.kl_basis[:, :num_modes])
        self.logger.info("Loaded KL basis into %s with num_modes=%s", wfc_section, num_modes)
        return num_modes

    def compute_and_load_kl_basis(self):
        """Compute the KL basis and immediately apply it to the active WFC component."""

        self.compute_kl_basis()
        return self.load_kl_basis_to_wfc()

    def add_atmosphere(self):
        self.context.add_atmosphere()
        self.logger.info("Enabled OOPAO atmosphere")

    def remove_atmosphere(self):
        self.context.remove_atmosphere()
        self.logger.info("Disabled OOPAO atmosphere")

    def get_hardware(self):
        """Return the wrapped WFS, WFC, and PSF components in standalone mode."""

        if self._standalone_mode:
            return self.wfs_interface, self.dm_interface, self.psf_interface
        return (
            self.context.get_component("wfs"),
            self.context.get_component(self.wfc_section),
            self.context.get_component("psf"),
        )

_OOPAOWFSensor = OOPAOWFSensor
_OOPAOWFCorrector = OOPAOWFCorrector
_OOPAOScienceCamera = OOPAOScienceCamera

if __name__ == "__main__":

    # Create argument parser
    parser = argparse.ArgumentParser(description="Read a config file from the command line.")

    # Add command-line argument for the config file
    parser.add_argument("-c", "--config", required=True, help="Path to the pyrtc config file")
    parser.add_argument("--param-file", help="Path to an OOPAO parameter YAML file used to build the simulator objects")
    parser.add_argument("-p", "--port", required=True, help="Port for communication")

    # Parse command-line arguments
    args = parser.parse_args()

    conf = read_yaml_file(args.config)

    pid = os.getpid()
    set_affinity((conf["wfs"]["affinity"])%os.cpu_count()) 
    decrease_nice(pid)

    param = read_yaml_file(args.param_file) if args.param_file else None

    sim = OOPAOInterface(conf=conf, param=param)
    
    listener = Listener(sim, port= int(args.port))
    while listener.running:
        listener.listen()
        time.sleep(1e-3)