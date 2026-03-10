"""Bridge between pyRTC components and an OOPAO optical simulation.

This module adapts the OOPAO telescope, atmosphere, deformable-mirror, pyramid
sensor, and PSF-camera objects into the pyRTC component interfaces. It is used
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

from pyRTC.logging_utils import get_logger
from pyRTC.Pipeline import Listener
from pyRTC.ScienceCamera import ScienceCamera
from pyRTC.WavefrontCorrector import WavefrontCorrector
from pyRTC.WavefrontSensor import WavefrontSensor
from pyRTC.utils import decrease_nice, read_yaml_file, set_affinity

from OOPAO.Atmosphere import Atmosphere
from OOPAO.DeformableMirror import DeformableMirror
from OOPAO.Pyramid import Pyramid
from OOPAO.Source import Source
from OOPAO.Telescope import Telescope


logger = get_logger(__name__)

class _OOPAOWFSensor(WavefrontSensor):
    """Wavefront-sensor wrapper around an OOPAO pyramid sensor.

    The wrapper advances the simulated atmosphere when required, propagates the
    guide star through the telescope and deformable mirror, and exposes the
    resulting detector frame through the standard pyRTC ``WavefrontSensor`` API.
    """

    def __init__(self, wfsConf, tel, ngs, atm, dm, wfs) -> None:
        
        self.tel = tel
        self.ngs = ngs
        self.atm = atm
        self.dm = dm
        self.wfs = wfs
        
        super().__init__(wfsConf)

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

    def addAtmosphere(self):
        self.tel+self.atm

    def removeAtmosphere(self):
        self.tel-self.atm

class _OOPAOWFCorrector(WavefrontCorrector):
    """Wavefront-corrector wrapper for an OOPAO deformable mirror.

    This adapter maps pyRTC command vectors onto the OOPAO deformable-mirror
    coefficient array so the simulated optical train responds to control-loop
    updates exactly where a physical mirror would in a deployed system.
    """

    def __init__(self, correctorConf, tel, dm) -> None:
    
        self.tel = tel
        self.dm = dm
        
        self.dm.coefs = 0
        super().__init__(correctorConf)

        #Set-up additional pyRTC parameters from simulation
        numActuators = self.dm.validAct.size
        self.setLayout(self.dm.validAct.reshape(int(np.sqrt(numActuators)),
                                                            int(np.sqrt(numActuators))))

    def readM2C(self, filename=''):
        self.setM2C(None)
    
    def sendToHardware(self):
        
        super().sendToHardware()

        self.dm.coefs = self.currentShape.astype(np.float64)

    def setFlat(self, flat):
        super().setFlat(flat)
        self.dm.flat = flat 


class _OOPAOScienceCamera(ScienceCamera):
    """Science-camera wrapper around the OOPAO PSF path.

    The class reuses the current atmosphere and deformable-mirror state to
    synthesize a PSF image that can be consumed by pyRTC exactly like a hardware
    science camera. It is intentionally simulation-facing and does not attempt
    to hide OOPAO-specific PSF generation details.
    """

    def __init__(self, scienceConf, tel, src, atm, dm) -> None:
        self.tel = tel
        self.src = src
        self.atm = atm
        self.dm = dm
        self._atmosphere_enabled = False
        super().__init__(scienceConf)
        self._reference_psf = self._render_reference_psf()
        self._reference_peak = float(np.max(self._reference_psf)) if self._reference_psf.size else 1.0
        if self._reference_peak <= 0:
            self._reference_peak = 1.0
        self.setModelPSF(self._scale_psf_to_detector(self._reference_psf).astype(self.psfLongDtype))

    def _compute_psf(self, opd_no_pupil):
        self.src ** self.tel
        self.src.OPD_no_pupil = opd_no_pupil
        self.tel.computePSF(zeroPaddingFactor=5)
        return np.array(self.tel.PSF, dtype=np.float64, copy=True)

    def _render_reference_psf(self):
        zero_opd = np.zeros(self.tel.pupil.shape, dtype=np.float64)
        reference = self._compute_psf(zero_opd)
        return np.nan_to_num(reference, nan=0.0, posinf=0.0, neginf=0.0)

    def _scale_psf_to_detector(self, psf):
        psf = np.nan_to_num(psf, nan=0.0, posinf=0.0, neginf=0.0)
        if psf.size == 0:
            return np.zeros(self.imageShape, dtype=np.float64)

        scaled = psf / self._reference_peak
        scaled *= np.iinfo(self.imageRawDType).max
        return np.clip(scaled, 0, np.iinfo(self.imageRawDType).max)

    def _current_opd_no_pupil(self):
        base_opd = np.zeros(self.tel.pupil.shape, dtype=np.float64)
        if self._atmosphere_enabled:
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
        return self._scale_psf_to_detector(psf).astype(self.imageRawDType)
        
    def expose(self):
        self.data = self._render_psf_frame()
        
        super().expose()

        return

    def integrate(self):
        super().integrate()
        if np.max(self.model) > 0:
            self.computeStrehl(median_filter_size=1, gaussian_sigma=0)
        return
    
    def addAtmosphere(self):
        self._atmosphere_enabled = True

    def removeAtmosphere(self):
        self._atmosphere_enabled = False

class OOPAOInterface():
    """Assembles a complete pyRTC-compatible OOPAO simulation stack.

    ``OOPAOInterface`` creates the simulated telescope, atmosphere, guide star,
    deformable mirror, pyramid sensor, and science camera, then wraps the key
    pieces in pyRTC component adapters. The resulting objects can be launched or
    driven through the same orchestration code used for physical hardware,
    making the class useful for algorithm development, documentation examples,
    and end-to-end synthetic tests.
    """

    OBJECT_NAMES = ("tel", "tel_psf", "ngs", "src", "atm", "dm", "wfs")

    def __init__(self, conf, param=None, tel=None, tel_psf=None, ngs=None, src=None, atm=None, dm=None, wfs=None) -> None:

        self.conf = conf
        self.param = self._load_param_mapping(param)
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
                "OOPAOInterface no longer ships an embedded default parameter payload. "
                + self._describe_supported_inputs()
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
            self.wfs = self._build_object("wfs", Pyramid, telescope=self.tel)

        if hasattr(self.atm, "initializeAtmosphere"):
            logger.info("Initializing OOPAO atmosphere against the telescope model")
            self.atm.initializeAtmosphere(self.tel)

        wfsConf = conf["wfs"]
        correctorConf = conf["wfc"]
        scienceConf = conf["psf"]

        self.wfsInterface = _OOPAOWFSensor(wfsConf, self.tel, self.ngs, self.atm, self.dm, self.wfs)
        self.dmInterface  = _OOPAOWFCorrector(correctorConf, self.tel, self.dm)
        self.psfInterface = _OOPAOScienceCamera(scienceConf, self.tel_psf, self.src, self.atm, self.dm)

        logger.info(
            "OOPAOInterface ready. Pass param as a loaded flat dict of OOPAO constructor-style keys, "
            "or provide explicit tel=..., atm=..., wfs=... arguments to reuse existing OOPAO objects."
        )

        #Add the atmosphere to the system
        self.addAtmosphere()

    def addAtmosphere(self):
        self.psfInterface.addAtmosphere()
        self.wfsInterface.addAtmosphere()

    def removeAtmosphere(self):
        self.psfInterface.removeAtmosphere()
        self.wfsInterface.removeAtmosphere()

    def restartSimulation(self):
        del self.wfsInterface
        del self.dmInterface
        del self.psfInterface

        self.__init__(
            self.conf,
            param=self.param,
            tel=self.tel_input,
            tel_psf=self.tel_psf_input,
            ngs=self.ngs_input,
            src=self.src_input,
            atm=self.atm_input,
            dm=self.dm_input,
            wfs=self.wfs_input,
        )

    def get_hardware(self):
        return self.wfsInterface, self.dmInterface, self.psfInterface

    def _describe_supported_inputs(self):
        return (
            "Provide param as a flat dict using OOPAO constructor argument names where possible. "
            "Non-source objects are built by forwarding any matching keys from that flat dict. "
            "The two Source objects are the only special case: use ngs_band and ngs_magnitude for the guide star, "
            "and science_band and science_magnitude for the science source. "
            "If a flat key is ambiguous across OOPAO constructors, pass the already-built object explicitly. "
            "You can also pass already-built objects through the matching explicit constructor arguments."
        )

    def _load_param_mapping(self, param: Any) -> dict[str, Any]:
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

if __name__ == "__main__":

    # Create argument parser
    parser = argparse.ArgumentParser(description="Read a config file from the command line.")

    # Add command-line argument for the config file
    parser.add_argument("-c", "--config", required=True, help="Path to the pyRTC config file")
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