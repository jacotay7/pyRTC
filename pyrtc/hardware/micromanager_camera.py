"""Camera adapters on Micro-Manager via pymmcore-plus.

`pymmcore-plus <https://pymmcore-plus.github.io/pymmcore-plus/>`_ drives
Micro-Manager's device layer from Python, and with it any camera Micro-Manager
supports (Andor, Hamamatsu, PCO, Photometrics, many machine-vision cameras).
Install with ``pip install pyrtcao[micromanager]`` plus a Micro-Manager
installation (``mmcore install`` fetches one). Then describe the hardware
in a Micro-Manager ``.cfg`` file.

Config keys, on top of the usual WFS or science-camera ones:

``mm_config``
    Micro-Manager system configuration (``.cfg``) to load.
``mm_path``
    Micro-Manager installation directory. Default: pymmcore-plus's
    ``find_micromanager()``.
``camera``
    Camera device label, if the configuration has more than one camera.
``exposure`` (ms), ``binning``, ROI (``width``, ``height``, ``left``, ``top``)
    Applied through the core.
``properties``
    ``{device: {property: value}}`` for anything else (gain, readout mode,
    cooling, ...), applied in order.
``frame_timeout``
    Seconds to wait for a frame (default 1.0).

Frames stream from continuous sequence acquisition. Each ``expose`` returns
the newest frame and drops older ones, so the loop always sees the latest
exposure. Micro-Manager images are ``(height, width)``; they are transposed
into pyrtc's ``(width, height)`` stream shape.
"""

from __future__ import annotations

import time
from typing import Any, Mapping

import numpy as np

from pyrtc.logging_utils import get_logger
from pyrtc.manager import launch_component
from pyrtc.science_camera import ScienceCamera
from pyrtc.utils import require_optional
from pyrtc.wavefront_sensor import WavefrontSensor

logger = get_logger(__name__)

_MM_KEYS = (
    "mm_config",
    "mm_path",
    "camera",
    "exposure",
    "binning",
    "left",
    "top",
    "properties",
    "frame_timeout",
)


class _MicroManagerDevice:
    """Core setup, property access and frame streaming shared by the adapters."""

    def _open_micromanager(self, conf: Mapping[str, Any]) -> None:
        pymmcore_plus = require_optional(
            "pymmcore_plus", "micromanager", "The Micro-Manager camera adapter"
        )
        if not conf.get("mm_config"):
            raise ValueError("Micro-Manager camera: set 'mm_config' to a Micro-Manager .cfg file")
        self._core = pymmcore_plus.CMMCorePlus()
        mm_path = conf.get("mm_path") or pymmcore_plus.find_micromanager()
        if mm_path:
            self._core.setDeviceAdapterSearchPaths([str(mm_path)])
        self._core.loadSystemConfiguration(str(conf["mm_config"]))
        if conf.get("camera"):
            self._core.setCameraDevice(str(conf["camera"]))
        self.camera_label = self._core.getCameraDevice()
        if not self.camera_label:
            raise RuntimeError("Micro-Manager configuration has no camera device")
        self.frame_timeout = float(conf.get("frame_timeout", 1.0))

        if "binning" in conf:
            self.set_binning(conf["binning"])
        if all(key in conf for key in ("width", "height", "left", "top")):
            self.set_roi([conf["width"], conf["height"], conf["left"], conf["top"]])
        if "exposure" in conf:
            self.set_exposure(conf["exposure"])
        for device, properties in dict(conf.get("properties") or {}).items():
            for name, value in dict(properties).items():
                self.set_property(device, name, value)
        self._core.startContinuousSequenceAcquisition(0)
        self.logger.info("Opened Micro-Manager camera %s", self.camera_label)

    def _ready(self) -> bool:
        return getattr(self, "_core", None) is not None

    def set_property(self, device: str, name: str, value) -> None:
        """Set any Micro-Manager device property."""

        self._core.setProperty(str(device), str(name), value)
        self.logger.info("Set Micro-Manager %s.%s=%s", device, name, value)

    def set_exposure(self, exposure):
        super().set_exposure(exposure)
        if self._ready():
            self._core.setExposure(float(exposure))

    def set_binning(self, binning):
        super().set_binning(binning)
        if self._ready():
            self._core.setProperty(self.camera_label, "Binning", str(int(binning)))

    def set_roi(self, roi):
        super().set_roi(roi)
        if self._ready():
            width, height, left, top = (int(value) for value in roi)
            self._core.setROI(left, top, width, height)

    def _grab(self) -> np.ndarray:
        core = self._core
        deadline = time.monotonic() + self.frame_timeout
        while core.getRemainingImageCount() == 0:
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"Micro-Manager camera {self.camera_label}: no frame in {self.frame_timeout} s"
                )
            time.sleep(1e-4)
        image = core.popNextImage()
        while core.getRemainingImageCount() > 0:  # keep only the newest frame
            image = core.popNextImage()
        return np.ascontiguousarray(np.asarray(image).T, dtype=np.uint16)

    def _close_micromanager(self) -> None:
        core = getattr(self, "_core", None)
        if core is None:
            return
        self._core = None
        try:
            if core.isSequenceRunning():
                core.stopSequenceAcquisition()
            core.reset()
            self.logger.info("Closed Micro-Manager camera")
        except Exception:
            self.logger.exception("Failed while closing the Micro-Manager camera")


class MicroManagerWFS(_MicroManagerDevice, WavefrontSensor):
    """Wavefront-sensor camera on any Micro-Manager camera device."""

    EXTRA_CONFIG_KEYS = _MM_KEYS

    def __init__(self, conf):
        super().__init__(conf)
        self._open_micromanager(conf)

    def expose(self):
        self.data = self._grab()
        super().expose()

    def close(self, *args, **kwargs):
        try:
            super().close(*args, **kwargs)
        finally:
            self._close_micromanager()


class MicroManagerScienceCamera(_MicroManagerDevice, ScienceCamera):
    """Science camera on any Micro-Manager camera device."""

    EXTRA_CONFIG_KEYS = _MM_KEYS

    def __init__(self, conf):
        super().__init__(conf)
        self._open_micromanager(conf)

    def expose(self):
        self.data = self._grab()
        super().expose()

    def close(self, *args, **kwargs):
        try:
            super().close(*args, **kwargs)
        finally:
            self._close_micromanager()


if __name__ == "__main__":
    launch_component(MicroManagerWFS, "wfs", start=True)
