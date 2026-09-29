"""Generic GenICam camera adapters (GigE Vision / USB3 Vision) via Harvesters.

`Harvesters <https://github.com/genicam/harvesters>`_ is the GenICam
consumer library. Through a vendor's GenTL producer (a ``.cti`` file), one
adapter drives Basler, Allied Vision, FLIR/Teledyne, IDS, Baumer and other
machine-vision cameras. Install with ``pip install pyrtcao[genicam]`` and
point ``cti_file`` at the vendor's producer. Without ``cti_file``, the
producers on ``GENICAM_GENTL64_PATH`` are used.

Config keys, on top of the usual WFS or science-camera ones:

``cti_file``
    GenTL producer path, or a list of paths.
``serial`` / ``device_index``
    Which camera to open: by serial number, or the n-th camera found
    (default 0).
``exposure`` (µs), ``gain``, ``bit_depth`` (8, 10, 12, 14 or 16, mapped to
``MonoN``), ``pixel_format``, ``binning``, and ROI (``width``, ``height``,
``left``, ``top``)
    Applied through the standard GenICam SFNC nodes.
``node_settings``
    Mapping of any other node names to values, applied in order after the
    above (for example ``{AcquisitionFrameRateEnable: true,
    AcquisitionFrameRate: 500}``).
``fetch_timeout``
    Seconds to wait for a frame (default 1.0).

pyrtc image streams have shape ``(width, height)``; camera frames are
``(Height, Width)`` (rows, columns), so frames are transposed. ``width`` and
``height`` then mean the same as the camera's ``Width``/``Height`` nodes.
"""

from __future__ import annotations

import os
from typing import Any, Mapping

import numpy as np

from pyrtc.logging_utils import get_logger
from pyrtc.manager import launch_component
from pyrtc.science_camera import ScienceCamera
from pyrtc.utils import require_optional
from pyrtc.wavefront_sensor import WavefrontSensor

logger = get_logger(__name__)

_GENICAM_KEYS = (
    "cti_file",
    "serial",
    "device_index",
    "exposure",
    "gain",
    "bit_depth",
    "pixel_format",
    "binning",
    "left",
    "top",
    "node_settings",
    "fetch_timeout",
)


def _producer_files(cti_file) -> list[str]:
    if cti_file:
        files = [cti_file] if isinstance(cti_file, str) else list(cti_file)
    else:
        files = []
        for directory in os.environ.get("GENICAM_GENTL64_PATH", "").split(os.pathsep):
            if directory and os.path.isdir(directory):
                files += [
                    os.path.join(directory, name)
                    for name in sorted(os.listdir(directory))
                    if name.endswith(".cti")
                ]
    if not files:
        raise RuntimeError(
            "GenICam camera: no GenTL producer; set 'cti_file' or GENICAM_GENTL64_PATH"
        )
    return files


class _GenICamDevice:
    """Connection, node access and frame grabbing shared by the adapters."""

    def _open_genicam(self, conf: Mapping[str, Any]) -> None:
        harvesters = require_optional("harvesters.core", "genicam", "The GenICam camera adapter")
        self._harvester = harvesters.Harvester()
        for path in _producer_files(conf.get("cti_file")):
            self._harvester.add_file(path)
        self._harvester.update()
        if "serial" in conf:
            key = {"serial_number": str(conf["serial"])}
        else:
            key = int(conf.get("device_index", 0))
        create = getattr(self._harvester, "create", None) or self._harvester.create_image_acquirer
        self._acquirer = create(key)
        self._node_map = self._acquirer.remote_device.node_map
        self.fetch_timeout = float(conf.get("fetch_timeout", 1.0))

        if "pixel_format" in conf:
            self._set_node("PixelFormat", conf["pixel_format"])
        if "bit_depth" in conf:
            self.set_bit_depth(conf["bit_depth"])
        if "binning" in conf:
            self.set_binning(conf["binning"])
        if all(key in conf for key in ("width", "height", "left", "top")):
            self.set_roi([conf["width"], conf["height"], conf["left"], conf["top"]])
        if "exposure" in conf:
            self.set_exposure(conf["exposure"])
        if "gain" in conf:
            self.set_gain(conf["gain"])
        for name, value in dict(conf.get("node_settings") or {}).items():
            self._set_node(name, value)
        self._acquirer.start()
        self.logger.info(
            "Opened GenICam camera %s (%s)",
            key,
            self._node_value("DeviceModelName", "unknown model"),
        )

    def _node(self, name: str):
        node = getattr(self._node_map, name, None)
        if node is None:
            raise AttributeError(f"GenICam camera has no node {name!r}")
        return node

    def _set_node(self, name: str, value) -> None:
        if getattr(self, "_node_map", None) is None:
            return  # base-class __init__ runs before the device is open; reapplied later
        self._node(name).value = value
        self.logger.info("Set GenICam node %s=%s", name, value)

    def _node_value(self, name: str, default=None):
        node = getattr(getattr(self, "_node_map", None), name, None)
        return default if node is None else node.value

    def _grab(self) -> np.ndarray:
        with self._acquirer.fetch(timeout=self.fetch_timeout) as buffer:
            component = buffer.payload.components[0]
            image = np.asarray(component.data).reshape(component.height, component.width)
            # (Height, Width) -> pyrtc's (width, height). The buffer returns to
            # the producer on exit, so copy.
            return np.ascontiguousarray(image.T, dtype=np.uint16)

    # Standard setters: record the value (base class), then apply the node.

    def set_exposure(self, exposure):
        super().set_exposure(exposure)
        self._set_node("ExposureTime", float(exposure))

    def set_gain(self, gain):
        super().set_gain(gain)
        self._set_node("Gain", float(gain))

    def set_binning(self, binning):
        super().set_binning(binning)
        self._set_node("BinningHorizontal", int(binning))
        self._set_node("BinningVertical", int(binning))

    def set_bit_depth(self, bit_depth):
        super().set_bit_depth(bit_depth)
        self._set_node("PixelFormat", f"Mono{int(bit_depth)}")

    def set_roi(self, roi):
        super().set_roi(roi)
        width, height, left, top = (int(value) for value in roi)
        # Shrink before moving the offset, so every step stays inside the sensor.
        self._set_node("OffsetX", 0)
        self._set_node("OffsetY", 0)
        self._set_node("Width", width)
        self._set_node("Height", height)
        self._set_node("OffsetX", left)
        self._set_node("OffsetY", top)

    def _close_genicam(self) -> None:
        acquirer = getattr(self, "_acquirer", None)
        if acquirer is not None:
            try:
                acquirer.stop()
                acquirer.destroy()
            except Exception:
                self.logger.exception("Failed while closing the GenICam camera")
            self._acquirer = None
        harvester = getattr(self, "_harvester", None)
        if harvester is not None:
            harvester.reset()
            self._harvester = None


class GenICamWFS(_GenICamDevice, WavefrontSensor):
    """Wavefront-sensor camera on any GenICam (GigE / USB3 Vision) device."""

    EXTRA_CONFIG_KEYS = _GENICAM_KEYS

    def __init__(self, conf):
        super().__init__(conf)
        self._open_genicam(conf)

    def expose(self):
        self.data = self._grab()
        super().expose()

    def close(self, *args, **kwargs):
        try:
            super().close(*args, **kwargs)
        finally:
            self._close_genicam()


class GenICamScienceCamera(_GenICamDevice, ScienceCamera):
    """Science camera on any GenICam (GigE / USB3 Vision) device."""

    EXTRA_CONFIG_KEYS = _GENICAM_KEYS

    def __init__(self, conf):
        super().__init__(conf)
        self._open_genicam(conf)

    def expose(self):
        self.data = self._grab()
        super().expose()

    def close(self, *args, **kwargs):
        try:
            super().close(*args, **kwargs)
        finally:
            self._close_genicam()


if __name__ == "__main__":
    launch_component(GenICamWFS, "wfs", start=True)
