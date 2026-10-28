"""Allied Vision camera adapters (Alvium, Mako, ...) via Vimba X ``vmbpy``.

``vmbpy`` is Allied Vision's Python API for the Vimba X SDK. Install it with
``pip install pyrtcao[vimba]`` (or the wheel shipped with the SDK) and make
sure a transport layer is available: either set ``cti_path`` to the SDK's
``.cti`` file (e.g. ``VimbaUSBTL.cti``) or directory, or rely on
``GENICAM_GENTL64_PATH``.

The camera streams continuously into ``buffer_count`` driver buffers and
``expose()`` publishes the newest complete frame, so a slow consumer drops old
frames rather than falling behind.

Config keys, on top of the usual WFS or science-camera ones:

``serial`` / ``device_index``
    Which camera to open: by serial number, or the n-th camera found
    (default 0).
``cti_path``
    Transport layer file or directory, or a list of them.
``exposure``, ``gain``, ``bit_depth``, ``pixel_format``, ``binning``, ROI
    Applied through the standard SFNC features: ``exposure`` in µs,
    ``bit_depth`` (8, 10, 12) as ``PixelFormat`` ``MonoN``, and the ROI from
    ``width``, ``height``, ``left`` and ``top``. Auto exposure and auto gain
    are switched off.
``frame_rate``
    Enables the camera's frame-rate limiter at this rate (Hz).
``node_settings``
    Mapping of any other feature names to values, applied in order after the
    above (for example ``{DeviceLinkThroughputLimit: 100000000}`` to share a
    USB hub between cameras).
``fetch_timeout``
    Seconds to wait for a frame (default 1.0).
``buffer_count``
    Driver frame buffers (default 5).
``restore_on_close``
    Put every feature this adapter changed back to its value at open
    (default true), so other software finds the camera as it left it.

Camera frames are ``(Height, Width)``, the shape of pyrtc's image streams
(``[y, x]``), so they are published without transposing. The configured
``width`` and ``height`` must match the camera's frame after the ROI and
binning are applied; the adapter checks this at open.
"""

from __future__ import annotations

import os
import queue
import threading
from typing import Any, Mapping

import numpy as np

from pyrtc.logging_utils import get_logger
from pyrtc.manager import launch_component
from pyrtc.science_camera import ScienceCamera
from pyrtc.utils import require_optional
from pyrtc.wavefront_sensor import WavefrontSensor

logger = get_logger(__name__)

_VIMBA_KEYS = (
    "serial",
    "device_index",
    "cti_path",
    "exposure",
    "gain",
    "bit_depth",
    "pixel_format",
    "binning",
    "left",
    "top",
    "frame_rate",
    "node_settings",
    "fetch_timeout",
    "buffer_count",
    "restore_on_close",
)


# vmbpy's VmbSystem is one object per process: it takes its transport path
# before the first user enters it, and refuses it afterwards. These track the
# Vimba components of this process so cameras can share it.
_SYSTEM_LOCK = threading.Lock()
_SYSTEM_USERS = 0
_SYSTEM_PATHS: list[str] | None = None


def _transport_paths(cti_path) -> list[str]:
    if not cti_path:
        return []
    paths = [cti_path] if isinstance(cti_path, (str, os.PathLike)) else list(cti_path)
    return [os.path.abspath(os.path.expanduser(str(path))) for path in paths]


class _VimbaDevice:
    """Connection, feature access and frame streaming shared by the adapters."""

    def _open_vimba(self, conf: Mapping[str, Any]) -> None:
        vmbpy = require_optional("vmbpy", "vimba", "The Allied Vision Vimba X camera adapter")
        self._vmbpy = vmbpy
        self._original_features: dict[str, Any] = {}
        self._frames: queue.Queue = queue.Queue(maxsize=1)
        self._streaming = False
        self._feature_lock = threading.RLock()
        self.fetch_timeout = float(conf.get("fetch_timeout", 1.0))
        self.buffer_count = int(conf.get("buffer_count", 5))
        self.restore_on_close = bool(conf.get("restore_on_close", True))

        global _SYSTEM_USERS, _SYSTEM_PATHS
        paths = _transport_paths(conf.get("cti_path"))
        with _SYSTEM_LOCK:
            vmb = vmbpy.VmbSystem.get_instance()
            if _SYSTEM_USERS == 0:
                if paths:
                    vmb.set_path_configuration(*paths)
                _SYSTEM_PATHS = paths
            elif paths and paths != _SYSTEM_PATHS:
                raise ValueError(
                    f"cti_path {paths} differs from the one already in use in this process "
                    f"({_SYSTEM_PATHS}); Vimba takes one transport configuration per process"
                )
            vmb.__enter__()
            _SYSTEM_USERS += 1
            self._vmb = vmb
        self._camera = None
        try:
            cameras = list(self._vmb.get_all_cameras())
            if "serial" in conf:
                wanted = str(conf["serial"])
                matches = [cam for cam in cameras if cam.get_serial() == wanted]
                if not matches:
                    found = ", ".join(cam.get_serial() for cam in cameras) or "none"
                    raise RuntimeError(f"No Vimba camera with serial {wanted!r} (found: {found})")
                camera = matches[0]
            else:
                index = int(conf.get("device_index", 0))
                if index >= len(cameras):
                    raise RuntimeError(
                        f"Vimba camera index {index}: {len(cameras)} camera(s) found"
                    )
                camera = cameras[index]
            camera.__enter__()
            self._camera = camera

            self._disable_auto()
            self._free_running()
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
            if "frame_rate" in conf:
                self._set_node("AcquisitionFrameRateEnable", True)
                self._set_node("AcquisitionFrameRate", float(conf["frame_rate"]))
            for name, value in dict(conf.get("node_settings") or {}).items():
                self._set_node(name, value)
            self._check_frame_shape()
            self._start_streaming()
            self._opened = True
        except Exception:
            self._close_vimba()
            raise
        self.logger.info(
            "Opened Vimba camera %s (%s)", self._camera.get_serial(), self._camera.get_model()
        )

    # Features. The first write of each one records its value at open, so
    # close can restore it.

    def _feature(self, name: str):
        try:
            return self._camera.get_feature_by_name(name)
        except Exception as exc:
            raise AttributeError(f"Vimba camera has no feature {name!r}") from exc

    def _set_node(self, name: str, value) -> None:
        if getattr(self, "_camera", None) is None:
            return  # base-class __init__ runs before the device is open; reapplied later
        with self._feature_lock:
            feature = self._feature(name)
            if name not in self._original_features:
                self._original_features[name] = feature.get()
            feature.set(value)
        self.logger.info("Set Vimba feature %s=%s", name, value)

    def _node_value(self, name: str, default=None):
        try:
            return self._feature(name).get()
        except Exception:
            return default

    def _disable_auto(self) -> None:
        for name in ("ExposureAuto", "GainAuto"):
            if self._node_value(name, "Off") != "Off":
                self._set_node(name, "Off")

    def _free_running(self) -> None:
        # Alvium locks TriggerMode while its frame-rate limiter is on, so turn
        # the limiter off first when the trigger needs changing.
        if self._node_value("TriggerMode", "Off") != "Off":
            if self._node_value("AcquisitionFrameRateEnable", False):
                self._set_node("AcquisitionFrameRateEnable", False)
            self._set_node("TriggerMode", "Off")

    def _check_frame_shape(self) -> None:
        shape = (int(self._node_value("Height")), int(self._node_value("Width")))
        if shape != self.data.shape:
            raise ValueError(
                f"Vimba camera frame is (height, width) = {shape}, the config says "
                f"{self.data.shape}; set width/height (and left/top for an ROI)"
            )

    # Streaming

    def _on_frame(self, camera, stream, frame) -> None:
        try:
            if frame.get_status() == self._vmbpy.FrameStatus.Complete:
                image = np.array(frame.as_numpy_ndarray()[..., 0], dtype=np.uint16, copy=True)
                try:
                    self._frames.get_nowait()  # keep only the newest frame
                except queue.Empty:
                    pass
                self._frames.put_nowait(image)
        finally:
            camera.queue_frame(frame)

    def _start_streaming(self) -> None:
        if not self._streaming:
            self._camera.start_streaming(self._on_frame, buffer_count=self.buffer_count)
            self._streaming = True

    def _stop_streaming(self) -> None:
        if self._streaming:
            self._streaming = False
            self._camera.stop_streaming()
        while not self._frames.empty():
            self._frames.get_nowait()

    _GEOMETRY_FEATURES = (
        "BinningHorizontal",
        "BinningVertical",
        "Width",
        "Height",
        "OffsetX",
        "OffsetY",
    )

    def _reconfigure(self, apply) -> None:
        """Apply a payload-changing setting, which needs streaming stopped.

        Once the camera is open, the frame must keep the streams' shape: a
        change that alters it is undone and raises ``ValueError``.
        """
        if getattr(self, "_camera", None) is None:
            return
        streaming = self._streaming
        self._stop_streaming()
        try:
            if not getattr(self, "_opened", False):
                apply()
                return
            before = {name: self._node_value(name) for name in self._GEOMETRY_FEATURES}
            apply()
            try:
                self._check_frame_shape()
            except ValueError:
                # Clear offsets, then size, then offsets, so every step fits the sensor.
                with self._feature_lock:
                    for name in ("OffsetX", "OffsetY"):
                        self._feature(name).set(0)
                    for name in self._GEOMETRY_FEATURES:
                        if before[name] is not None and name not in ("OffsetX", "OffsetY"):
                            self._feature(name).set(before[name])
                    for name in ("OffsetX", "OffsetY"):
                        if before[name] is not None:
                            self._feature(name).set(before[name])
                raise
        finally:
            if streaming:
                self._start_streaming()

    def _grab(self) -> np.ndarray:
        try:
            return self._frames.get(timeout=self.fetch_timeout)
        except queue.Empty:
            raise TimeoutError(f"No Vimba frame within {self.fetch_timeout} s") from None

    # Standard setters: record the value (base class), then apply the feature.

    def set_exposure(self, exposure):
        super().set_exposure(exposure)
        self._set_node("ExposureTime", float(exposure))

    def set_gain(self, gain):
        super().set_gain(gain)
        self._set_node("Gain", float(gain))

    # Geometry setters apply to the camera first and record the value (base
    # class) only once the change has been accepted.

    def set_binning(self, binning):
        self._reconfigure(
            lambda: (
                self._set_node("BinningHorizontal", int(binning)),
                self._set_node("BinningVertical", int(binning)),
            )
        )
        super().set_binning(binning)

    def set_bit_depth(self, bit_depth):
        self._reconfigure(lambda: self._set_node("PixelFormat", f"Mono{int(bit_depth)}"))
        super().set_bit_depth(bit_depth)

    def set_roi(self, roi):
        width, height, left, top = (int(value) for value in roi)

        def apply():
            # Shrink before moving the offset, so every step stays inside the sensor.
            self._set_node("OffsetX", 0)
            self._set_node("OffsetY", 0)
            self._set_node("Width", width)
            self._set_node("Height", height)
            self._set_node("OffsetX", left)
            self._set_node("OffsetY", top)

        self._reconfigure(apply)
        super().set_roi(roi)

    def _restore_features(self) -> None:
        saved = self._original_features
        # Payload size and bandwidth bound the frame-rate range, so they go
        # first; offsets are cleared so the old size fits, and set after it.
        # The limiter locks TriggerMode, so the trigger is restored with the
        # limiter off, and the limiter and its rate after it.
        first = (
            "DeviceLinkThroughputLimit",
            "PixelFormat",
            "BinningHorizontal",
            "BinningVertical",
            "Width",
            "Height",
            "OffsetX",
            "OffsetY",
        )
        last = ("TriggerMode", "AcquisitionFrameRateEnable", "AcquisitionFrameRate")
        steps = [(name, 0) for name in ("OffsetX", "OffsetY") if name in saved]
        if "TriggerMode" in saved:
            steps.append(("AcquisitionFrameRateEnable", False))
        steps += [(name, saved[name]) for name in first if name in saved]
        steps += [(name, value) for name, value in saved.items() if name not in first + last]
        steps += [(name, saved[name]) for name in last if name in saved]
        for name, value in steps:
            try:
                feature = self._feature(name)
                if feature.get() != value:
                    feature.set(value)
            except Exception as exc:
                self.logger.warning("Could not restore Vimba feature %s: %s", name, exc)

    def _close_vimba(self) -> None:
        camera = getattr(self, "_camera", None)
        if camera is not None:
            try:
                self._stop_streaming()
            except Exception:
                self.logger.exception("Failed while stopping the Vimba stream")
            if self.restore_on_close:
                self._restore_features()
            try:
                camera.__exit__(None, None, None)
            except Exception:
                self.logger.exception("Failed while closing the Vimba camera")
            self._camera = None
        global _SYSTEM_USERS
        vmb = getattr(self, "_vmb", None)
        if vmb is not None:
            with _SYSTEM_LOCK:
                try:
                    vmb.__exit__(None, None, None)
                except Exception:
                    self.logger.exception("Failed while closing the Vimba system")
                _SYSTEM_USERS = max(0, _SYSTEM_USERS - 1)
            self._vmb = None


class VimbaWFS(_VimbaDevice, WavefrontSensor):
    """Wavefront-sensor camera on an Allied Vision (Vimba X) device."""

    EXTRA_CONFIG_KEYS = _VIMBA_KEYS

    def __init__(self, conf):
        super().__init__(conf)
        self._open_vimba(conf)

    def expose(self):
        self.data = self._grab()
        super().expose()

    def close(self, *args, **kwargs):
        try:
            super().close(*args, **kwargs)
        finally:
            self._close_vimba()


class VimbaScienceCamera(_VimbaDevice, ScienceCamera):
    """Science camera on an Allied Vision (Vimba X) device."""

    EXTRA_CONFIG_KEYS = _VIMBA_KEYS

    def __init__(self, conf):
        super().__init__(conf)
        self._open_vimba(conf)

    def expose(self):
        self.data = self._grab()
        super().expose()

    def close(self, *args, **kwargs):
        try:
            super().close(*args, **kwargs)
        finally:
            self._close_vimba()


if __name__ == "__main__":
    launch_component(VimbaWFS, "wfs", start=True)
