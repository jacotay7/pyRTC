"""Thorlabs DMP40 / DMH40 piezo deformable-mirror adapter.

Drives the mirror through the native Linux ``dmp40`` driver package (pyusb;
no Thorlabs DLL). The package is not on PyPI: install it from wherever your
copy lives, or set ``sdk_path`` to the directory that holds it.

The mirror has 40 segment actuators and 3 tip/tilt arms, each driven at
0-200 V. pyrtc controls the 40 segments; commands are volts about a bias,
``segment_volts = bias_v + shape`` (``bias_v`` 100 V by default), so a
``command_cap`` of 89 keeps every segment within 11-189 V. The tilt arms are
held at ``tilt_volts`` and never moved by the loop.

The driver writes complete 43-channel frames and cannot read the tilt arms
back, so ``tilt_volts`` is required: it is sent with every segment command,
and a wrong value moves the mirror's tip/tilt on the first write.

Opening the mirror writes nothing. Unless a ``flat_file`` is given, the flat
is the segment voltages measured at open, so a zero correction holds the
shape the mirror already had. Further config keys:

``serial``
    Mirror serial number (default: the first DMP40 found).
``max_step_v``
    Largest change of any segment in one write (slew limit; default off).
``max_offset_v``
    Largest distance of any segment from its voltage at open (default off).
    Both limits clip the command rather than stopping the loop, and are
    counted in ``limited_actuators`` (the last write) and ``limited_writes``
    (writes with any segment clipped, by these or ``command_cap``). They and ``command_cap`` can be
    tightened while running but not loosened; ``tilt_volts`` and ``bias_v``
    are fixed once the mirror is open.
``on_close``
    ``"restore"`` (default) writes the frame measured at open back when the
    component closes, if it sent anything, in steps of at most
    ``max_step_v``; ``"hold"`` leaves the last command.
    The mirror is never relaxed or zeroed.
``layout_file``
    ``.npy`` layout for the 2D view and modal bases: a boolean mask with 40
    cells (segment ``k`` on the ``k``-th, row-major), or an integer index map
    (``-1`` empty, ``k`` for segment ``k``) when the numbering is not
    row-major, such as the DMP40's rings. There is no built-in layout; without
    one the corrector is 1D.

Besides the corrector's streams, the component publishes ``wfc_volts``
(float64, 42 values): ``[writes, limited_writes, segment volts x 40]``, at
open (the measured frame) and after every write. A client commanding the
mirror through the ``wfc`` stream reads it to confirm its command was
applied unclipped.
"""

from __future__ import annotations

import errno
import importlib
import math
import os
import sys
import threading
import time

import numpy as np

from pyrtc.logging_utils import get_logger
from pyrtc.manager import launch_component
from pyrtc.streams import create_stream
from pyrtc.utils import TightenOnly
from pyrtc.wavefront_corrector import WavefrontCorrector

logger = get_logger(__name__)

NUM_SEGMENTS = 40
NUM_TILT = 3
VOLTAGE_MIN = 0.0
VOLTAGE_MAX = 200.0
# Commands are float32, so a step of exactly max_step_v can come out a few
# tens of nV over it. The DAC resolves ~3.3 mV, so treat anything within this
# tolerance of a limit as on it: not clipped, not counted.
LIMIT_TOLERANCE_V = 1e-6


def _load_dmp40_module(sdk_path=None):
    """Import the ``dmp40`` driver package, adding ``sdk_path`` to ``sys.path``."""

    if sdk_path:
        sdk_path = os.path.abspath(os.path.expanduser(str(sdk_path)))
        if sdk_path not in sys.path:
            sys.path.insert(0, sdk_path)
    try:
        return importlib.import_module("dmp40")
    except ImportError as exc:
        raise ImportError(
            "The DMP40 adapter needs the native 'dmp40' driver package (pyusb); install it "
            "or set 'sdk_path' to the directory that contains it"
        ) from exc


def _volts(values, count, label) -> np.ndarray:
    volts = np.asarray(values, dtype=np.float64).ravel()
    if volts.size != count:
        raise ValueError(f"{label} needs {count} values, got {volts.size}")
    if not np.all(np.isfinite(volts)) or volts.min() < VOLTAGE_MIN or volts.max() > VOLTAGE_MAX:
        raise ValueError(f"{label} must be finite and within {VOLTAGE_MIN}-{VOLTAGE_MAX} V")
    return volts


class ThorlabsDMP40(WavefrontCorrector):
    """Wavefront corrector on a Thorlabs DMP40 piezo deformable mirror."""

    # Safety limits can be tightened while running (e.g. from the manager
    # GUI), never loosened or removed.
    command_cap = TightenOnly()
    max_step_v = TightenOnly()
    max_offset_v = TightenOnly()

    EXTRA_CONFIG_KEYS = (
        "serial",
        "sdk_path",
        "tilt_volts",
        "bias_v",
        "max_step_v",
        "max_offset_v",
        "on_close",
        "layout_file",
    )

    def __init__(self, conf) -> None:
        conf = dict(conf)
        if "tilt_volts" not in conf:
            raise ValueError(
                "tilt_volts is required: the DMP40 cannot read its tilt arms back, and every "
                "write sets them"
            )
        self._tilt_volts = _volts(conf["tilt_volts"], NUM_TILT, "tilt_volts")
        self._bias_v = float(conf.get("bias_v", 100.0))
        if not VOLTAGE_MIN <= self.bias_v <= VOLTAGE_MAX:
            raise ValueError("bias_v must be within 0-200 V")
        self.max_step_v = conf.get("max_step_v")
        self.max_offset_v = conf.get("max_offset_v")
        for name in ("max_step_v", "max_offset_v"):
            value = getattr(self, name)
            if value is not None and not (math.isfinite(float(value)) and float(value) > 0):
                raise ValueError(f"{name} must be positive")
        self.on_close = str(conf.get("on_close", "restore"))
        if self.on_close not in ("restore", "hold"):
            raise ValueError("on_close must be 'restore' or 'hold'")
        if conf.get("num_actuators") != NUM_SEGMENTS:
            logger.info("DMP40 has %s segments; overriding the config", NUM_SEGMENTS)
            conf["num_actuators"] = NUM_SEGMENTS
        conf.setdefault("command_cap", 89.0)  # 11-189 V about the default 100 V bias

        dmp40 = _load_dmp40_module(conf.get("sdk_path"))
        self._dm_lock = threading.Lock()
        self.serial = None if conf.get("serial") is None else str(conf["serial"])
        try:
            self.dm = dmp40.DMP40(serial=self.serial)
        except Exception as exc:
            # The driver blames udev for every open failure; a busy device is
            # simply open in another program, which only one may have.
            if getattr(exc.__cause__, "errno", None) == errno.EBUSY:
                raise RuntimeError(
                    "The DMP40 is open in another program (USB resource busy); close it there first"
                ) from exc
            raise
        self._sent = False
        try:
            self.serial = str(self.dm.get_serial_number())
            self.model = str(self.dm.get_instrument_name())
            # The frame on the mirror at open: the default flat, the reference
            # for max_offset_v, and what on_close="restore" writes back.
            self.baseline_volts = _volts(
                self.dm.get_measured_segment_voltages(), NUM_SEGMENTS, "segment readback"
            )
            self.last_volts = self.baseline_volts.copy()
            self.limited_actuators = 0
            # Writes in which any segment was clipped (by cap, step or offset):
            # a calibration with clipped pokes did not get the shape it asked for.
            self.limited_writes = 0
            self.writes = 0
            self._last_limit_warning = 0.0

            super().__init__(conf)
            if not self.flat_file:
                self.set_flat((self.baseline_volts - self.bias_v).astype(np.float32))
                self.flat_modal = (self.C2M @ self.flat).astype(self.flat.dtype)
                self.set_delay(self.frame_delay)  # refill the delay line with the new flat
            self.current_shape = self.flat.copy()
            self.volts_stream = create_stream(
                self.output_stream_name("wfc_volts"), (NUM_SEGMENTS + 2,), np.float64
            )
            self.register_output_stream("wfc_volts", self.volts_stream)
            self._publish_volts(self.baseline_volts)
            if self.command_cap is not None and np.any(np.abs(self.flat) > self.command_cap):
                # Holding the flat would already be clipped, so the first write
                # would move the mirror.
                raise ValueError(
                    f"the flat spans {self.bias_v + self.flat.min():.2f}-"
                    f"{self.bias_v + self.flat.max():.2f} V, outside command_cap "
                    f"{self.command_cap} V about the {self.bias_v} V bias"
                )
            self.layout_file = conf.get("layout_file", "")
            if self.layout_file:
                self.set_layout(self.generate_layout())
            self.logger.info(
                "Initialized %s serial=%s segments %.2f-%.2f V tilt=%s cap=%s step=%s offset=%s",
                self.model,
                self.serial,
                self.baseline_volts.min(),
                self.baseline_volts.max(),
                self.tilt_volts.tolist(),
                self.command_cap,
                self.max_step_v,
                self.max_offset_v,
            )
        except Exception:
            self._close_dm(restore=False)
            raise

    # Fixed once the mirror is open: every write sends these.

    @property
    def tilt_volts(self) -> np.ndarray:
        return self._tilt_volts.copy()

    @property
    def bias_v(self) -> float:
        return self._bias_v

    def generate_layout(self) -> np.ndarray:
        layout = np.load(self.layout_file)
        if np.issubdtype(layout.dtype, np.integer) and np.any(layout < 0):
            return layout  # an index map; set_layout checks it holds all 40 segments
        layout = layout.astype(bool)
        if int(layout.sum()) != NUM_SEGMENTS:
            raise ValueError(f"layout_file has {int(layout.sum())} actuators, the DMP40 has 40")
        return layout

    def to_volts(self, shape) -> np.ndarray:
        """Map an actuator command to segment volts, applying the safety limits."""

        volts = self.bias_v + np.asarray(shape, dtype=np.float64)
        limited = np.zeros(NUM_SEGMENTS, dtype=bool)
        tol = LIMIT_TOLERANCE_V
        if self.max_offset_v is not None:
            low = self.baseline_volts - float(self.max_offset_v) - tol
            high = self.baseline_volts + float(self.max_offset_v) + tol
            limited |= (volts < low) | (volts > high)
            volts = np.clip(volts, low, high)
        if self.max_step_v is not None:
            low = self.last_volts - float(self.max_step_v) - tol
            high = self.last_volts + float(self.max_step_v) + tol
            limited |= (volts < low) | (volts > high)
            volts = np.clip(volts, low, high)
        self.limited_actuators = int(np.count_nonzero(limited))
        if self.limited_actuators or self.saturated_actuators:
            self.limited_writes += 1
        if self.limited_actuators and time.monotonic() - self._last_limit_warning > 5.0:
            self._last_limit_warning = time.monotonic()
            self.logger.warning(
                "%s of %s segments clipped by max_offset_v=%s / max_step_v=%s",
                self.limited_actuators,
                NUM_SEGMENTS,
                self.max_offset_v,
                self.max_step_v,
            )
        return np.clip(volts, VOLTAGE_MIN, VOLTAGE_MAX)

    def _write_volts(self, volts) -> None:
        with self._dm_lock:
            self.dm.set_voltages(volts.tolist(), self.tilt_volts.tolist())
            self._sent = True
            self.writes += 1
        self.last_volts = volts
        if not getattr(self, "_closed", False):
            self._publish_volts(volts)

    def _publish_volts(self, volts) -> None:
        record = np.empty(NUM_SEGMENTS + 2, dtype=np.float64)
        record[0], record[1] = self.writes, self.limited_writes
        record[2:] = volts
        self.write_stream("wfc_volts", record)

    def send_to_hardware(self):
        super().send_to_hardware()
        volts = self.to_volts(self.current_shape)
        self._write_volts(volts)
        # Report what the mirror was actually sent.
        self.current_shape = (volts - self.bias_v).astype(self.flat.dtype)

    def _ramp_to(self, target) -> int:
        """Write ``target`` volts in steps of at most ``max_step_v``; return the write count."""

        target = np.asarray(target, dtype=np.float64)
        current = np.asarray(self.last_volts, dtype=np.float64).copy()
        step = self.max_step_v
        writes = 0
        while True:
            delta = target - current
            if step is None or np.max(np.abs(delta)) <= step + LIMIT_TOLERANCE_V:
                self._write_volts(target.copy())
                return writes + 1
            current = current + np.clip(delta, -step, step)
            self._write_volts(current.copy())
            writes += 1

    def readback(self) -> np.ndarray:
        """ADC-measured segment voltages."""

        with self._dm_lock:
            return np.asarray(self.dm.get_measured_segment_voltages(), dtype=np.float64)

    def temperatures(self) -> list:
        """``[IC1, IC2, mirror, electronics]`` in degrees C."""

        with self._dm_lock:
            return list(self.dm.get_temperatures())

    def _close_dm(self, restore=True) -> None:
        dm = getattr(self, "dm", None)
        if dm is None:
            return
        component_logger = getattr(self, "logger", logger)
        try:
            if restore and self.on_close == "restore" and self._sent:
                writes = self._ramp_to(self.baseline_volts)
                component_logger.info(
                    "Restored the DMP40 frame measured at open (%s write(s))", writes
                )
        except Exception:
            component_logger.exception("Failed to restore the DMP40 frame measured at open")
        finally:
            self.dm = None
            try:
                dm.close()
            except Exception:
                component_logger.exception("Failed while closing the DMP40")

    def close(self, *args, **kwargs):
        try:
            super().close(*args, **kwargs)
        finally:
            self._close_dm()


if __name__ == "__main__":
    launch_component(ThorlabsDMP40, "wfc", start=True)
