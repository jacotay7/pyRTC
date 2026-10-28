"""Base class for light sources (lasers, LEDs, superluminescent diodes).

A light source sits outside the real-time loop: it has no streams and no
worker functions, and pyrtc never switches it on by itself. The base class
holds the safety rules every source shares; adapters implement the device
calls (:meth:`LightSource._open`, :meth:`_read_status`, :meth:`_set_power`,
:meth:`_set_emission`, :meth:`_close`).

Safety rules:

- ``max_power`` (required, in mW) caps every setpoint, on top of whatever the
  device allows. It can be lowered while running, never raised.
  :meth:`enable` also refuses while the device's own setpoint is above it,
  since another program may have set it.
- Building or starting the component never turns emission on; only
  :meth:`enable` does. If emission is on when the source opens, it is turned
  off.
- :meth:`stop` and :meth:`close` turn emission off and read it back. The
  manager stops and closes the ``source`` section before every other one.
- Every change is read back from the device; a mismatch raises.

To keep a light on while several programs use it, keep the component (and
the system that holds it) running and share its streams, rather than
reopening it.

Config keys:

``max_power``
    Highest allowed setpoint in mW. Required.
``power``
    Setpoint in mW applied when the source opens (emission off). Unset keeps
    the device's setpoint.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

from pyrtc.component import Component
from pyrtc.logging_utils import get_logger
from pyrtc.utils import TightenOnly, set_from_config

logger = get_logger(__name__)


class LightSourceError(RuntimeError):
    """A light source refused a command or did not read back as commanded."""


class LightSource(Component, ABC):
    """Common safety rules and controls for a bench light source."""

    # Lowering the cap while running is allowed; raising it is not.
    max_power = TightenOnly()
    # How far a read-back setpoint may differ from the request (mW, fraction).
    POWER_TOLERANCE_MW = 0.011
    POWER_TOLERANCE_FRACTION = 0.02

    def __init__(self, conf) -> None:
        self._source_open = False
        try:
            super().__init__(conf)
            self.name = set_from_config(conf, "name", "source")
            if "max_power" not in conf:
                raise ValueError("source: 'max_power' (mW) is required")
            self.max_power = _power_value(conf["max_power"], "max_power")
            if self.max_power <= 0:
                raise ValueError("source: 'max_power' must be positive")
            self.last_status: dict = {}

            self._open(conf)
            self._source_open = True
            if self.status()["emission"]:
                self.logger.warning("Emission was on when the source opened; turning it off")
                self.disable()
            self._configure(conf)
            if conf.get("power") is not None:
                self.set_power(conf["power"])
            status = self.status()
            self.logger.info(
                "Initialized light source name=%s setpoint=%s mW max_power=%s mW emission=%s",
                self.name,
                status.get("power"),
                self.max_power,
                status.get("emission"),
            )
        except Exception:
            logger.exception("Failed to initialize light source")
            self._shutdown_source()
            raise

    # Device interface for adapters.

    @abstractmethod
    def _open(self, conf) -> None:
        """Connect to the device. Must not turn emission on."""

    def _configure(self, conf) -> None:
        """Apply device settings at open; emission is off. Optional."""

    @abstractmethod
    def _read_status(self) -> dict:
        """Return at least ``emission`` (bool) and ``power`` (setpoint, mW)."""

    @abstractmethod
    def _set_power(self, power: float) -> None:
        """Set the setpoint in mW (already checked against ``max_power``)."""

    @abstractmethod
    def _set_emission(self, enabled: bool) -> None:
        """Turn emission on or off."""

    @abstractmethod
    def _close(self) -> None:
        """Release the device connection."""

    # Controls

    def status(self) -> dict:
        """Read the device: emission, setpoint and whatever else it reports."""

        self.last_status = dict(self._read_status())
        return self.last_status

    @property
    def emission(self) -> bool:
        return bool(self.status()["emission"])

    @property
    def power(self):
        """Setpoint in mW; assigning it calls :meth:`set_power`."""

        return self.last_status.get("power")

    @power.setter
    def power(self, value) -> None:
        self.set_power(value)

    def set_power(self, power) -> None:
        """Set the setpoint in mW, within ``0..max_power``."""

        power = _power_value(power, "power")
        if not 0.0 <= power <= self.max_power:
            raise ValueError(f"power {power} mW is outside 0-{self.max_power} mW (max_power)")
        self._set_power(power)
        actual = self.status()["power"]
        tolerance = max(self.POWER_TOLERANCE_MW, self.POWER_TOLERANCE_FRACTION * power)
        if actual is None or abs(float(actual) - power) > tolerance:
            raise LightSourceError(
                f"Setpoint reads back {actual} mW after setting {power} mW; check the device"
            )
        self.logger.info("Set light source setpoint to %s mW (read back %s mW)", power, actual)

    def enable(self) -> None:
        """Turn emission on, after checking the setpoint against ``max_power``."""

        setpoint = self.status()["power"]
        if setpoint is None or not 0.0 <= float(setpoint) <= self.max_power:
            raise LightSourceError(
                f"Setpoint {setpoint} mW is outside 0-{self.max_power} mW; set the power first"
            )
        self._set_emission(True)
        if not self.status()["emission"]:
            raise LightSourceError("Emission did not turn on; check the device")
        self.logger.warning("Light source %s emission ON at %s mW", self.name, setpoint)

    def disable(self) -> None:
        """Turn emission off and check that it is off."""

        self._set_emission(False)
        if self.status()["emission"]:
            raise LightSourceError("Emission did not turn off; check the hardware")
        self.logger.info("Light source %s emission off", self.name)

    # Lifecycle: stopping or closing always turns emission off.

    def stop(self):
        try:
            if self._source_open:
                self.disable()
        finally:
            super().stop()

    def _shutdown_source(self) -> None:
        if not getattr(self, "_source_open", False):
            try:
                self._close()
            except Exception:
                pass
            return
        self._source_open = False
        component_logger = getattr(self, "logger", logger)
        try:
            # Write only if needed (a status read stays read-only), but write
            # whenever the state cannot be read.
            try:
                emitting = bool(self._read_status()["emission"])
            except Exception:
                component_logger.exception("Could not read emission while closing; turning it off")
                emitting = True
            if emitting:
                self._set_emission(False)
                if self._read_status()["emission"]:
                    component_logger.error(
                        "Emission still reads ON after closing; check the hardware"
                    )
        except Exception:
            component_logger.exception("Could not confirm emission off while closing")
        finally:
            try:
                self._close()
            except Exception:
                component_logger.exception("Failed while closing the light source")

    def close(self, *args, **kwargs):
        try:
            super().close(*args, **kwargs)
        finally:
            self._shutdown_source()


def _power_value(value, label) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number of mW")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return value
