"""Boston Micromachines (BMC) deformable-mirror adapter.

Drives BMC MEMS mirrors (Multi-DM, Kilo-DM, 2K, 3K, ...) through the BMC DM
SDK's Python module ``bmc`` on Linux and Windows. Set ``sdk_path`` to the
directory holding the module if it is not importable already.

The SDK takes one value per actuator in ``[0, 1]`` (a fraction of the
driver's range). MEMS mirrors only pull, so pyrtc's bipolar commands are
applied about a bias: ``value = clip(bias + command_scale * command, 0, 1)``,
with ``bias`` 0.5 and ``command_scale`` 0.5 by default (commands in
``[-1, 1]`` span the full range). Use ``command_cap`` to limit the stroke.

The actuator count comes from the SDK. The 2D layout is, in order:
``layout_file`` (a boolean ``.npy`` map), else BMC's standard square
geometries (a square grid, or a square with its four corners removed, as on
the Multi-DM 140 and Kilo-DM 1020), else the smallest centred disk with that
many actuators (the 2K/3K circular apertures). Check it against the vendor's
actuator map.
"""

from __future__ import annotations

import importlib
import math
import os
import sys

import numpy as np

from pyrtc.hardware.alpao_dm import circular_actuator_layout
from pyrtc.logging_utils import get_logger
from pyrtc.manager import launch_component
from pyrtc.wavefront_corrector import WavefrontCorrector

logger = get_logger(__name__)


def _load_bmc_module(sdk_path=None):
    """Import the BMC SDK's ``bmc`` module, adding ``sdk_path`` to ``sys.path``."""

    if sdk_path:
        sdk_path = os.path.abspath(os.path.expanduser(str(sdk_path)))
        if sdk_path not in sys.path:
            sys.path.insert(0, sdk_path)
    try:
        return importlib.import_module("bmc")
    except ImportError as exc:
        raise ImportError(
            "The BMC DM adapter needs the BMC DM SDK's Python module 'bmc'; install the SDK "
            "and set 'sdk_path' to its Python directory"
        ) from exc


def bmc_actuator_layout(num_actuators: int) -> np.ndarray:
    """Return the actuator map of a standard BMC mirror with ``num_actuators``.

    A square grid (``n = s^2``), a square without its four corners
    (``n = s^2 - 4``, e.g. Multi-DM 140 on 12x12, Kilo-DM 1020 on 32x32), or
    otherwise the smallest centred disk holding exactly ``num_actuators``.
    """

    num_actuators = int(num_actuators)
    if num_actuators < 1:
        raise ValueError("num_actuators must be positive")
    side = math.isqrt(num_actuators)
    if side * side == num_actuators:
        return np.ones((side, side), dtype=bool)
    side = math.isqrt(num_actuators + 4)
    if side * side == num_actuators + 4 and side >= 3:
        layout = np.ones((side, side), dtype=bool)
        layout[[0, 0, -1, -1], [0, -1, 0, -1]] = False
        return layout
    return circular_actuator_layout(num_actuators)


class BMCDM(WavefrontCorrector):
    """Wavefront corrector on a Boston Micromachines MEMS deformable mirror."""

    EXTRA_CONFIG_KEYS = ("serial", "sdk_path", "layout_file", "bias", "command_scale")

    def __init__(self, conf) -> None:
        conf = dict(conf)
        self.serial = str(conf["serial"])
        self.bias = float(conf.get("bias", 0.5))
        self.command_scale = float(conf.get("command_scale", 0.5))
        if not 0.0 <= self.bias <= 1.0:
            raise ValueError("bias must be in [0, 1]")
        if self.command_scale <= 0:
            raise ValueError("command_scale must be positive")
        bmc = _load_bmc_module(conf.get("sdk_path"))
        self.dm = bmc.BmcDm()
        error = self.dm.open_dm(self.serial)
        if error:
            raise RuntimeError(f"BMC open_dm({self.serial!r}) failed: {self._error_text(error)}")
        self._dm_open = True
        num_actuators = int(self.dm.num_actuators())
        if conf.get("num_actuators") != num_actuators:
            logger.info(
                "BMC DM %s has %s actuators; overriding the config", self.serial, num_actuators
            )
            conf["num_actuators"] = num_actuators
        self._values = np.empty(num_actuators, dtype=np.float64)
        try:
            super().__init__(conf)
            self.layout_file = conf.get("layout_file", "")
            self.set_layout(self.generate_layout())
            self.flatten()
            self.logger.info(
                "Initialized BMC DM serial=%s actuators=%s bias=%s scale=%s cap=%s",
                self.serial,
                self.num_actuators,
                self.bias,
                self.command_scale,
                self.command_cap,
            )
        except Exception:
            self._close_dm()
            raise

    def _error_text(self, error) -> str:
        describe = getattr(self.dm, "error_string", None)
        try:
            return f"{error} ({describe(error)})" if callable(describe) else str(error)
        except Exception:
            return str(error)

    def generate_layout(self) -> np.ndarray:
        """Return ``layout_file`` if set, else the standard BMC geometry."""

        if self.layout_file:
            layout = np.load(self.layout_file).astype(bool)
            if int(layout.sum()) != self.num_actuators:
                raise ValueError(
                    f"layout_file has {int(layout.sum())} actuators, "
                    f"the mirror reports {self.num_actuators}"
                )
            return layout
        return bmc_actuator_layout(self.num_actuators)

    def to_dm_values(self, shape) -> np.ndarray:
        """Map a bipolar actuator command to the SDK's ``[0, 1]`` range."""

        np.multiply(np.asarray(shape, dtype=np.float64), self.command_scale, out=self._values)
        self._values += self.bias
        np.clip(self._values, 0.0, 1.0, out=self._values)
        return self._values

    def send_to_hardware(self):
        super().send_to_hardware()
        error = self.dm.send_data(self.to_dm_values(self.current_shape))
        if error:
            raise RuntimeError(f"BMC send_data failed: {self._error_text(error)}")

    def _close_dm(self) -> None:
        if not getattr(self, "_dm_open", False):
            return
        self._dm_open = False
        try:
            # Leave the mirror unpowered rather than holding the last shape.
            self.dm.send_data(np.zeros(self._values.size, dtype=np.float64))
            self.dm.close_dm()
            getattr(self, "logger", logger).info("Closed BMC DM %s", self.serial)
        except Exception:
            getattr(self, "logger", logger).exception("Failed while closing the BMC DM")

    def close(self, *args, **kwargs):
        try:
            super().close(*args, **kwargs)
        finally:
            self._close_dm()


if __name__ == "__main__":
    launch_component(BMCDM, "wfc", start=True)
