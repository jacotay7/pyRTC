"""ALPAO deformable-mirror adapter.

This module exposes a pyrtc-compatible wavefront-corrector implementation for
ALPAO mirrors driven through the vendor SDK. The adapter translates pyrtc modal
or zonal correction vectors into the actuator command format expected by the
device and centralizes mirror-specific initialization such as layout discovery,
command clipping, and optional floating-actuator masking.
"""

import importlib
import os
import struct
import sys

import numpy as np

from pyrtc.logging_utils import get_logger
from pyrtc.manager import launch_component
from pyrtc.wavefront_corrector import WavefrontCorrector


logger = get_logger(__name__)


def _load_alpao_dm_class(sdk_path=None):
    """Import the ALPAO SDK ``DM`` class (``Lib/asdk`` or ``Lib64/asdk``).

    ``sdk_path`` is the directory containing the SDK's ``Lib``/``Lib64``
    package; it is added to ``sys.path`` when given. The SDK prints on import,
    which would corrupt the hard-RTC stdout protocol, so stdout is silenced.
    """
    if sdk_path:
        sdk_path = os.path.abspath(os.path.expanduser(str(sdk_path)))
        if sdk_path not in sys.path:
            sys.path.insert(0, sdk_path)
    package = "Lib" if (8 * struct.calcsize("P")) == 32 else "Lib64"
    original_stdout = sys.stdout
    with open(os.devnull, "w") as devnull:
        sys.stdout = devnull
        try:
            return importlib.import_module(f"{package}.asdk").DM
        finally:
            sys.stdout = original_stdout


def circular_actuator_layout(num_actuators: int) -> np.ndarray:
    """Return the smallest centred-disk actuator mask with ``num_actuators``.

    ALPAO mirrors place actuators on a square grid inside a circular pupil;
    the DM97 layout is the 11x11 grid within radius 5.5 of the centre. The
    mask returned is the smallest square grid whose actuators within some
    radius of the grid centre number exactly ``num_actuators``. Check it
    against the vendor's actuator map, and use ``layout_file`` when the real
    layout differs (e.g. non-circular or offset grids).
    """
    num_actuators = int(num_actuators)
    if num_actuators < 1:
        raise ValueError("num_actuators must be positive")
    for size in range(1, 257):
        centre = (size - 1) / 2.0
        rows, cols = np.indices((size, size))
        distance_sq = (rows - centre) ** 2 + (cols - centre) ** 2
        ordered = np.sort(distance_sq.ravel())
        thresholds = np.unique(ordered)
        counts = np.searchsorted(ordered, thresholds, side="right")
        matches = np.nonzero(counts == num_actuators)[0]
        if matches.size:
            return distance_sq <= thresholds[matches[0]]
    raise ValueError(f"No circular layout with {num_actuators} actuators; provide layout_file")


class ALPAODM(WavefrontCorrector):
    """Wavefront-corrector adapter for an ALPAO deformable mirror.

    The class wraps the ALPAO SDK object and presents it through the standard
    ``WavefrontCorrector`` interface used by the rest of pyrtc. It is
    responsible for discovering the mirror geometry, applying safety limits to
    outgoing commands, handling optional floating-actuator masks, and resetting
    the device on teardown.
    """

    EXTRA_CONFIG_KEYS = ("serial", "floating_actuators_file", "layout_file", "sdk_path")

    def __init__(self, conf) -> None:
        try:
            super().__init__(conf)

            self.serial = conf["serial"]
            self.layout_file = conf.get("layout_file", "")
            DM = _load_alpao_dm_class(conf.get("sdk_path"))
            self.dm = DM(self.serial)
            self.CAP = self.command_cap
            self.num_actuators = int(self.dm.Get("NBOfActuator"))

            layout = self.generate_layout()
            self.set_layout(layout)

            floating_file = conf.get("floating_actuators_file", "")
            if floating_file.endswith(".npy"):
                float_actuator_inds = np.load(floating_file)
                self.deactivate_actuators(float_actuator_inds)
                self.logger.info("Loaded floating actuators from %s", floating_file)

            self.flatten()
            self.logger.info(
                "Initialized ALPAO DM serial=%s actuators=%s cap=%s",
                self.serial,
                self.num_actuators,
                self.command_cap,
            )
        except Exception:
            logger.exception("Failed to initialize ALPAO DM")
            raise

        return

    def generate_layout(self):
        """Return the actuator layout: ``layout_file`` if set, else a disk."""
        try:
            if self.layout_file:
                layout = np.load(self.layout_file).astype(bool)
                if int(np.count_nonzero(layout)) != self.num_actuators:
                    raise ValueError(
                        f"layout_file has {int(np.count_nonzero(layout))} actuators, "
                        f"the mirror reports {self.num_actuators}"
                    )
            else:
                layout = circular_actuator_layout(self.num_actuators)
            self.logger.info("Using ALPAO %s-actuator layout %s", self.num_actuators, layout.shape)
            return layout
        except Exception:
            self.logger.exception(
                "Failed to generate ALPAO layout for actuators=%s",
                getattr(self, "num_actuators", None),
            )
            raise

    def send_to_hardware(self):
        # Do all of the normal updating of the super class
        super().send_to_hardware()
        # Send the correction to the actual mirror
        self.dm.Send(self.current_shape)
        return

    def __del__(self):
        component_logger = getattr(self, "logger", logger)
        try:
            super().__del__()
        finally:
            dm = getattr(self, "dm", None)
            if dm is not None:
                try:
                    dm.Reset()
                    component_logger.info("Reset ALPAO DM")
                except Exception:
                    component_logger.exception("Failed while resetting ALPAO DM")
        return


if __name__ == "__main__":
    launch_component(ALPAODM, "wfc", start=True)
