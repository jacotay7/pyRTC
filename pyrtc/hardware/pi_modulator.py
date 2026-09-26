"""PI tip-tilt modulation stage adapter.

This module implements the pyrtc ``Modulator`` interface for Physik
Instrumente stages driven through ``pipython``. The adapter configures a pair
of PI wave generators to trace a circular modulation pattern that can be used to
drive a pyramid wavefront sensor.
"""

import os

from pipython import GCSDevice, pitools

from pyrtc.modulator import Modulator
from pyrtc.manager import launch_component
from pyrtc.utils import set_from_config


class PIModulator(Modulator):
    """Hardware modulator for a two-axis PI motion stage.

    The class converts pyrtc modulation parameters into PI wave-table and
    wave-generator commands. It encapsulates USB device discovery, servo setup,
    optional auto-zeroing, circle generation, and the start/stop lifecycle used
    when the modulator is run as a standalone component.
    """

    def __init__(self, conf) -> None:
        try:
            super().__init__(conf)

            self.amplitude_x = conf["amplitude"]
            self.relative_amp = set_from_config(conf, "relative_amplitude", 1.0)
            self.frequency = conf["frequency"]
            self.amplitude_y = conf["amplitude"] * self.relative_amp
            self.offset_x = conf["offset_x"]
            self.offset_y = conf["offset_x"]
            self.phase_offset = conf["phase_offset"]
            self.sampling = 1 / conf["digital_freq"]

            self.wavegens = (1, 2)
            self.wavetables = (1, 2)

            original_directory = os.getcwd()
            try:
                os.chdir(conf["lib_folder"])
                self.mod = GCSDevice()
                devices = self.mod.EnumerateUSB()
                if not devices:
                    raise RuntimeError("No PI modulator USB devices detected")
                self.mod.ConnectUSB(devices[0])
                self.logger.info("Connected to PI modulator device %s", devices[0])
            finally:
                os.chdir(original_directory)

            self.servos_on = conf["servos_on"]
            for axis in self.mod.axes:
                self.mod.SVO(axis, int(conf["servos_on"]))
            self.logger.info(
                "Servo state set to %s for axes %s", self.servos_on, tuple(self.mod.axes)
            )

            if conf["auto_zero"]:
                self.logger.info("Auto-zeroing PI modulator")
                self.mod.ATZ()

            try:
                self.define_circle()
            except Exception:
                self.logger.exception(
                    "Failed to define modulation circle on first attempt; retrying after stop"
                )
                self.stop()
                self.define_circle()
        except Exception:
            self.logger.exception("Failed to initialize PI modulator")
            raise

        return

    def __del__(self):
        self.logger.info("Destroying PI modulator")
        super().__del__()

        return

    def define_circle(self):
        try:
            num_points = int(1.0 / (self.frequency * self.sampling))
            self.logger.info(
                "Defining modulation circle points=%s amplitude_x=%s amplitude_y=%s phase_offset=%s",
                num_points,
                self.amplitude_x,
                self.amplitude_y,
                self.phase_offset,
            )

            self.mod.WAV_SIN_P(
                table=self.wavetables[0],
                firstpoint=0,
                numpoints=num_points,
                append="X",
                center=num_points // 2,
                amplitude=self.amplitude_x,
                offset=self.offset_x - self.amplitude_x // 2,
                seglength=num_points,
            )
            self.mod.WAV_SIN_P(
                table=self.wavetables[1],
                firstpoint=num_points // 4 + self.phase_offset,
                numpoints=num_points,
                append="X",
                center=num_points // 2,
                amplitude=self.amplitude_y,
                offset=self.offset_y - self.amplitude_y // 2,
                seglength=num_points,
            )
            pitools.waitonready(self.mod)

            if self.mod.HasWSL():
                self.mod.WSL(self.wavegens, self.wavetables)
            self.logger.info("Defined modulation circle and linked wave tables")
        except Exception:
            self.logger.exception("Failed to define PI modulation circle")
            raise

    def start(self):
        try:
            super().start()
            startpos = (self.offset_x, self.offset_y + self.amplitude_y // 2)
            self.logger.info("Moving modulator to start position %s", startpos)
            self.set_position(startpos)
            self.mod.WGO(self.wavegens, mode=[1] * len(self.wavegens))
            self.logger.info("Started PI modulator wave generators %s", self.wavegens)
        except Exception:
            self.logger.exception("Failed to start PI modulator")
            raise
        return

    def stop(self):
        try:
            super().stop()
            self.mod.WGO(self.wavegens, mode=[0] * len(self.wavegens))
            self.logger.info("Stopped PI modulator wave generators %s", self.wavegens)
        except Exception:
            self.logger.exception("Failed to stop PI modulator")
            raise
        return

    def set_position(self, position):
        try:
            self.logger.info("Setting PI modulator position to %s", position)
            if len(position) < 2:
                raise ValueError("Position must contain at least two axis values")
            if not self.servos_on:
                self.logger.warning("Ignoring set_position because servos are disabled")
                return -1
            for i, ax in enumerate(self.mod.axes[:2]):
                self.mod.MOV(ax, int(position[i]))
            pitools.waitontarget(self.mod, self.mod.axes[:2])
            self.logger.info("PI modulator reached position %s", position)
            return 1
        except Exception:
            self.logger.exception("Failed to set PI modulator position to %s", position)
            raise

    def go_to(self, x):
        return super().go_to(x)

    def adjust_amp(self, amp, restart=True):
        try:
            self.logger.info("Adjusting PI modulator amplitude to %s restart=%s", amp, restart)
            self.amplitude_x = amp
            self.amplitude_y = amp * self.relative_amp
            if restart:
                self.restart()
        except Exception:
            self.logger.exception("Failed to adjust PI modulator amplitude to %s", amp)
            raise
        return

    def restart(self):
        try:
            self.logger.info("Restarting PI modulator")
            self.stop()
            self.define_circle()
            self.start()
        except Exception:
            self.logger.exception("Failed to restart PI modulator")
            raise


if __name__ == "__main__":
    launch_component(PIModulator, "modulator", start=True)
