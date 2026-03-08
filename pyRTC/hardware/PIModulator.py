import os

from pipython import GCSDevice, pitools

from pyRTC.Modulator import Modulator
from pyRTC.Pipeline import launchComponent
from pyRTC.utils import setFromConfig

class PIModulator(Modulator):

    def __init__(self, conf) -> None:
        try:
            super().__init__(conf)

            self.amplitudeX = conf["amplitude"]
            self.relativeAmp = setFromConfig(conf, "relativeAmplitude", 1.0)
            self.frequency = conf["frequency"]
            self.amplitudeY = conf["amplitude"] * self.relativeAmp
            self.offsetX = conf["offsetX"]
            self.offsetY = conf["offsetX"]
            self.phaseOffset = conf["phaseOffset"]
            self.sampling = 1 / conf["digitalFreq"]

            self.wavegens = (1, 2)
            self.wavetables = (1, 2)

            originalDirectory = os.getcwd()
            try:
                os.chdir(conf["libFolder"])
                self.mod = GCSDevice()
                devices = self.mod.EnumerateUSB()
                if not devices:
                    raise RuntimeError("No PI modulator USB devices detected")
                self.mod.ConnectUSB(devices[0])
                self.logger.info("Connected to PI modulator device %s", devices[0])
            finally:
                os.chdir(originalDirectory)

            self.servosOn = conf["servosOn"]
            for axis in self.mod.axes:
                self.mod.SVO(axis, int(conf["servosOn"]))
            self.logger.info("Servo state set to %s for axes %s", self.servosOn, tuple(self.mod.axes))

            if conf["autoZero"]:
                self.logger.info("Auto-zeroing PI modulator")
                self.mod.ATZ()

            try:
                self.defineCircle()
            except Exception:
                self.logger.exception("Failed to define modulation circle on first attempt; retrying after stop")
                self.stop()
                self.defineCircle()
        except Exception:
            self.logger.exception("Failed to initialize PI modulator")
            raise

        return

    def __del__(self):
        self.logger.info("Destroying PI modulator")
        super().__del__()
        
        return    

    def defineCircle(self):
        try:
            numPoints = int(1.0 / (self.frequency * self.sampling))
            self.logger.info(
                "Defining modulation circle points=%s amplitudeX=%s amplitudeY=%s phaseOffset=%s",
                numPoints,
                self.amplitudeX,
                self.amplitudeY,
                self.phaseOffset,
            )

            self.mod.WAV_SIN_P(
                table=self.wavetables[0],
                firstpoint=0,
                numpoints=numPoints,
                append='X',
                center=numPoints // 2,
                amplitude=self.amplitudeX,
                offset=self.offsetX - self.amplitudeX // 2,
                seglength=numPoints,
            )
            self.mod.WAV_SIN_P(
                table=self.wavetables[1],
                firstpoint=numPoints // 4 + self.phaseOffset,
                numpoints=numPoints,
                append='X',
                center=numPoints // 2,
                amplitude=self.amplitudeY,
                offset=self.offsetY - self.amplitudeY // 2,
                seglength=numPoints,
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
            startpos = (self.offsetX, self.offsetY + self.amplitudeY // 2)
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
            if not self.servosOn:
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

    def goTo(self, x):
        return super().goTo(x)
    
    def adjustAmp(self, amp, restart=True):
        try:
            self.logger.info("Adjusting PI modulator amplitude to %s restart=%s", amp, restart)
            self.amplitudeX = amp
            self.amplitudeY = amp * self.relativeAmp
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
            self.defineCircle()
            self.start()
        except Exception:
            self.logger.exception("Failed to restart PI modulator")
            raise

if __name__ == "__main__":

    launchComponent(PIModulator, "modulator", start = True)