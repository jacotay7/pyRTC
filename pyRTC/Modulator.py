from abc import ABC, abstractmethod

from pyRTC.logging_utils import get_logger
from pyRTC.pyRTCComponent import pyRTCComponent
from pyRTC.utils import setFromConfig


logger = get_logger(__name__)

class Modulator(pyRTCComponent, ABC):
    """
    A placeholder class for any modulator specific logic. See hardware/PIModulator for an 
    implementation.
    """
    def __init__(self, conf) -> None:
        try:
            super().__init__(conf)
            self.name = setFromConfig(conf, "name", "modulator")
            self.logger.info("Initialized modulator name=%s", self.name)
        except Exception:
            logger.exception("Failed to initialize modulator")
            raise
        return

    def start(self):
        try:
            self.logger.info("Starting modulator %s", self.name)
            super().start()
        except Exception:
            self.logger.exception("Failed to start modulator %s", getattr(self, "name", "unknown"))
            raise

    def stop(self):
        try:
            self.logger.info("Stopping modulator %s", self.name)
            super().stop()
        except Exception:
            self.logger.exception("Failed to stop modulator %s", getattr(self, "name", "unknown"))
            raise

    def goTo(self, position):
        self.logger.info("goTo called for modulator %s; forwarding to set_position", self.name)
        return self.set_position(position)

    @abstractmethod
    def set_position(self, position):
        """Move the modulator to a requested position."""

    @abstractmethod
    def restart(self):
        """Restart the modulator waveform or hardware state."""
 