"""Abstract base class for beam and waveform modulators.

Modulators sit outside the steady-state real-time AO loop but still need a
consistent lifecycle and motion interface. This module defines that shared
contract so concrete vendor integrations can expose a uniform API for startup,
shutdown, restarts, and position changes.
"""

from abc import ABC, abstractmethod

from pyRTC.logging_utils import get_logger
from pyRTC.pyRTCComponent import pyRTCComponent
from pyRTC.utils import setFromConfig


logger = get_logger(__name__)

class Modulator(pyRTCComponent, ABC):
    """
    Common lifecycle and positioning interface for modulator devices.

    Concrete subclasses are expected to bind this abstract interface to a real
    hardware controller or simulator. The base class handles configuration and
    logging while subclasses implement the actual move/restart behavior.

    New code should use :meth:`set_position`; :meth:`goTo` is retained as a
    compatibility alias for older call sites.
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
 