"""Base class for threaded pyRTC runtime components.

Most pyRTC subsystems share the same lifecycle model: validate configuration,
normalize optional GPU settings, spawn one or more worker threads from YAML,
and expose lightweight start/stop controls. This module provides that shared
behavior.
"""
import os
import threading

from pyRTC.logging_utils import ensure_logging_configured, get_logger
from pyRTC.Pipeline import launchComponent, normalize_gpu_device, work
from pyRTC.utils import setFromConfig, validate_component_config


logger = get_logger(__name__)


class pyRTCComponent:
    """
    Common threaded component base used throughout pyRTC.

    The base class standardizes the repeated mechanics shared by the wavefront
    sensor, slopes processor, loop controller, telemetry recorder, and many
    hardware-facing helpers. Components list runtime methods under the
    configuration key ``functions`` and the base class starts one worker thread
    per listed method.

    Those worker functions are assumed to matter for their side effects rather
    than their return values. They usually produce, consume, or transform shared-
    memory streams inside the running RTC.

    For examples:

    psf:
        functions:
        - expose
        - integrate

    Config Parameters
    -----------------
    affinity : int
        Base CPU affinity for the component. Additional worker functions are
        assigned subsequent cores when possible.
    functions : list
        Bound method names to run in worker threads.
    gpuDevice : str, optional
        Requested GPU device identifier. When PyTorch is unavailable this is
        normalized back to CPU mode.

    Attributes
    ----------
    alive : bool
        Indicates whether the component is alive.
    running : bool
        Indicates whether the component is currently running.

    The class intentionally does not define component-specific data flow. It is
    only responsible for the shared runtime lifecycle.
    """
    def __init__(self, conf) -> None:
        """
        Constructs all the necessary attributes for the real-time control component object.

        Parameters
        ----------
        conf : dict
            Configuration dictionary for the component. The following keys are used:
            - affinity (int, optional): The CPU affinity for the component. Default 0.
            - functions (list, optional): A list of functions to run in separate threads. Default is an empty list.
        """
        ensure_logging_configured(app_name="pyrtc", component_name=self.__class__.__name__)
        self.logger = get_logger(f"{self.__class__.__module__}.{self.__class__.__name__}")

        try:
            validate_component_config(conf, [cls.__name__ for cls in self.__class__.mro()])

            self.alive = True
            self.running = False
            self.affinity = setFromConfig(conf, "affinity", 0)
            requested_gpu_device = setFromConfig(conf, "gpuDevice", None)
            self.gpuDevice = normalize_gpu_device(requested_gpu_device, self.__class__.__name__)

            functionsToRun = setFromConfig(conf, "functions", [])
            self.workThreads = []
            self.RELEASE_GIL = True

            if isinstance(functionsToRun, list) and len(functionsToRun) > 0:
                for i, functionName in enumerate(functionsToRun):
                    threadAffinity = (self.affinity + i) % os.cpu_count()
                    workThread = threading.Thread(
                        target=work,
                        args=(self, functionName, threadAffinity),
                        daemon=True,
                    )
                    workThread.start()
                    self.workThreads.append(workThread)

            self.logger.info(
                "Initialized component affinity=%s gpuDevice=%s functions=%s",
                self.affinity,
                self.gpuDevice,
                functionsToRun,
            )
        except Exception:
            self.logger.exception("Failed to initialize component")
            raise

        return

    @classmethod
    def describe(cls):
        """Return the nearest built-in component descriptor for this class."""

        from pyRTC.component_descriptors import describe_component_class

        return describe_component_class(cls)

    def __del__(self):
        """
        Destructor to clean up the component.
        """
        component_logger = getattr(self, "logger", logger)
        try:
            self.stop()
        except Exception:
            component_logger.exception("Failed while stopping component during destruction")
        finally:
            self.alive = False
        return

    def start(self):
        """
        Start the registered real-time functions.
        """
        component_logger = getattr(self, "logger", logger)
        try:
            self.running = True
            component_logger.info("Started component")
        except Exception:
            component_logger.exception("Failed to start component")
            raise
        return

    def stop(self):
        """
        Stops the registered real-time functions.
        """
        component_logger = getattr(self, "logger", logger)
        try:
            self.running = False
            component_logger.info("Stopped component")
        except Exception:
            component_logger.exception("Failed to stop component")
            raise
        return

if __name__ == "__main__":

    launchComponent(pyRTCComponent, "component", start = True)