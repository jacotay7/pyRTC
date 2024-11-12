"""
pyRTC Component Superclass
"""
from pyRTC.Pipeline import *
from pyRTC.utils import *
import threading
import argparse
import sys
import os
import time


class pyRTCComponent:
    """
    A base class for real-time control components.

    This class provides a framework for real-time control components, allowing for the 
    management of threads and CPU affinity settings. You can register a function to the 
    real-time pipeline in the config by including their name under the key "functions". These
    function will then be spawned into their own thread and controlled by the start and stop
    functions. Note: any return value from registered functions is not used or stored.

    For examples:

    psf:
        functions:
        - expose
        - integrate

    Config Parameters
    -----------------
    affinity : int
        The CPU affinity for the component. Default is 0.
    functions : list
        A list of functions to run in separate threads. Default is an empty list.

    Attributes
    ----------
    alive : bool
        Indicates whether the component is alive.
    running : bool
        Indicates whether the component is currently running.

    Methods
    -------
    start():
        Start the registered real-time functions.
    stop():
        Stop the registered real-time functions.
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
        self.alive = True
        self.running = False
        self.affinity = setFromConfig(conf, "affinity", 0)
        self.gpuDevice = setFromConfig(conf, "gpuDevice", None)

        # if self.gpuDevice is not None:
        #     self.gpuDevice = torch.device(self.gpuDevice)
        
        functionsToRun = setFromConfig(conf, "functions", [])
        self.workThreads = []
        self.RELEASE_GIL = True
        
        if isinstance(functionsToRun, list) and len(functionsToRun) > 0:
            for i, functionName in enumerate(functionsToRun):
                threadAffinity = (self.affinity+i)%os.cpu_count()
                # Launch a separate thread
                workThread = threading.Thread(target=work, 
                                            args = (self,functionName, threadAffinity), 
                                            daemon=True)
                # Start the thread
                workThread.start()
                self.workThreads.append(workThread)

        return

    def __del__(self):
        """
        Destructor to clean up the component.
        """
        self.stop()
        self.alive = False
        return

    def start(self):
        """
        Start the registered real-time functions.
        """
        self.running = True
        return

    def stop(self):
        """
        Stops the registered real-time functions.
        """
        self.running = False
        return

if __name__ == "__main__":

    launchComponent(pyRTCComponent, "component", start = True)