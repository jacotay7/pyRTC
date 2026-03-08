"""PID gain optimizer for a running control loop.

The optimizer in this module tunes proportional, integral, and derivative loop
gains against a live performance metric exposed through shared memory. It is a
slow control-plane tool rather than a real-time component and is intended for
supervised commissioning or laboratory retuning.
"""

import argparse
import os
import sys
import time

import numpy as np

from pyRTC.logging_utils import get_logger
from pyRTC.Optimizer import Optimizer
from pyRTC.Pipeline import Listener, initExistingShm
from pyRTC.utils import decrease_nice, read_yaml_file, setFromConfig, set_affinity


logger = get_logger(__name__)

class PIDOptimizer(Optimizer):
    """Optuna-based tuner for PID-style loop gains.

    The optimizer evaluates candidate ``pGain``, ``iGain``, and ``dGain``
    settings by applying them to an existing loop object, restarting the loop,
    and averaging several measurements from shared-memory telemetry. It can also
    mirror the proportional gain into ``leakyGain`` when operating in a POL-like
    configuration.
    """

    def __init__(self, conf, loop) -> None:
        try:
            self.loop = loop

            self.mode = 'strehl'
            self.strehlShm, _, _ = initExistingShm("strehl")
            self.tipTiltShm, _, _ = initExistingShm("tiptilt")
            self.maxPGain = setFromConfig(conf, "maxPGain", 0.5)
            self.maxIGain = setFromConfig(conf, "maxIGain", 0.05)
            self.maxDGain = setFromConfig(conf, "maxDGain", 0.05)
            self.numReads = setFromConfig(conf, "numReads", 5)
            self.isPOL = False

            super().__init__(conf)
            self.logger.info("Initialized PID optimizer mode=%s numReads=%s", self.mode, self.numReads)
        except Exception:
            logger.exception("Failed to initialize PID optimizer")
            raise

    def objective(self, trial):
        try:
            self.applyTrial(trial)
            self.loop.run("stop")
            for _ in range(10):
                self.loop.run("flatten")
            self.loop.run("start")

            result = np.empty(self.numReads)
            for i in range(self.numReads):
                if self.mode == 'strehl':
                    result[i] = self.strehlShm.read()
                elif self.mode == 'tiptilt':
                    result[i] = self.strehlShm.read() - 1 * self.tipTiltShm.read()
            score = np.mean(result)
            self.logger.info("Evaluated PID trial mode=%s score=%s", self.mode, score)
            return score
        except Exception:
            self.logger.exception("Failed while evaluating PID trial")
            raise
    
    def applyTrial(self, trial):
        try:
            self.loop.setProperty("pGain", trial.suggest_float('pGain', 0, self.maxPGain))
            self.loop.setProperty("iGain", trial.suggest_float('iGain', 0, self.maxIGain))
            self.loop.setProperty("dGain", trial.suggest_float('dGain', 0, self.maxDGain))

            if self.isPOL:
                self.loop.setProperty("leakyGain", self.loop.getProperty('pGain'))
            self.logger.info("Applied PID optimizer trial isPOL=%s", self.isPOL)
        except Exception:
            self.logger.exception("Failed to apply PID optimizer trial")
            raise

        return super().applyTrial(trial)

    def applyOptimum(self):
        try:
            super().applyOptimum()
            self.loop.setProperty("pGain", self.study.best_params["pGain"])
            self.loop.setProperty("iGain", self.study.best_params["iGain"])
            self.loop.setProperty("dGain", self.study.best_params["dGain"])

            if self.isPOL:
                self.loop.setProperty("leakyGain", self.loop.getProperty('pGain'))
            self.logger.info("Applied optimum PID gains isPOL=%s", self.isPOL)
        except Exception:
            self.logger.exception("Failed to apply optimum PID gains")
            raise


        return 
    

if __name__ == "__main__":

    #Prevents camera output from messing with communication
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')

    # Create argument parser
    parser = argparse.ArgumentParser(description="Read a config file from the command line.")

    # Add command-line argument for the config file
    parser.add_argument("-c", "--config", required=True, help="Path to the config file")
    parser.add_argument("-p", "--port", required=True, help="Port for communication")

    # Parse command-line arguments
    args = parser.parse_args()

    conf = read_yaml_file(args.config)["optimizer"]

    pid = os.getpid()
    set_affinity((conf["affinity"])%os.cpu_count()) 
    decrease_nice(pid)

    component = PIDOptimizer(conf=conf)
    component.start()

    # Go back to communicating with the main program through stdout
    sys.stdout = original_stdout

    listener = Listener(component, port = int(args.port))
    while listener.running:
        listener.listen()
        time.sleep(1e-3)