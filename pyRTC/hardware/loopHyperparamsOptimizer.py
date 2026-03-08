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

class loopOptimizer(Optimizer):

    def __init__(self, conf, loop) -> None:
        try:
            self.loop = loop

            self.strehlShm, _, _ = initExistingShm("strehl")
            self.minGain = setFromConfig(conf, "minGain", 0.3)
            self.maxGain = setFromConfig(conf, "maxGain", 0.6)
            self.maxLeak = setFromConfig(conf, "maxLeak", 0.1)
            self.maxDroppedModes = setFromConfig(conf, "maxDroppedModes", 50)
            self.numReads = setFromConfig(conf, "numReads", 5)

            super().__init__(conf)
            self.logger.info(
                "Initialized loop optimizer minGain=%s maxGain=%s maxLeak=%s maxDroppedModes=%s numReads=%s",
                self.minGain,
                self.maxGain,
                self.maxLeak,
                self.maxDroppedModes,
                self.numReads,
            )
        except Exception:
            logger.exception("Failed to initialize loop optimizer")
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
                result[i] = self.strehlShm.read()
            score = np.mean(result)
            self.logger.info("Evaluated loop optimizer trial score=%s", score)
            return score
        except Exception:
            self.logger.exception("Failed while evaluating loop optimizer trial")
            raise
    
    def applyTrial(self, trial):
        try:
            self.loop.setProperty("numDroppedModes", trial.suggest_int('numDroppedModes', 0, self.maxDroppedModes))
            self.loop.setProperty("gain", trial.suggest_float('gain', self.minGain, self.maxGain))
            self.loop.setProperty("leakyGain", trial.suggest_float('leakyGain', 0, self.maxLeak))
            self.loop.run("loadIM")
            self.logger.info("Applied loop optimizer trial")
        except Exception:
            self.logger.exception("Failed to apply loop optimizer trial")
            raise

        return super().applyTrial(trial)

    def applyOptimum(self):
        try:
            super().applyOptimum()

            self.loop.setProperty("numDroppedModes", self.study.best_params["numDroppedModes"])
            self.loop.setProperty("gain", self.study.best_params["gain"])
            self.loop.setProperty("leakyGain", self.study.best_params["leakyGain"])

            self.loop.run("loadIM")
            self.logger.info("Applied optimum loop hyperparameters")
        except Exception:
            self.logger.exception("Failed to apply optimum loop hyperparameters")
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

    component = loopOptimizer(conf=conf)
    component.start()

    # Go back to communicating with the main program through stdout
    sys.stdout = original_stdout

    listener = Listener(component, port = int(args.port))
    while listener.running:
        listener.listen()
        time.sleep(1e-3)