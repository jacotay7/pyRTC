"""Loop hyperparameter optimizer.

The optimizer in this module tunes coarse loop-level control settings such as
gain, leak, and the number of dropped modes. It is intended for calibration and
commissioning workflows where operators want to search a small parameter space
around a known-stable reconstructor.
"""

import argparse
import os
import sys
import time

import numpy as np

from pyrtc.logging_utils import get_logger
from pyrtc.optimizer import Optimizer
from pyrtc.rpc import Listener
from pyrtc.streams import open_stream
from pyrtc.utils import decrease_nice, read_yaml_file, set_from_config, set_affinity


logger = get_logger(__name__)


def _input_stream_name(conf, stream_name: str) -> str:
    mapping = conf.get("input_streams", {}) if isinstance(conf.get("input_streams"), dict) else {}
    value = mapping.get(stream_name, stream_name)
    if isinstance(value, dict):
        value = value.get("shm", value.get("name", stream_name))
    return str(value)

class LoopOptimizer(Optimizer):
    """Optimizer for high-level AO loop hyperparameters.

    This class evaluates candidate integrator-style loop settings against the
    shared-memory Strehl stream. It is broader than ``PIDOptimizer`` because it
    includes reconstructor-side parameters, such as mode truncation, that require
    reloading the interaction matrix after each trial.
    """

    def __init__(self, conf, loop) -> None:
        try:
            self.loop = loop

            self.strehl_shm = open_stream(_input_stream_name(conf, "strehl"))
            self.min_gain = set_from_config(conf, "min_gain", 0.3)
            self.max_gain = set_from_config(conf, "max_gain", 0.6)
            self.max_leak = set_from_config(conf, "max_leak", 0.1)
            self.max_dropped_modes = set_from_config(conf, "max_dropped_modes", 50)
            self.num_reads = set_from_config(conf, "num_reads", 5)

            super().__init__(conf)
            self.logger.info(
                "Initialized loop optimizer min_gain=%s max_gain=%s max_leak=%s max_dropped_modes=%s num_reads=%s",
                self.min_gain,
                self.max_gain,
                self.max_leak,
                self.max_dropped_modes,
                self.num_reads,
            )
        except Exception:
            logger.exception("Failed to initialize loop optimizer")
            raise

    def objective(self, trial):
        try:
            self.apply_trial(trial)
            self.loop.run("stop")
            for _ in range(10):
                self.loop.run("flatten")
            self.loop.run("start")

            result = np.empty(self.num_reads)
            for i in range(self.num_reads):
                result[i] = self.strehl_shm.read_new()
            score = np.mean(result)
            self.logger.info("Evaluated loop optimizer trial score=%s", score)
            return score
        except Exception:
            self.logger.exception("Failed while evaluating loop optimizer trial")
            raise
    
    def apply_trial(self, trial):
        try:
            self.loop.set_property("num_dropped_modes", trial.suggest_int('num_dropped_modes', 0, self.max_dropped_modes))
            self.loop.set_property("gain", trial.suggest_float('gain', self.min_gain, self.max_gain))
            self.loop.set_property("leaky_gain", trial.suggest_float('leaky_gain', 0, self.max_leak))
            self.loop.run("load_im")
            self.logger.info("Applied loop optimizer trial")
        except Exception:
            self.logger.exception("Failed to apply loop optimizer trial")
            raise

        return super().apply_trial(trial)

    def apply_optimum(self):
        try:
            super().apply_optimum()

            self.loop.set_property("num_dropped_modes", self.study.best_params["num_dropped_modes"])
            self.loop.set_property("gain", self.study.best_params["gain"])
            self.loop.set_property("leaky_gain", self.study.best_params["leaky_gain"])

            self.loop.run("load_im")
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

    component = LoopOptimizer(conf=conf)
    component.start()

    # Go back to communicating with the main program through stdout
    sys.stdout = original_stdout

    listener = Listener(component, port = int(args.port))
    while listener.running:
        listener.listen()
        time.sleep(1e-3)