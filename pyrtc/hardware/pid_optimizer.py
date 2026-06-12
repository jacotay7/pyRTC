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


class PIDOptimizer(Optimizer):
    """Optuna-based tuner for PID-style loop gains.

    The optimizer evaluates candidate ``p_gain``, ``i_gain``, and ``d_gain``
    settings by applying them to an existing loop object, restarting the loop,
    and averaging several measurements from shared-memory telemetry. It can also
    mirror the proportional gain into ``leaky_gain`` when operating in a POL-like
    configuration.
    """

    def __init__(self, conf, loop) -> None:
        try:
            self.loop = loop

            self.mode = "strehl"
            self.strehl_shm = open_stream(_input_stream_name(conf, "strehl"))
            self.tip_tilt_shm = open_stream(_input_stream_name(conf, "tiptilt"))
            self.max_p_gain = set_from_config(conf, "max_p_gain", 0.5)
            self.max_i_gain = set_from_config(conf, "max_i_gain", 0.05)
            self.max_d_gain = set_from_config(conf, "max_d_gain", 0.05)
            self.num_reads = set_from_config(conf, "num_reads", 5)
            self.is_pol = False

            super().__init__(conf)
            self.logger.info(
                "Initialized PID optimizer mode=%s num_reads=%s", self.mode, self.num_reads
            )
        except Exception:
            logger.exception("Failed to initialize PID optimizer")
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
                if self.mode == "strehl":
                    result[i] = self.strehl_shm.read_new()
                elif self.mode == "tiptilt":
                    result[i] = self.strehl_shm.read_new() - 1 * self.tip_tilt_shm.read()
            score = np.mean(result)
            self.logger.info("Evaluated PID trial mode=%s score=%s", self.mode, score)
            return score
        except Exception:
            self.logger.exception("Failed while evaluating PID trial")
            raise

    def apply_trial(self, trial):
        try:
            self.loop.set_property("p_gain", trial.suggest_float("p_gain", 0, self.max_p_gain))
            self.loop.set_property("i_gain", trial.suggest_float("i_gain", 0, self.max_i_gain))
            self.loop.set_property("d_gain", trial.suggest_float("d_gain", 0, self.max_d_gain))

            if self.is_pol:
                self.loop.set_property("leaky_gain", self.loop.get_property("p_gain"))
            self.logger.info("Applied PID optimizer trial is_pol=%s", self.is_pol)
        except Exception:
            self.logger.exception("Failed to apply PID optimizer trial")
            raise

        return super().apply_trial(trial)

    def apply_optimum(self):
        try:
            super().apply_optimum()
            self.loop.set_property("p_gain", self.study.best_params["p_gain"])
            self.loop.set_property("i_gain", self.study.best_params["i_gain"])
            self.loop.set_property("d_gain", self.study.best_params["d_gain"])

            if self.is_pol:
                self.loop.set_property("leaky_gain", self.loop.get_property("p_gain"))
            self.logger.info("Applied optimum PID gains is_pol=%s", self.is_pol)
        except Exception:
            self.logger.exception("Failed to apply optimum PID gains")
            raise

        return


if __name__ == "__main__":
    # Prevents camera output from messing with communication
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")

    # Create argument parser
    parser = argparse.ArgumentParser(description="Read a config file from the command line.")

    # Add command-line argument for the config file
    parser.add_argument("-c", "--config", required=True, help="Path to the config file")
    parser.add_argument("-p", "--port", required=True, help="Port for communication")

    # Parse command-line arguments
    args = parser.parse_args()

    conf = read_yaml_file(args.config)["optimizer"]

    pid = os.getpid()
    set_affinity((conf["affinity"]) % os.cpu_count())
    decrease_nice(pid)

    component = PIDOptimizer(conf=conf)
    component.start()

    # Go back to communicating with the main program through stdout
    sys.stdout = original_stdout

    listener = Listener(component, port=int(args.port))
    while listener.running:
        listener.listen()
        time.sleep(1e-3)
