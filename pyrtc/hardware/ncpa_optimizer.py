"""Non-common-path aberration optimizer.

This module contains an optimizer that searches for modal corrections which
maximize a science-camera quality metric, typically Strehl ratio. It supports
both open-loop correction by writing directly to the wavefront-corrector shared
memory and closed-loop correction by perturbing reference slopes.
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
from pyrtc.utils import decrease_nice, get_tmp_filepath, read_yaml_file, set_from_config, set_affinity


logger = get_logger(__name__)


def _input_stream_name(conf, stream_name: str) -> str:
    mapping = conf.get("input_streams", {}) if isinstance(conf.get("input_streams"), dict) else {}
    value = mapping.get(stream_name, stream_name)
    if isinstance(value, dict):
        value = value.get("shm", value.get("name", stream_name))
    return str(value)

class NCPAOptimizer(Optimizer):
    """Optimizer that searches modal NCPA corrections.

    The class explores a configurable modal range and evaluates each trial using
    science-camera telemetry. In open-loop mode it writes correction vectors
    directly to the deformable-mirror command stream; in closed-loop mode it
    synthesizes updated reference slopes so the existing reconstructor absorbs
    the NCPA compensation.
    """

    def __init__(self, conf, loop, slopes) -> None:
        try:
            self.loop = loop
            self.slopes = slopes
            self.wfc_shm = open_stream(_input_stream_name(conf, "wfc"))
            self.wfc_dims = tuple(self.wfc_shm.shape)
            self.wfc_dtype = np.dtype(self.wfc_shm.dtype)
            self.strehl_shm = open_stream(_input_stream_name(conf, "strehl"))
            self.start_mode = set_from_config(conf, "start_mode", 0)
            self.end_mode = set_from_config(conf, "end_mode", 20)
            self.correction_mag = set_from_config(conf, "correction_mag", 2e-3)
            self.num_reads = set_from_config(conf, "num_reads", 5)
            self.is_cl = False
            self.orig_ref_slopes = None
            self.valid_sub_aps = None
            self.im = None

            super().__init__(conf)
            self.logger.info(
                "Initialized NCPA optimizer start_mode=%s end_mode=%s correction_mag=%s num_reads=%s",
                self.start_mode,
                self.end_mode,
                self.correction_mag,
                self.num_reads,
            )
        except Exception:
            logger.exception("Failed to initialize NCPA optimizer")
            raise

    def objective(self, trial):
        try:
            self.apply_trial(trial)

            result = np.empty(self.num_reads)
            for i in range(self.num_reads):
                result[i] = self.strehl_shm.read_new()
            score = np.mean(result)
            self.logger.info("Evaluated NCPA trial score=%s", score)
            return score
        except Exception:
            self.logger.exception("Failed while evaluating NCPA trial")
            raise
    
    def apply_trial(self, trial):
        try:
            modal_coefs = np.zeros(self.wfc_dims, dtype=self.wfc_dtype)
            for i in range(self.start_mode, self.end_mode):
                modal_coefs[i] = np.float32(trial.suggest_float(f'{i}', -self.correction_mag, self.correction_mag))
            if self.is_cl:
                ref_slopes_adjust = np.zeros_like(self.orig_ref_slopes)
                ref_slopes_adjust[self.valid_sub_aps] = self.im @ modal_coefs
                ref_slopes = self.orig_ref_slopes + ref_slopes_adjust
                np.save(self.new_ref_slopes_file, ref_slopes)
                self.slopes.set_property("ref_slopes_file", self.new_ref_slopes_file)
                self.slopes.run("load_ref_slopes")
                self.slopes.set_property("ref_slopes_file", self.ref_slopes_file)
                self.logger.info("Applied NCPA trial in closed-loop mode")
            else:
                self.wfc_shm.write(modal_coefs)
                self.logger.info("Applied NCPA trial in open-loop mode")
        except Exception:
            self.logger.exception("Failed to apply NCPA trial")
            raise
        return super().apply_trial(trial)

    def apply_optimum(self, overwrite=False):
        try:
            super().apply_optimum()
            modal_coefs = np.zeros(self.wfc_dims, dtype=self.wfc_dtype)
            for k in self.study.best_params.keys():
                modal_coefs[int(k)] = self.study.best_params[k]

            if self.is_cl:
                ref_slopes_adjust = np.zeros_like(self.orig_ref_slopes)
                ref_slopes_adjust[self.valid_sub_aps] = self.im @ modal_coefs
                ref_slopes = self.orig_ref_slopes + ref_slopes_adjust
                if overwrite:
                    np.save(self.ref_slopes_file, ref_slopes)
                    self.slopes.set_property("ref_slopes_file", self.ref_slopes_file)
                else:
                    np.save(self.new_ref_slopes_file, ref_slopes)
                    self.slopes.set_property("ref_slopes_file", self.new_ref_slopes_file)

                self.slopes.run("load_ref_slopes")
                self.slopes.set_property("ref_slopes_file", self.ref_slopes_file)
                self.logger.info("Applied optimum NCPA correction in closed-loop mode overwrite=%s", overwrite)
            else:
                self.wfc_shm.write(modal_coefs)
                self.logger.info("Applied optimum NCPA correction in open-loop mode")
        except Exception:
            self.logger.exception("Failed to apply optimum NCPA correction")
            raise

        return 
    
    def optimize(self):
        try:
            self.ref_slopes_file = self.slopes.get_property("ref_slopes_file")
            self.is_cl = self.loop.get_property("running")
            if self.is_cl:
                self.valid_sub_aps = np.load(self.slopes.get_property("valid_sub_aps_file"))
                self.im = np.load(self.loop.get_property("im_file"))

                self.orig_ref_slopes = np.load(self.ref_slopes_file)
                self.new_ref_slopes_file = get_tmp_filepath(self.ref_slopes_file)

            self.logger.info("Starting NCPA optimization closed_loop=%s", self.is_cl)
            super().optimize()
            self.slopes.set_property("ref_slopes_file", self.ref_slopes_file)
            self.logger.info("Completed NCPA optimization")
        except Exception:
            self.logger.exception("Failed during NCPA optimization")
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

    component = NCPAOptimizer(conf=conf)
    component.start()

    # Go back to communicating with the main program through stdout
    sys.stdout = original_stdout

    listener = Listener(component, port = int(args.port))
    while listener.running:
        listener.listen()
        time.sleep(1e-3)