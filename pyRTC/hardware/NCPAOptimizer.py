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

from pyRTC.logging_utils import get_logger
from pyRTC.Optimizer import Optimizer
from pyRTC.Pipeline import Listener, initExistingShm
from pyRTC.utils import decrease_nice, get_tmp_filepath, read_yaml_file, setFromConfig, set_affinity


logger = get_logger(__name__)


def _input_stream_name(conf, stream_name: str) -> str:
    mapping = conf.get("inputStreams", {}) if isinstance(conf.get("inputStreams"), dict) else {}
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
            self.wfcShm, self.wfcDims, self.wfcDtype = initExistingShm(_input_stream_name(conf, "wfc"))
            self.strehlShm, _, _ = initExistingShm(_input_stream_name(conf, "strehl"))
            self.startMode = setFromConfig(conf, "startMode", 0)
            self.endMode = setFromConfig(conf, "endMode", 20)
            self.correctionMag = setFromConfig(conf, "correctionMag", 2e-3)
            self.numReads = setFromConfig(conf, "numReads", 5)
            self.isCL = False
            self.origRefSlopes = None
            self.validSubAps = None
            self.IM = None

            super().__init__(conf)
            self.logger.info(
                "Initialized NCPA optimizer startMode=%s endMode=%s correctionMag=%s numReads=%s",
                self.startMode,
                self.endMode,
                self.correctionMag,
                self.numReads,
            )
        except Exception:
            logger.exception("Failed to initialize NCPA optimizer")
            raise

    def objective(self, trial):
        try:
            self.applyTrial(trial)

            result = np.empty(self.numReads)
            self.strehlShm.read()
            for i in range(self.numReads):
                result[i] = self.strehlShm.read()
            score = np.mean(result)
            self.logger.info("Evaluated NCPA trial score=%s", score)
            return score
        except Exception:
            self.logger.exception("Failed while evaluating NCPA trial")
            raise
    
    def applyTrial(self, trial):
        try:
            modalCoefs = np.zeros(self.wfcDims, dtype=self.wfcDtype)
            for i in range(self.startMode, self.endMode):
                modalCoefs[i] = np.float32(trial.suggest_float(f'{i}', -self.correctionMag, self.correctionMag))
            if self.isCL:
                refSlopesAdjust = np.zeros_like(self.origRefSlopes)
                refSlopesAdjust[self.validSubAps] = self.IM @ modalCoefs
                refSlopes = self.origRefSlopes + refSlopesAdjust
                np.save(self.newRefSlopesFile, refSlopes)
                self.slopes.setProperty("refSlopesFile", self.newRefSlopesFile)
                self.slopes.run("loadRefSlopes")
                self.slopes.setProperty("refSlopesFile", self.refSlopesFile)
                self.logger.info("Applied NCPA trial in closed-loop mode")
            else:
                self.wfcShm.write(modalCoefs)
                self.logger.info("Applied NCPA trial in open-loop mode")
        except Exception:
            self.logger.exception("Failed to apply NCPA trial")
            raise
        return super().applyTrial(trial)

    def applyOptimum(self, overwrite=False):
        try:
            super().applyOptimum()
            modalCoefs = np.zeros(self.wfcDims, dtype=self.wfcDtype)
            for k in self.study.best_params.keys():
                modalCoefs[int(k)] = self.study.best_params[k]

            if self.isCL:
                refSlopesAdjust = np.zeros_like(self.origRefSlopes)
                refSlopesAdjust[self.validSubAps] = self.IM @ modalCoefs
                refSlopes = self.origRefSlopes + refSlopesAdjust
                if overwrite:
                    np.save(self.refSlopesFile, refSlopes)
                    self.slopes.setProperty("refSlopesFile", self.refSlopesFile)
                else:
                    np.save(self.newRefSlopesFile, refSlopes)
                    self.slopes.setProperty("refSlopesFile", self.newRefSlopesFile)

                self.slopes.run("loadRefSlopes")
                self.slopes.setProperty("refSlopesFile", self.refSlopesFile)
                self.logger.info("Applied optimum NCPA correction in closed-loop mode overwrite=%s", overwrite)
            else:
                self.wfcShm.write(modalCoefs)
                self.logger.info("Applied optimum NCPA correction in open-loop mode")
        except Exception:
            self.logger.exception("Failed to apply optimum NCPA correction")
            raise

        return 
    
    def optimize(self):
        try:
            self.refSlopesFile = self.slopes.getProperty("refSlopesFile")
            self.isCL = self.loop.getProperty("running")
            if self.isCL:
                self.validSubAps = np.load(self.slopes.getProperty("validSubApsFile"))
                self.IM = np.load(self.loop.getProperty("IMFile"))

                self.origRefSlopes = np.load(self.refSlopesFile)
                self.newRefSlopesFile = get_tmp_filepath(self.refSlopesFile)

            self.logger.info("Starting NCPA optimization closed_loop=%s", self.isCL)
            super().optimize()
            self.slopes.setProperty("refSlopesFile", self.refSlopesFile)
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