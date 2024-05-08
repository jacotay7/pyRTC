from pyRTC.utils import *
from pyRTC.Pipeline import *
from pyRTC.Optimizer import *

import numpy as np

class NCAPOptimizer(Optimizer):

    def __init__(self, conf) -> None:
        
        self.wfcShm, self.wfcDims, self.wfcDtype = initExistingShm("wfc")
        self.strehlShm, _, _ = initExistingShm("strehl")
        self.startMode = setFromConfig(conf, "startMode", 0)
        self.endMode = setFromConfig(conf, "endMode", 20)
        self.correctionMag = setFromConfig(conf, "correctionMag", 2e-3)
        self.numReads = setFromConfig(conf, "numReads", 5)

        super().__init__(conf)

    def objective(self, trial):

        numModesCorrect = self.endMode - self.startMode
        modalCoefs = np.zeros(self.wfcDims, dtype=self.wfcDtype)
        for i in range(self.startMode,numModesCorrect):
            modalCoefs[i] = np.float32(trial.suggest_float(f'{i}', 
                                                           -self.correctionMag,
                                                            self.correctionMag))

        self.wfcShm.write(modalCoefs)

        result = np.empty(self.numReads)
        for i in range(self.numReads):
            result[i] = self.strehlShm.read()
        return np.mean(result)
    
    def applyOptimum(self):
        super().applyOptimum()
        modalCoefs = np.zeros(self.wfcDims, dtype=self.wfcDtype)
        for k in self.study.best_params.keys():
            modalCoefs[int(k)] = self.study.best_params[k]
        self.wfcShm.write(modalCoefs)
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

    component = NCAPOptimizer(conf=conf)
    component.start()

    # Go back to communicating with the main program through stdout
    sys.stdout = original_stdout

    l = Listener(component, port = int(args.port))
    while l.running:
        l.listen()
        time.sleep(1e-3)