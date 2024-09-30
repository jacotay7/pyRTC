from pyRTC.utils import *
from pyRTC.Pipeline import *
from pyRTC.Optimizer import *

import numpy as np

class loopOptimizer(Optimizer):

    def __init__(self, conf, loop) -> None:
        
        self.loop = loop

        self.strehlShm, _, _ = initExistingShm("strehl")
        self.minGain = setFromConfig(conf, "minGain", 0.3)
        self.maxGain = setFromConfig(conf, "maxGain", 0.6)
        self.maxLeak = setFromConfig(conf, "maxLeak", 0.1)
        self.maxDroppedModes = setFromConfig(conf, "maxDroppedModes", 50)
        self.numReads = setFromConfig(conf, "numReads", 5)

        super().__init__(conf)

    def objective(self, trial):
        
        self.applyTrial(trial)
        self.loop.run("stop")
        for i in range(10):
            self.loop.run("flatten")
        self.loop.run("start")

        result = np.empty(self.numReads)
        for i in range(self.numReads):
            result[i] = self.strehlShm.read()
        return np.mean(result)
    
    def applyTrial(self, trial):
        self.loop.setProperty("numDroppedModes", trial.suggest_int('numDroppedModes', 0, self.maxDroppedModes))
        self.loop.setProperty("gain",trial.suggest_float('gain', self.minGain, self.maxGain))
        self.loop.setProperty("leakyGain", trial.suggest_float('leakyGain', 0, self.maxLeak))
        self.loop.run("loadIM")

        return super().applyTrial(trial)

    def applyOptimum(self):
        super().applyOptimum()
        
        self.loop.setProperty("numDroppedModes", self.study.best_params["numDroppedModes"])
        self.loop.setProperty("gain", self.study.best_params["gain"])
        self.loop.setProperty("leakyGain", self.study.best_params["leakyGain"])

        self.loop.run("loadIM")
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

    l = Listener(component, port = int(args.port))
    while l.running:
        l.listen()
        time.sleep(1e-3)