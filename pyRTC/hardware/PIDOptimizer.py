from pyRTC.utils import *
from pyRTC.Pipeline import *
from pyRTC.Optimizer import *

import numpy as np

class PIDOptimizer(Optimizer):

    def __init__(self, conf, loop) -> None:
        
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

    def objective(self, trial):
        
        self.applyTrial(trial)
        self.loop.run("stop")
        for i in range(10):
            self.loop.run("flatten")
        self.loop.run("start")

        result = np.empty(self.numReads)
        for i in range(self.numReads):
            if self.mode == 'strehl':
                result[i] = self.strehlShm.read()
            elif self.mode == 'tiptilt':
                result[i] = self.strehlShm.read() -1*self.tipTiltShm.read()
        
        return np.mean(result)
    
    def applyTrial(self, trial):

        # Suggest values for Kp, Ki, Kd
        self.loop.setProperty("pGain", trial.suggest_float('pGain', 0, self.maxPGain))
        self.loop.setProperty("iGain", trial.suggest_float('iGain', 0, self.maxIGain))
        self.loop.setProperty("dGain", trial.suggest_float('dGain', 0, self.maxDGain))

        if self.isPOL:
            self.loop.setProperty("leakyGain", self.loop.getProperty('pGain'))

        return super().applyTrial(trial)

    def applyOptimum(self):
        super().applyOptimum()
        
        self.loop.setProperty("pGain", self.study.best_params["pGain"])
        self.loop.setProperty("iGain", self.study.best_params["iGain"])
        self.loop.setProperty("dGain", self.study.best_params["dGain"])

        if self.isPOL:
            self.loop.setProperty("leakyGain", self.loop.getProperty('pGain'))


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

    l = Listener(component, port = int(args.port))
    while l.running:
        l.listen()
        time.sleep(1e-3)