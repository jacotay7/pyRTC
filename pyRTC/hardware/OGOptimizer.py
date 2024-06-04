from pyRTC.utils import *
from pyRTC.Pipeline import *
from pyRTC.Optimizer import *

import numpy as np

class OGOptimizer(Optimizer):

    def __init__(self, conf, loop, psf) -> None:
        
        self.loop = loop
        self.psf = psf

        self.strehlShm, _, _ = initExistingShm("strehl")
        self.tipTiltShm, _, _ = initExistingShm("tiptilt")
        self.opticalGainShm, self.opticalGainShmDims, _ = initExistingShm("og")

        self.mode = 'exponent'

        self.metric = 'strehl'

        self.startMode = 0
        self.endMode = self.opticalGainShm.read_noblock().size

        super().__init__(conf)

    def objective(self, trial):
        
        #Stop the loop
        self.loop.run("stop")
        #Flatten the mirror
        for i in range(3):
            self.loop.run("flatten")
            time.sleep(1e-3)
        #Change the loop settings
        self.applyTrial(trial)
        #Start the loop again
        self.loop.run("start")

        result = np.empty(self.numReads)
        for i in range(self.numReads):
            if self.metric == 'strehl':
                #Force a new strehl to be ready
                self.psf.run("readStrehl")
                result[i] = self.psf.getProperty("strehl_ratio")
            elif self.metric == 'tiptilt':
                self.psf.run("readTipTilt")
                result[i] = -self.psf.getProperty("peak_dist")
        
        return np.mean(result)
    
    def applyTrial(self, trial):

        if self.mode == 'exponent':
            self.opticalGainShm.write(powerLawOG(self.opticalGainShmDims[0], 
                                                 trial.suggest_float('k', 0, 1)))
        elif self.mode == 'relative':
            ogVec = self.ogVec
            for i in range(self.startMode, self.endMode):
                ogVec[i] *= trial.suggest_float(f'{i}', 0.5, 2)
            self.opticalGainShm.write(ogVec)

        return super().applyTrial(trial)

    def applyOptimum(self):
        super().applyOptimum()
        if self.mode == 'exponent':
            self.opticalGainShm.write(powerLawOG(self.opticalGainShmDims[0], 
                                                 self.study.best_params["k"]))
        elif self.mode == 'relative':
            ogVec = self.ogVec
            for i in range(ogVec.size):
                ogVec[i] *= self.study.best_params[f'{i}']
            self.opticalGainShm.write(ogVec)
        return 
    
    def optimize(self):
        self.ogVec = self.opticalGainShm.read_noblock()
        return super().optimize()
    

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

    component = OGOptimizer(conf=conf)
    component.start()

    # Go back to communicating with the main program through stdout
    sys.stdout = original_stdout

    l = Listener(component, port = int(args.port))
    while l.running:
        l.listen()
        time.sleep(1e-3)