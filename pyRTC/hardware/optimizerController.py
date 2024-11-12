from pyRTC.utils import *
from pyRTC.Pipeline import *
from pyRTC.Optimizer import *

import numpy as np

class controlOptim(Optimizer):

    def __init__(self, conf, loop, power) -> None:
        
        self.loop = loop
        self.power = power
        self.min = -3
        self.max = 3

        super().__init__(conf)
    
    def objective(self, trial):

        self.applyTrial(trial)
        #define Success
        return self.power.readLong()
    
    def applyTrial(self, trial):
        
        vec = np.empty(self.loop.wfcShape, dtype=self.loop.wfcDType)
        for i in range(vec.size):
            vec[i] = trial.suggest_float(f'{i}', 
                            self.min ,
                            self.max)
        self.loop.wfcShm.write(vec)

        return super().applyTrial(trial)


    def applyOptimum(self):
        super().applyOptimum()

        return 
    
    def optimize(self):
        
        super().optimize()
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

    component = controlOptim(conf=conf)
    component.start()

    # Go back to communicating with the main program through stdout
    sys.stdout = original_stdout

    l = Listener(component, port = int(args.port))
    while l.running:
        l.listen()
        time.sleep(1e-3)