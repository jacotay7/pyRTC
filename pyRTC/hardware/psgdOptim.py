from pyRTC.utils import *
from pyRTC.Pipeline import *
from pyRTC.Optimizer import *

import numpy as np

class psgdOptimizer(Optimizer):

    def __init__(self, conf, loop, power) -> None:
        
        self.loop = loop
        self.power = power

        self.gdTime = 5
        self.avgTime = 5
        self.minRate = 1e-2
        self.maxRate = 1e1
        self.minAmp = 1e-1
        self.maxAmp = 1e0
        self.minIntegrate = 1
        self.maxIntegrate = 100
        self.numFlats = 10
        self.relaxTime = 5
        super().__init__(conf)

    def averagePowerForNSeconds(self,N):
        pwr = 0
        n = 0
        start = time.time()
        while time.time()- start < N:
            pwr +=  self.power.readLong()
            n += 1
        pwr /= n
        return pwr
    
    def objective(self, trial):

        # self.loop.useLong = True

        #Average the power
        
        self.loop.stop()

        for i in range(self.numFlats): #Flatten over the course of 3 seconds
            self.loop.flatten()
            time.sleep(self.relaxTime/self.numFlats)
        
        #before = self.averagePowerForNSeconds(self.avgTime)

        self.loop.start()

        #Apply gradient descent for fixzed time
        self.applyTrial(trial)

        #Let gradient descent work for 5 seconds
        time.sleep(self.gdTime)

        #Average power
        after = self.averagePowerForNSeconds(self.avgTime)
        
        #define Success
        return after #- before
    
    def applyTrial(self, trial):
        
        #get a rate
        rate = trial.suggest_float(f'rate', 
                            self.minRate,
                            self.maxRate)
        #get an amplitude
        amp = trial.suggest_float(f'amp', 
                            self.minAmp,
                            self.maxAmp)
        #get other stuff
        integrate = trial.suggest_int(f'intergration', 
                            self.minIntegrate,
                            self.maxIntegrate)
        #configure loop
        self.loop.rate = rate
        self.loop.amp = amp
        self.power.integrationLength = int(integrate)

        return super().applyTrial(trial)


    def applyOptimum(self, overwrite=False):
        super().applyOptimum()

        rate = self.study.best_params['rate'] 
        amp = self.study.best_params['amp'] 
        integrate = self.study.best_params['intergration'] 

        #configure loop
        self.loop.rate = rate
        self.loop.amp = amp
        self.power.integrationLength = integrate

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

    component = psgdOptimizer(conf=conf)
    component.start()

    # Go back to communicating with the main program through stdout
    sys.stdout = original_stdout

    l = Listener(component, port = int(args.port))
    while l.running:
        l.listen()
        time.sleep(1e-3)