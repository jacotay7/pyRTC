"""
A Superclass for Peformance Optimizer
"""
from pyRTC.Pipeline import *
from pyRTC.utils import *
from pyRTC.pyRTCComponent import *

import argparse
import sys
import os
import time
import optuna

class Optimizer(pyRTCComponent):

    def __init__(self, conf) -> None:
        
        self.name = "Optimizer"
        self.study = optuna.create_study(direction='maximize', 
                                         sampler=optuna.samplers.CmaEsSampler())
        self.numSteps = setFromConfig(conf, "numSteps", 100)

        super().__init__(conf)

        return
    
    def objective(self):
        return

    def optimize(self):
        self.study.optimize(self.objective, 
                            n_trials=self.numSteps)
        self.applyOptimum()
        return
    
    def applyOptimum(self):
        return
    
    def applyTrial(self, trial):
        return
    
    def applyNext(self):
        self.applyTrial(self.study.ask())
        return

    def gradientDescent(self, numSteps, eps, rate):

        #Vector to hold performance
        loss = np.empty(numSteps)
        #Vector to hold current NCPA
        ncpaVec = np.zeros(self.env.num_correction_modes, 
                           dtype=self.loop.IM.dtype)
        
        #Arrays to hold derivatives
        pds = np.zeros((numSteps,self.env.num_correction_modes))
        pd = np.zeros(self.env.num_correction_modes)

        ncpaVecs = np.empty((numSteps, ncpaVec.size))

        for i in range(numSteps):
            
            #Adjust the environmant
            obs = self.env.step(ncpaVec)[0]
            #Adjust loss function for GD
            loss[i] = obs["strehl"][-1]

            #At the beginning of the cycle
            if i % 2 == 0:

                #Make a base measurement of intensity
                base = loss[i]
                poke = np.random.uniform(-eps/base,eps/base,ncpaVec.size).reshape(ncpaVec.shape)
                ncpaVec += poke

            else:

                #Compute parrallel gradient
                pd = (loss[i] - base)*poke
                #Remove phase adjustment of the random parameter
                ncpaVec -= poke

                #change the ncpaVec
                adjust = rate/base*pd
                ncpaVec += adjust

            pds[i] = pd
            ncpaVecs[i] = ncpaVec

        return loss, pds, ncpaVecs

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

    conf = read_yaml_file(args.config)

    pid = os.getpid()
    set_affinity((conf["loop"]["affinity"])%os.cpu_count()) 
    decrease_nice(pid)

    component = Optimizer(conf=conf)
    component.start()

    # Go back to communicating with the main program through stdout
    sys.stdout = original_stdout

    l = Listener(component, port = int(args.port))
    while l.running:
        l.listen()
        time.sleep(1e-3)