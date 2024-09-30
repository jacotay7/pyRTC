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
    """
    The Optimizer component for is for general optimization tasks 
    in pyRTC. This class should be used by defining a child class held in pyRTC.hardware, 
    which overwrites the relevant functions which actual hardware connectivity code. 
    The child class can call its parent implementations in order to make use of the code 
    which sets the relevant parameters, write to shared memory, etc... or they can overwrite 
    them completely. See hardware/NCPAOptimizer.py for an example.

    This class is designed to perform optimization using Optuna's 
    CmaEsSampler and manages the optimization process including 
    the application of optimum values and trial management.

    :param conf: Configuration dictionary containing necessary parameters.

    **Config Parameters**:
    - **numSteps** (*int*): The number of steps/trials to perform during optimization. Default is 100.

    Attributes
    ----------
    name : str
        Name of the optimizer component.
    study : optuna.Study
        The study object from Optuna, initialized with a CmaEsSampler.
    numSteps : int
        Number of steps/trials to perform during optimization.

    Methods
    -------
    objective():
        Defines the objective function for the optimization.
    optimize():
        Performs the optimization process.
    applyOptimum():
        Applies the optimum values obtained from the optimization process.
    applyTrial(trial):
        Applies a given trial.
    applyNext():
        Requests and applies the next trial from the study.
    """


    def __init__(self, conf) -> None:
        """
        Initializes the Optimizer with the given configuration.

        :param conf: Configuration dictionary containing necessary parameters.
        """
        self.name = "Optimizer"
        self.study = optuna.create_study(direction='maximize', 
                                         sampler=optuna.samplers.CmaEsSampler())
        self.numSteps = setFromConfig(conf, "numSteps", 100)

        super().__init__(conf)

        return
    
    def objective(self):
        """
        Defines the objective function for the optimization.

        This method should be overridden by subclasses to provide the 
        specific objective function for the optimization task.

        :return: The objective value to be optimized.
        """
        return

    def optimize(self):
        """
        Performs the optimization process.

        This method runs the optimization process using the defined objective 
        function and the number of steps specified in the configuration.
        """
        self.study.optimize(self.objective, 
                            n_trials=self.numSteps)
        self.applyOptimum()
        return
    
    def applyOptimum(self):
        """
        Applies the optimum values obtained from the optimization process.

        This method should be implemented to apply the optimal parameters 
        found during the optimization to the system or component.
        """
        return
    
    def applyTrial(self, trial):
        """
        Applies a given trial.

        :param trial: The trial object containing the parameters to be applied.
        """
        return
    
    def applyNext(self):
        """
        Requests and applies the next trial from the study.

        This method obtains the next trial from the study and applies it 
        using the applyTrial method.
        """
        self.applyTrial(self.study.ask())
        return

    def resetStudy(self):

        self.study = optuna.create_study(direction='maximize', 
                                         sampler=optuna.samplers.CmaEsSampler())

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