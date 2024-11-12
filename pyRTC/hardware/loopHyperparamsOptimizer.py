from pyRTC.utils import *
from pyRTC.Pipeline import *
from pyRTC.Optimizer import *

import numpy as np

class loopOptimizer(Optimizer):

    def __init__(self, conf, loop) -> None:
        
        self.loop = loop

        self.strehlShm, _, _ = initExistingShm("strehl")
        # self.minGain = setFromConfig(conf, "minGain", 0.3)
        # self.maxGain = setFromConfig(conf, "maxGain", 0.6)
        # self.maxLeak = setFromConfig(conf, "maxLeak", 0.1)
        # self.maxDroppedModes = setFromConfig(conf, "maxDroppedModes", 50)
        self.numReads = setFromConfig(conf, "numReads", 5)

        self.params  = setFromConfig(conf, "params", [])
        self.mins = setFromConfig(conf, "mins", [])
        self.maxs = setFromConfig(conf, "maxs", [])
        types = setFromConfig(conf, "types", [])
        self.types = []
        for i in range(len(types)):
            if types[i] == "float":
                self.types.append(float)
            elif types[i] == "int":
                self.types.append(int)
            else:
                raise Exception("Unsupported Type Given: {types[i]}")
            
        assert(len(self.params) == len(self.mins))
        assert(len(self.params) == len(self.maxs))
        assert(len(self.params) == len(self.types))

        self.checkValidFunc = None
        # for i, param in enumerate(self.params):
        #     self.mins = self.types[i](self.mins[i])
        #     self.maxs = self.types[i](self.maxs[i])

        super().__init__(conf)

    def registerParam(self, param, min_, max_, type=float):
        self.params.append(param)
        self.mins.append(min_)
        self.maxs.append(max_)
        self.types.append(type)
        return

    def adjustParam(self, param, min_, max_, type=float):

        if param not in self.params:
            self.registerParam(param, min_, max_, type=type)
            return
        idx = self.params.index(param)
        self.mins[idx] = min_
        self.maxs[idx] = max_
        self.types[idx] = type
        return

    def objective(self, trial):
        
        self.loop.run("stop")
        for i in range(10):
            self.loop.run("flatten")
        self.applyTrial(trial)
        self.loop.run("start")

        result = np.empty(self.numReads)
        for i in range(self.numReads):
            result[i] = self.strehlShm.read()
            if self.checkValidFunc is not None: 
                if not self.checkValidFunc():
                    return np.nan
        return np.mean(result)
    
    def applyTrial(self, trial):

        for i, param in enumerate(self.params):
            if self.types[i] == float:
                self.loop.setProperty(param, trial.suggest_float(param, self.mins[i], self.maxs[i]))
            elif self.types[i] == int:
                self.loop.setProperty(param, trial.suggest_int(param, self.mins[i], self.maxs[i]))

        self.loop.run("loadIM")

        return super().applyTrial(trial)

    def applyOptimum(self):
        super().applyOptimum()
        best_trial_id = 0 #max(self.study.trials[1:], key=lambda t: t.value)
        best_value = -1e8
        for i, t in enumerate(self.study.trials):
            if i == 0 or t.values is None:
                continue
            if t.values[0] > best_value:
                best_value = t.values[0]
                best_trial_id = i
        
        for i, param in enumerate(self.params):
            self.loop.setProperty(param, self.study.trials[best_trial_id].params[param])
        self.loop.run("stop")
        for i in range(10):
            self.loop.run("flatten")
        self.loop.run("loadIM")
        self.loop.run("start")
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