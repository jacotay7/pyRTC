from pyRTC.Loop import *
from pyRTC.Pipeline import *
from pyRTC.utils import *
import argparse


class RLLoop(Loop):

    def __init__(self, conf):
        #Parent class is Loop, has the IM, etc...
        super().__init__(conf)

        #Load Config variables

            #Save Directory
            #....

        #Initialize Environment


        return

    def __del__(self):
        #Add anythng special for destroying the object
        return super().__del__()

    def runWarmupEpisode(self, numIters):

        return
    
    def runPolicy(self):

        #Run one inference, to be pinned to the start/stop function
        return
    
    def runTrainingEpisode(self, numIters):

        return
    
    def warmUpTrain(self):

        return
    
    def compareBaselineToPolicy(self):

        return
    
    def saveModels(self, tag=""):

        #Allow for saving with unique tag

        return
    
    def loadModels(self):

        return
        
    def loadLastModel(self):

        return

    def loadFromTag(self):

        return
    





    
if __name__ == "__main__":

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

    loop = RLLoop(conf=conf)
    
    l = Listener(loop, port= int(args.port))
    while l.running:
        l.listen()
        time.sleep(1e-3)