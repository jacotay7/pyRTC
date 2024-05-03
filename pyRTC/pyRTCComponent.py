"""
Loop Superclass
"""
from pyRTC.Pipeline import *
from pyRTC.utils import *
import threading
import argparse
import sys
import os
import time


class pyRTCComponent:

    def __init__(self, conf) -> None:

        self.alive = True
        self.running = False
        self.affinity = conf["affinity"]

        functionsToRun = setFromConfig(conf, "functions", [])
        self.workThreads = []
        if isinstance(functionsToRun, list) and len(functionsToRun) > 0:
            for i, functionName in enumerate(functionsToRun):
                # Launch a separate thread
                workThread = threading.Thread(target=work, args = (self,functionName), daemon=True)
                # Start the thread
                workThread.start()
                # Set CPU affinity for the thread
                set_affinity((self.affinity+i)%os.cpu_count()) 
                self.workThreads.append(workThread)

        return

    def __del__(self):
        self.stop()
        self.alive=False
        return

    def start(self):
        self.running = True
        return

    def stop(self):
        self.running = False
        return



if __name__ == "__main__":

    #Prevents camera output from messing with communication
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')

    # Create argument parser
    parser = argparse.ArgumentParser(description="Read a config file from the command line.")

    # Add command-line argument for the config file
    parser.add_argument("-c", "--config", required=True, help="Path to the config file")

    # Parse command-line arguments
    args = parser.parse_args()

    conf = read_yaml_file(args.config)

    pid = os.getpid()
    set_affinity((conf["loop"]["affinity"])%os.cpu_count()) 
    decrease_nice(pid)

    component = pyRTCComponent(conf=conf)
    component.start()

    # Go back to communicating with the main program through stdout
    sys.stdout = original_stdout

    # input()

    l = Listener(component)
    while l.running:
        l.listen()
        time.sleep(1e-3)