from pyRTC.utils import *
from pyRTC.Modulator import *
from pyRTC.Pipeline import *

import os
import argparse

from pipython import GCSDevice, pitools

class PIModulator(Modulator):

    def __init__(self, conf) -> None:
        #Initialize the pyRTC super class
        super().__init__(conf)

        self.amplitudeX = conf["amplitude"]
        self.frequency = conf["frequency"]
        self.amplitudeY = conf["amplitude"]*conf["relativeAmplitude"]
        self.offsetX = conf["offsetX"]
        self.offsetY = conf["offsetX"]
        self.phaseOffset = conf["phaseOffset"]
        self.sampling = 1/conf["digitalFreq"]

        self.wavegens = (1, 2)
        self.wavetables = (1, 2)

        originalDirectory = os.getcwd()
        os.chdir(conf['libFolder'])
        self.mod = GCSDevice()
        devices = self.mod.EnumerateUSB()
        self.mod.ConnectUSB(devices[0])
        os.chdir(originalDirectory)

        self.servosOn = conf["servosOn"]
        for axis in self.mod.axes:
            self.mod.SVO(axis, int(conf["servosOn"]))

        if conf["autoZero"]:
            self.mod.ATZ()

        self.defineCircle()

        return

    def __del__(self):
        super().__del__()
        
        return    

    def defineCircle(self):
        numPoints = int(1.0 / (self.frequency * self.sampling) )

        # #Define sine and cosine waveforms for wave tables
        self.mod.WAV_SIN_P(table=self.wavetables[0], 
                           firstpoint=0, 
                           numpoints=numPoints, 
                           append='X',
                           center=numPoints // 2, 
                           amplitude=self.amplitudeX, 
                           offset=self.offsetX - self.amplitudeX//2, 
                           seglength=numPoints)
        self.mod.WAV_SIN_P(table=self.wavetables[1], 
                           firstpoint=numPoints // 4 + self.phaseOffset, 
                           numpoints=numPoints, append='X',
                           center=numPoints // 2, 
                           amplitude=self.amplitudeY, 
                           offset=self.offsetY- self.amplitudeY//2, 
                           seglength=numPoints)
        pitools.waitonready(self.mod)

        #Connect wave generators to wave tables 
        if self.mod.HasWSL(): 
            self.mod.WSL(self.wavegens, self.wavetables)

    def start(self):
        super().start()
        #Move axes to their start positions
        startpos = (self.offsetX, self.offsetY + self.amplitudeY // 2)
        self.goTo(startpos)
        
        #Start wave generators {}'.format(self.wavegens))
        self.mod.WGO(self.wavegens, mode=[1] * len(self.wavegens))


    def stop(self):
        super().stop()
        #Reset wave generators
        self.mod.WGO(self.wavegens, mode=[0] * len(self.wavegens))
        return
    
    def goTo(self,x):
        if len(x) < 2 or not self.servosOn:
            return -1
        for i, ax in enumerate(self.mod.axes[:2]):
            self.mod.MOV(ax,int(x[i]))
        pitools.waitontarget(self.mod, self.mod.axes[:2])
        return 1

if __name__ == "__main__":


    #Prevents camera output from messing with communication
    # original_stdout = sys.stdout
    # sys.stdout = open(os.devnull, 'w')

    # Create argument parser
    parser = argparse.ArgumentParser(description="Read a config file from the command line.")

    # Add command-line argument for the config file
    parser.add_argument("-c", "--config", required=True, help="Path to the config file")

    # Parse command-line arguments
    args = parser.parse_args()

    conf = read_yaml_file(args.config)

    pid = os.getpid()
    set_affinity((conf["wfc"]["affinity"])%os.cpu_count()) 
    decrease_nice(pid)

    confMod = conf["modulator"]
    mod = PIModulator(conf=confMod)
    mod.start()
    
    time.sleep(5)
    #Go back to communicating with the main program through stdout
    # sys.stdout = original_stdout

    # l = Listener(wfc)
    # while l.running:
    #     l.listen()
    #     time.sleep(1e-3)