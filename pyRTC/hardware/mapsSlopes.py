from pyRTC.utils import *
from pyRTC.Pipeline import *
from pyRTC.SlopesProcess import *

import numpy as np

class mapsSlopes(SlopesProcess):

    def __init__(self, conf) -> None:
        
        self.slopeMaskFile = setFromConfig(conf["slopes"], "slopeMaskFile", "")

        super().__init__(conf)


        return

    
    def setPupils(self, pupilLocs, pupilRadius):
        self.pupilLocs = pupilLocs
        self.pupilRadius = pupilRadius
        self.computePupilsMask()
        if self.signalType == "slopes":
            self.signalSize = np.count_nonzero(self.pupilMask)//2
            slopemask =  self.pupilMask[self.pupilLocs[0][1]-self.pupilRadius+1:self.pupilLocs[0][1]+self.pupilRadius, 
                                        self.pupilLocs[0][0]-self.pupilRadius+1:self.pupilLocs[0][0]+self.pupilRadius] > 0
            self.setValidSubAps(np.concatenate([slopemask, slopemask], axis=1))
            self.signal = ImageSHM("signal", (self.signalSize,), self.signalDType)
            self.signal2D = ImageSHM("signal2D", (self.validSubAps.shape[0], self.validSubAps.shape[1]), self.signalDType)
            
        return
    
    def computePupilsMask(self):

        self.pupilMask = np.zeros(self.imageShape)

        pupilTemplate = np.load(self.slopeMaskFile).astype(bool)       
        N = self.pupilMask .shape[0]
        n = pupilTemplate.shape[0]
        # Calculate the half size of the template
        half_n = n // 2

        for i, pupil_loc in enumerate(self.pupilLocs):
            px, py = pupil_loc

            # Determine the bounds of the subimage
            x_start = px - half_n
            x_end = px + half_n + (n % 2)
            y_start = py - half_n
            y_end = py + half_n + (n % 2)
            
            # Ensure the subimage bounds are within the bounds of the larger array
            if x_start < 0 or y_start < 0 or x_end > N or y_end > N:
                raise ValueError("The subimage exceeds the bounds of the larger array.")

            self.pupilMask[y_start:y_end, x_start:x_end] += pupilTemplate*(i+1)

        self.p1mask = self.pupilMask == 1
        self.p2mask = self.pupilMask == 2
        self.p3mask = self.pupilMask == 3
        self.p4mask = self.pupilMask == 4
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
    set_affinity(conf["slopes"]["affinity"]%os.cpu_count())
    decrease_nice(pid)

    slopes = mapsSlopes(conf=conf)
    slopes.start()

    l = Listener(slopes, port= int(args.port))
    while l.running:
        l.listen()
        time.sleep(1e-3)