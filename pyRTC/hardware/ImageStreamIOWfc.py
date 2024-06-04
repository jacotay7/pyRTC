import os 
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ['NUMBA_NUM_THREADS'] = '1'

from pyRTC.WavefrontCorrector import *
from pyRTC.Pipeline import *
from pyRTC.utils import *
import struct
import argparse
import sys
import ImageStreamIOWrap as ISIO
from scipy.interpolate import griddata

class ISIOWfc(WavefrontCorrector):

    def __init__(self, conf) -> None:
        #Initialize the pyRTC super class
        super().__init__(conf)

        self.isioShmName = conf["isioShmName"]

        self.isioShm = ISIO.Image()
        self.isioShm.open(self.isioShmName)
        self.isioShm.semflush(-1)

        self.layoutSize = 21

        self.count = 0

        #Generate the ALPAO actuator layout for the number of actuators
        layout = self.generateLayout()
        self.setLayout(layout)
        
        #flatten the mirror
        self.flatten()

        # Loading the dictionary from the file
        with open(conf["actMap"], 'r') as file:
            maps_act_map = json.load(file)


        self.x_act_pos = np.zeros(336)
        self.y_act_pos = np.zeros(336)

        for i in range(1,337):
            vals = maps_act_map[str(i)]
            self.x_act_pos[i-1] = vals[0]
            self.y_act_pos[i-1] = vals[1]


        # Create a grid to interpolate onto
        self.grid_x, self.grid_y = np.mgrid[min(self.x_act_pos):max(self.x_act_pos):self.layoutSize*1j, 
                                            min(self.y_act_pos):max(self.y_act_pos):self.layoutSize*1j]
            

        return

    def generateLayout(self):

        # Create a grid of coordinates
        x = np.linspace(-1,1,self.layoutSize)
        y = np.copy(x)
        x, y = np.meshgrid(x, y)

        # Calculate the distance from the center
        distance_from_center = np.sqrt(x**2 + y**2)

        # Create the boolean array
        self.layout = distance_from_center <= 1
        return self.layout
    
    def sendToHardware(self):
        #Read a new modal correction in M2C basis
        self.currentCorrection = self.correctionVector.read()

        self.currentShape = ModaltoZonalWithFlat(self.currentCorrection, 
                                                    self.f_M2C,
                                                    self.flat)
        
        #If we have a 2D SHM instance, update it 
        if isinstance(self.correctionVector2D, ImageSHM) and self.count % 10 == 0:
            # Interpolate the data
            grid_z = griddata((self.x_act_pos, self.y_act_pos), 
                              self.currentShape, 
                              (self.grid_x, self.grid_y), method='nearest')
            self.correctionVector2D.write(grid_z*self.layout)

        self.count += 1
        #Send to ISIO
        self.isioShm.write(self.currentShape.reshape(self.numActuators,1))

        return

    def deactivateActuators(self,actuators):

        #For all of the actuators
        for act in actuators:
            self.actuatorStatus[act] = False
            #Get spatial location of the actuator
            x, y = self.x_act_pos[act], self.y_act_pos[act]
            #Get a gaussian region of influence
            inlfluence_map = np.exp(-((x - self.x_act_pos)**2 + (y - self.x_act_pos)**2) / (2 * self.floatingInfluenceRadius**2))
            inlfluence_map[act] = 0
            inlfluence_map /= np.sum(inlfluence_map)
            #Set a bound on the lowest influence to a tenth of the maximum
            inlfluence_map[inlfluence_map < np.max(inlfluence_map)/10] = 0
            self.floatMatrix[act] = inlfluence_map

        self.setM2C(self.M2C)

        return

    def __del__(self):
        super().__del__()
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
    set_affinity((conf["wfc"]["affinity"])%os.cpu_count()) 
    decrease_nice(pid)

    confWFC = conf["wfc"]
    wfc = ISIOWfc(conf=confWFC)
    wfc.start()

    l = Listener(wfc, port = int(args.port))
    while l.running:
        l.listen()
        time.sleep(1e-3)