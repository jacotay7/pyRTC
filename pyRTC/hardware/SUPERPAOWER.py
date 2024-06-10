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
import argparse
import struct
import time
import serial

class SUPERPAOWER(WavefrontCorrector):

    def __init__(self, conf) -> None:
        #Initialize the pyRTC super class
        super().__init__(conf)

        #Initialize connection to ALPAO DM
        self.serialPort = setFromConfig(conf,"serialPort","/dev/tty0USB")
        self.baudRate = setFromConfig(conf,"baudRate",115200)
        self.voltRange = setFromConfig(conf, "voltRange", [0.0,4.1])

        self.communicationPause = setFromConfig(conf, "communicationPause", 0.05) #seconds

        #Ask for the number of actuators
        self.numActuators =  conf["numActuators"]

        #Generate the ALPAO actuator layout for the number of actuators
        layout = self.generateLayout()
        self.setLayout(layout)

        #Read the flat from the specified flat file
        if "flatFile" in conf.keys():
            if '.txt' in conf["flatFile"]:
                flat = np.genfromtxt(conf["flatFile"])
            elif '.npy' in conf["flatFile"]:
                flat = np.load(conf["flatFile"])
            self.setFlat(flat.astype(self.flat.dtype))
        
        #flatten the mirror
        self.flatten()
        self.device = None

        self.connectToChip()

        self.setMapping()

        return

    def generateLayout(self):
        
        actPerSide = int(np.round(np.sqrt(self.numActuators),0))
        layout = np.ones(self.numActuators).reshape(actPerSide,actPerSide).astype(bool)

        return layout
    
    def connectToChip(self):
        # self.device = serial.Serial(self.serialPort, self.baudRate)
        # Open the serial port with the same settings as MATLAB defaults
        self.device = serial.Serial(
            port=self.serialPort,       # Specify the port name
            baudrate=self.baudRate,    # Set the baud rate
            bytesize=serial.EIGHTBITS,  # Data bits (8)
            parity=serial.PARITY_NONE,  # Parity (none)
            stopbits=serial.STOPBITS_ONE,  # Stop bits (1)
            timeout=10,         # Timeout for read operations (10 seconds)
            xonxoff=False,      # Disable software flow control
            rtscts=False,       # Disable hardware (RTS/CTS) flow control
            dsrdtr=False        # Disable hardware (DSR/DTR) flow control
        )
        return

    def setMapping(self):
        self.channelMapping = [
                            (5, 19),
                            (0,0),
                            (0,0),
                            (0,0),
                            (0,0),
                            (0,0),
                            (0,0),
                            (0,0),
                            (0,0),
                            (0,0),
                            (0,0),
                            (0,0),
                            (0,0),
                            (0,0),
                            (0,0),
                            (0,0),                               
                               ]

    def sendToHardware(self):
        #Do all of the normal updating of the super class
        super().sendToHardware()

        #All of the modal stuff & Flat is handled by superclass

        #Loop through zonal shape
        for i, val in enumerate(self.currentShape):
            #Extract the corresponding DAC mapping
            pmod, chan = self.channelMapping[i]
            #Set the value by communicating with FPGA
            self.setSingleDAC(pmod, chan, val)

        return

    def setSingleDAC(self, pmod, chan, volt):
        # pmod = str(pmod)
        # chan = str(chan)
        # volt = str(volt)

        pmod = int(pmod)
        chan = int(chan)
        volt = float(volt)

        data = struct.pack('iif', pmod, chan, volt)
        self.device.write(data)

        # intstructions = [b'e', b'e', pmod.encode(), chan.encode(), volt.encode(), b'\r']
        # for instruct in intstructions:
        #     self.device.write(instruct)  # Send 'e' character
        #     time.sleep(self.communicationPause)

        return

    def __del__(self):
        super().__del__()
        #Code to disconnect from serial port
        self.device.close()
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
    wfc = SUPERPAOWER(conf=confWFC)
    wfc.start()

    l = Listener(wfc, port = int(args.port))
    while l.running:
        l.listen()
        time.sleep(1e-3)