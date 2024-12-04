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

        self.communicationPause = setFromConfig(conf, "communicationPause", 0.05) #seconds

        #Ask for the number of actuators
        self.numActuators =  conf["numActuators"]
        self.maxVoltage = setFromConfig(conf, "maxVoltage", 8.0)
        self.minVoltage =  setFromConfig(conf, "minVoltage", 0.0)
        self.maxCoeff = setFromConfig(conf, "maxCoeff", 1.0)
        self.minCoeff =  setFromConfig(conf, "minCoeff", -1.0)
        self.maxCommunicationAttempts = setFromConfig(conf, "maxCommunicationAttempts", 5)
        self.numDroppedFrames = 0
        self.acknowledgedFrames = 0
        self.ACK = 0x34
        self.NACK = 0x35
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

        self.device = None

        self.connectToChip()

        self.setMapping()

        self.device.write('\r'.encode())

        time.sleep(0.5)
        # Read the response from the device
        response = self.device.read(self.device.in_waiting).decode()

        if not response:
            print("No response received from the device.")
        else:
            print(response)
        # self.device.read(1)

        #flatten the mirror
        time.sleep(1)
        self.flatten()
        self.clockActive = False
        self.clockDelay = 1e-4
        self.BIN = True
        self.LOW = np.zeros_like(self.flat)
        self.HIGH = np.zeros_like(self.flat)

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
            # bytesize=serial.EIGHTBITS,  # Data bits (8)
            # parity=serial.PARITY_NONE,  # Parity (none)
            # stopbits=serial.STOPBITS_ONE,  # Stop bits (1)
            timeout=1e-1,         # Timeout for read operations (10 seconds)
            # xonxoff=False,      # Disable software flow control
            # rtscts=False,       # Disable hardware (RTS/CTS) flow control
            # dsrdtr=False        # Disable hardware (DSR/DTR) flow control
        )
        return

    def setMapping(self):
        # self.channelMapping = [
        #                      (4, 8),
        # ]
        # self.channelMapping = [
        #                     (4, 7),
        #                     (4, 6),
        #                     (4, 5),
        #                     (4, 4),
        #                     (4, 3),
        #                     (4, 2),
        #                     (4, 1),
        #                     (4, 0),
        #                     (4, 8),
        #                     (4, 9),
        #                     (4, 10),
        #                     (4, 11),
        #                     (4, 12),
        #                     (4, 13),
        #                     (4, 14),
        #                     (4, 15),                               
        #                        ]
        self.channelMapping = [(4,i) for i in range(16)]
    def sendToHardware(self):
        #Do all of the normal updating of the super class
        super().sendToHardware()

        #All of the modal stuff & Flat is handled by superclass
        
        self.currentShape = np.clip(self.currentShape,self.minVoltage,self.maxVoltage)
        #current = c2m@(max- flat)

        # self.currentCorrection = self.correctionVector.read_noblock()
        # self.currentCorrection = self.C2M@self.currentShape - self.flatModal
        if self.minCoeff is not None and self.maxCoeff is not None:
            self.currentCorrection = np.clip(self.currentCorrection,
                                            self.minCoeff,
                                            self.maxCoeff
                                            )
        #Copy to shared memory without setting flag
        np.copyto(self.correctionVector.arr, self.currentCorrection)

        pkt = self.correctionToPacket(self.currentShape)
        
        response = b'5'
        self.device.write(pkt)
        # attemptCount = 0
        # self.device.flush()
        # while int.from_bytes(response, "big") != self.ACK and attemptCount < self.maxCommunicationAttempts:
        #     # print(f"Sending packet: {pkt.hex()}")
        #     self.device.write(pkt)
        #     # Read the response from the device
        #     response = self.device.read(1)
        #     attemptCount += 1

        # if int.from_bytes(response, "big") != self.ACK and attemptCount >= self.maxCommunicationAttempts:
        #     self.numDroppedFrames += 1
        # else:
        #     self.acknowledgedFrames += 1
        # return

    def runClock(self):
        if self.clockActive:
            if self.BIN:
                correction = self.HIGH
            else:
                correction = self.LOW
            self.BIN = not self.BIN
            self.correctionVector.write(correction)
            time.sleep(self.clockDelay)
        else:
            time.sleep(1)
        return

    def correctionToPacket(self, correction):

        header = struct.pack('B', 33) #b'\x21'  # Example header
        num_commands = len(correction)
        
        # Start with the header and the number of commands
        packet = header + struct.pack('B', num_commands)
        
        # Append each command in 'iif' format
        for i, val in enumerate(correction):
            pmod, chan = self.channelMapping[i]
            packet += struct.pack('BB', int(pmod), int(chan))
            packet += struct.pack('>H', int(420*float(val)))
        
        # Calculate checksum
        checksum = calculate_checksum(packet)
        
        # Append checksum
        packet += struct.pack('B', checksum)

        return packet
    
    def readM2C(self, filename=''):
        super().readM2C(filename=filename)
        M2C = self.M2C.copy()
        #Normalize each mode to min/max of 1
        for i in range(M2C.shape[1]):
            M2C[:,i] /= np.max(np.abs(M2C[:,i]))
        self.setM2C(M2C)
        return 
    def __del__(self):
        super().__del__()
        #Code to disconnect from serial port
        self.device.close()
        return
    
    def startClock(self, freq, MAG, checkerboard = False):
        self.clockDelay = 1/(2*freq)

        self.BIN = True
        self.LOW = np.zeros_like(self.flat)
        self.HIGH = self.LOW + MAG
        if checkerboard:
            self.HIGH[::2] = self.LOW[::2]

        self.clockActive = True
        return
    
    def stopClock(self):
        self.clockActive = False
        return
if __name__ == "__main__":

    launchComponent(SUPERPAOWER, "wfc", start = True)