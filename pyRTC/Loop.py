"""
Loop Superclass
"""
from pyRTC.Pipeline import *
from pyRTC.utils import *
import threading
import argparse
import os 
import numpy as np
import matplotlib.pyplot as plt
import time
from numba import jit
from sys import platform

@jit(nopython=True)
def updateCorrection(correction=np.array([], dtype=np.float32), 
                     gCM=np.array([[]], dtype=np.float32),  
                     slopes=np.array([], dtype=np.float32)):
    return correction - np.dot(gCM,slopes)

# @jit(nopython=True)
# def updateCorrectionPerturb(correction=np.array([], dtype=np.float32),
#                             pertub=np.array([], dtype=np.float32),  
#                      gCM=np.array([[]], dtype=np.float32),  
#                      slopes=np.array([], dtype=np.float32)):
#     return correction - np.dot(gCM,slopes) + pertub

class Loop:

    def __init__(self, conf) -> None:

        self.confWFS = conf["wfs"]
        self.confWFC = conf["wfc"]
        self.confLoop = conf["loop"]
        self.name = "Loop"
        
        #Read wfs signal's metadata and open a stream to the shared memory
        self.wfsMeta = ImageSHM("signal_meta", (4,), np.float64).read_noblock_safe()
        self.signalDType = float_to_dtype(self.wfsMeta[3])
        self.signalSize = int(self.wfsMeta[2]//self.signalDType.itemsize)
        self.wfsShm = ImageSHM("signal", (self.signalSize,), self.signalDType)
        self.nullSignal = np.zeros(self.signalSize, dtype=self.signalDType)

        #Read wfc metadata and open a stream to the shared memory
        self.wfcMeta = ImageSHM("wfc_meta", (4,), np.float64).read_noblock_safe()
        self.wfcDType = float_to_dtype(self.wfcMeta[3])
        self.numModes = int(self.wfcMeta[2]//self.wfcDType.itemsize)
        self.wfcShm = ImageSHM("wfc", (self.numModes,), self.wfcDType)

        self.numDroppedModes = self.confLoop["numDroppedModes"]
        self.numActiveModes = self.numModes - self.numDroppedModes
        self.flat = np.zeros(self.numModes, dtype=self.wfcDType)

        self.IM = np.zeros((self.signalSize, self.numModes),dtype=self.signalDType)
        self.CM = np.zeros((self.numModes, self.signalSize),dtype=self.signalDType)
        self.gain = self.confLoop["gain"]
        self.perturbAmp = 0
        self.hardwareDelay = self.confWFC["hardwareDelay"]
        self.pokeAmp = self.confLoop["pokeAmp"] 
        self.numItersIM = self.confLoop["numItersIM"]
        self.delay = self.confLoop["delay"]
        self.IMMethod = self.confLoop["method"]
        self.IMFile = setFromConfig(self.confLoop, "IMFile", "")
        
        self.loadIM()

        self.alive = True
        self.running = False
        self.affinity = self.confLoop["affinity"]

        functionsToRun = self.confLoop["functions"]
        self.workThreads = []
        for i, functionName in enumerate(functionsToRun):
            # Launch a separate thread
            workThread = threading.Thread(target=work, args = (self,functionName), daemon=True)
            # Start the thread
            workThread.start()
            # Set CPU affinity for the thread
            # print(workThread.native_id, {self.affinity+i,})
            if platform != 'darwin':
                os.sched_setaffinity(workThread.native_id, {(self.affinity+i)%os.cpu_count(),})  
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

    def setGain(self, gain):
        self.gain = gain
        self.gCM = self.gain*self.CM
        return

    def setPeturbAmp(self, amp):
        self.perturbAmp = amp
        return

    def pushPullIM(self, flagInd=0):
         
        #For each mode
        for i in range(self.numModes):
            #Reset the correction
            correction = self.flat.copy()
            #Plus amplitude
            correction[i] = self.pokeAmp
            #Post a new shape to be made
            self.wfcShm.write(correction)
            #Add some delay to ensure one-to-one
            time.sleep(self.hardwareDelay)
            #Burn the first new image since we were moving the DM during the exposure
            self.wfsShm.read(flagInd=flagInd)
            #Average out N new WFS frames
            tmp_plus = np.zeros_like(self.IM[:,i])
            for n in range(self.numItersIM):
                tmp_plus += self.wfsShm.read(flagInd=flagInd)
            tmp_plus /= self.numItersIM

            #Minus amplitude
            correction[i] = -self.pokeAmp
            #Post a new shape to be made
            self.wfcShm.write(correction)
            #Add some delay to ensure one-to-one
            time.sleep(self.hardwareDelay)
            #Burn the first new image since we were moving the DM during the exposure
            self.wfsShm.read(flagInd=flagInd)
            #Average out N new WFS frames
            tmp_minus = np.zeros_like(self.IM[:,i])
            for n in range(self.numItersIM):
                tmp_minus += self.wfsShm.read(flagInd=flagInd)
            tmp_minus /= self.numItersIM

            #Compute the normalized difference
            self.IM[:,i] = (tmp_plus-tmp_minus)/(2*self.pokeAmp)

        return
    
    def docrimeIM(self, flagInd=1):
        
        #Send the flat command to the WFC
        self.flatten()

        #Get a correction to set the shape
        correction = self.flat.copy()
        corrShapeWFC = correction.shape
        correction = correction.reshape(correction.size,1)

        #Have a history of corrections
        corrections = np.zeros((1+self.delay, *correction.shape), dtype=correction.dtype)

        #Get an initial slope reading to set shapes
        slopes = self.nullSignal.copy()
        slopes = slopes.reshape(slopes.size,1)
        cross = np.zeros_like(slopes@correction.T)
        auto = np.zeros_like(correction@correction.T)

        for i in range(self.numItersIM):
            #Compute new random shape
            correction = np.random.uniform(-self.pokeAmp,self.pokeAmp,correction.size).astype(correction.dtype).reshape(correction.shape)
            #If we are in Closed Loop
            if self.running:
                #Read the current shape of the WFC and add our perturbation ontop
                correction = self.wfcShm.read_noblock() + correction

            #Send our new pertubation to the WFC
            self.wfcShm.write(correction.reshape(corrShapeWFC))
            #Move old shapes back in history
            corrections[:-1] = corrections[1:]
            #Add new correction
            corrections[-1] = correction

            #Get current WFS response
            slopes = self.wfsShm.read(flagInd=flagInd).reshape(slopes.shape)
        
            #Correlate Current response with old correction by delay time
            cross += slopes@corrections[0].T
            auto += corrections[0]@corrections[0].T

        cross /= self.numItersIM 
        auto /= self.numItersIM
        self.IM = cross@np.linalg.inv(auto) 
        return

    def computeIM(self):

        if self.IMMethod == 'docrime':
            self.docrimeIM()
        else:
            self.pushPullIM()

        self.computeCM()
        return
    
    def saveIM(self,filename=''):
        if filename == '':
            filename = self.IMFile
        np.save(filename, self.IM)

    def loadIM(self,filename=''):
        if filename == '':
            filename = self.IMFile
        if filename == '':
            self.IM = np.zeros_like(self.IM)
        else:
            self.IM = np.load(filename)
        self.computeCM()

    def flatten(self):
        self.wfcShm.write(self.flat)
        return
    
    def computeCM(self):
        self.numActiveModes = self.numModes-self.numDroppedModes
        self.CM[:self.numActiveModes,:] = np.linalg.pinv(self.IM[:,:self.numActiveModes], rcond=0)
        self.CM[self.numActiveModes:,:] = 0
        self.gCM = self.gain*self.CM
        self.fIM = np.copy(self.IM)
        self.fIM[:,self.numActiveModes:] = 0
        return 
        
    # @jit(nopython=True)
    def updateCorrectionPOL(self, correction=np.array([], dtype=np.float32), slopes=np.array([], dtype=np.float32)):
            
        # Compute POL Slopes s_{POL} = s_{RES} + IM*c_{n-1}
        # print(f'slopes: {slopes.shape}, IM: {self.IM.shape}, corr: {correction.shape}')
        s_pol = slopes - self.fIM@correction

        # Update Command Vector c_n = g*CM*s_{POL} + (1 − g) c_{n-1}  https://arxiv.org/pdf/1903.12124.pdf Eq 3
        return (1-self.gain)*correction - np.dot(self.gCM,s_pol)

    def standardIntegratorPOL(self,flagInd=0):

        residual_slopes = self.wfsShm.read(flagInd=flagInd)
        currentCorrection = self.wfcShm.read()
        # print(f'slopes: {residual_slopes.shape}, IM: {self.IM.shape}, corr: {currentCorrection.shape}')

        newCorrection = self.updateCorrectionPOL(correction=currentCorrection, 
                                                 slopes=residual_slopes)
        newCorrection[self.numActiveModes:] = 0
        self.wfcShm.write(newCorrection)

        return

    # // Compute POL Slopes s_{POL} = s_{RES} + IM*c_{n-1}  https://arxiv.org/pdf/1903.12124.pdf Eq 1
    # if(modal)
    #   pol_slope_vector = _InteractionMatrix(Eigen::all,Eigen::seqN(0,num_used_modes))*(command_modal(Eigen::seqN(0,num_used_modes)) - dm_flat_modal(Eigen::seqN(0,num_used_modes))) + slope_vector.cast<double>();
    # else
    #   pol_slope_vector = _InteractionMatrix*(command_zonal - dm_flat_zonal) + slope_vector.cast<double>();

    # pol_slopes_wo.send(&pol_slope_vector(0));

    
    def standardIntegrator(self,flagInd=0):

        slopes = self.wfsShm.read(flagInd=flagInd)
        currentCorrection = self.wfcShm.read()
        newCorrection = updateCorrection(correction=currentCorrection, 
                                        gCM=self.gCM, 
                                        slopes=slopes)
        newCorrection[self.numActiveModes:] = 0
        self.wfcShm.write(newCorrection)
        return

    def plotIM(self, row=None):
        # if not (row is None):
        #     row2D = signal2D(self.IM[:,row], )
        #     plt.imshow(row2D, cmap = 'inferno')
        #     plt.colorbar()
        #     plt.show()
        # else:
        plt.imshow(self.IM, cmap = 'inferno', aspect='auto')
        plt.show()

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
    os.sched_setaffinity(pid, {conf["loop"]["affinity"],})
    decrease_nice(pid)

    loop = Loop(conf=conf)
    
    l = Listener(loop, port= int(args.port))
    while l.running:
        l.listen()
        time.sleep(1e-3)