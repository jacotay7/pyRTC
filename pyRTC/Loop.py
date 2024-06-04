"""
Loop Superclass
"""

import os 
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ['NUMBA_NUM_THREADS'] = '1'

from pyRTC.Pipeline import *
from pyRTC.utils import *
from pyRTC.pyRTCComponent import *
import threading
import argparse

import numpy as np
import matplotlib.pyplot as plt
import time
from numba import jit
from sys import platform

@jit(nopython=True)
def compCorrection(CM=np.array([[]], dtype=np.float32),  
                    slopes=np.array([], dtype=np.float32)):
    return np.dot(CM,slopes)

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

class Loop(pyRTCComponent):

    def __init__(self, conf) -> None:

        self.confWFS = conf["wfs"]
        self.confWFC = conf["wfc"]
        self.confLoop = conf["loop"]
        self.name = "Loop"
        
        #Read wfs signal's metadata and open a stream to the shared memory
        self.signalMeta = ImageSHM("signal_meta", (ImageSHM.METADATA_SIZE,), np.float64).read_noblock_safe()
        self.signalDType = float_to_dtype(self.signalMeta[3])
        self.signalSize = int(self.signalMeta[2]//self.signalDType.itemsize)
        self.signalShm = ImageSHM("signal", (self.signalSize,), self.signalDType)
        self.nullSignal = np.zeros(self.signalSize, dtype=self.signalDType)

        #Read wfs SLOPES metadata and open a stream to the shared memory
        self.signal2DMeta = ImageSHM("signal2D_meta", (ImageSHM.METADATA_SIZE,), np.float64).read_noblock_safe()
        self.signal2DDType = float_to_dtype(self.signal2DMeta[3])
        self.signal2DSize = int(self.signal2DMeta[2]//self.signal2DDType.itemsize)
        self.signal2D_width, self.signal2D_height = int(self.signal2DMeta[4]),  int(self.signal2DMeta[5])
        print(self.signal2DMeta[3], (self.signal2D_width, self.signal2D_height), self.signal2DDType)
        self.signal2DShm = ImageSHM("signal2D", (self.signal2D_width, self.signal2D_height), self.signal2DDType)

        #Read wfc metadata and open a stream to the shared memory
        self.wfcMeta = ImageSHM("wfc_meta", (ImageSHM.METADATA_SIZE,), np.float64).read_noblock_safe()
        self.wfcDType = float_to_dtype(self.wfcMeta[3])
        self.numModes = int(self.wfcMeta[2]//self.wfcDType.itemsize)
        self.wfcShm = ImageSHM("wfc", (self.numModes,), self.wfcDType)

        #Read the wfc2D metadata and open a stream to the shared memory
        self.wfc2DMeta = ImageSHM("wfc2D_meta", (ImageSHM.METADATA_SIZE,), np.float64).read_noblock_safe()
        self.wfc2DDType = float_to_dtype(self.wfc2DMeta[3])
        self.wfc2DSize = int(self.wfc2DMeta[2]//self.wfc2DDType.itemsize)
        self.wfc2D_width, self.wfc2D_height = int(self.wfc2DMeta[4]),  int(self.wfc2DMeta[5])
        self.wfc2DShm = ImageSHM("wfc2D", (self.wfc2D_width, self.wfc2D_height), self.wfc2DDType)

        self.opticalGainShm = ImageSHM("og", (self.numModes,), self.wfcDType)
        self.ogPowerLawCoef = setFromConfig(self.confLoop, "ogPowerLawCoef", 0.1)
        self.opticalGainShm.write(powerLawOG(self.numModes, self.ogPowerLawCoef))
        self.stabilityLimit = setFromConfig(self.confLoop, "stabilityLimit", 0.01)


        self.numDroppedModes = setFromConfig(self.confLoop, "numDroppedModes", 0)
        self.numActiveModes = self.numModes - self.numDroppedModes
        self.flat = np.zeros(self.numModes, dtype=self.wfcDType)

        self.IM = np.zeros((self.signalSize, self.numModes),dtype=self.signalDType)
        self.CM = np.zeros((self.numModes, self.signalSize),dtype=self.signalDType)
        self.gain = setFromConfig(self.confLoop, "gain", 0.1)
        self.leakyGain = setFromConfig(self.confLoop, "leakyGain", 0)
        self.perturbAmp = 0
        self.hardwareDelay = setFromConfig(self.confWFC, "hardwareDelay", 0)
        self.pokeAmp = setFromConfig(self.confLoop, "pokeAmp", 1e-2)
        self.numItersIM = setFromConfig(self.confLoop, "numItersIM", 100) 
        self.delay = setFromConfig(self.confLoop, "delay", 0)
        self.IMMethod = setFromConfig(self.confLoop, "IMMethod", "push-pull") 
        self.IMFile = setFromConfig(self.confLoop, "IMFile", "")
        self.clDocrime = False     
        self.numItersDC = 0   
        #Have a history of corrections
        
        tmp2 = self.flat.copy()
        tmp2 = tmp2.reshape(tmp2.size,1) 
        tmp = self.nullSignal.copy()
        tmp = tmp.reshape(tmp.size,1)
        self.docrimeCross = np.zeros_like(tmp@tmp2.T)
        self.docrimeAuto = np.zeros_like(tmp2@tmp2.T)
        self.docrimeBuffer = np.zeros((1+self.delay, *tmp2.shape), 
                                dtype=self.wfcDType)

        """
        Terms for PID integrator
        """
        self.pGain = setFromConfig(self.confLoop, "pGain", 0.1)
        self.iGain = setFromConfig(self.confLoop, "iGain", 0)
        self.dGain = setFromConfig(self.confLoop, "dGain", 0)
        self.controlLimits = setFromConfig(self.confLoop, "controlLimits", [-np.inf, np.inf])
        self.integralLimits = setFromConfig(self.confLoop, "integralLimits", [-np.inf, np.inf])
        self.derivativeFilter = setFromConfig(self.confLoop, "derivativeFilter", 0.1)
        self.controlClipModeStart = setFromConfig(self.confLoop, "controlClipModeStart", 0)
        self.integral = 0

        self.previousWfError = np.zeros_like(self.wfcShm.read_noblock())
        self.previousDerivative = np.zeros_like(self.previousWfError)
        self.controlOutput = np.zeros_like(self.previousWfError)

        self.loadIM()

        super().__init__(self.confLoop)        
        return

    def setGain(self, gain):
        self.gain = gain
        self.gCM = self.gain*self.CM
        return

    def setPeturbAmp(self, amp):
        self.perturbAmp = amp
        return

    def pushPullIM(self):
        
        #For each mode
        for i in range(self.numModes):
            #Reset the correction
            correction = self.flat.copy()
            #Plus amplitude
            correction[i] = self.pokeAmp
            #Post a new shape to be made
            self.sendToWfc(correction)
            #Add some delay to ensure one-to-one
            time.sleep(self.hardwareDelay)
            #Burn the first new image since we were moving the DM during the exposure
            self.signalShm.read()
            #Average out N new WFS frames
            tmp_plus = np.zeros_like(self.IM[:,i])
            for n in range(self.numItersIM):
                tmp_plus += self.signalShm.read()
            tmp_plus /= self.numItersIM

            #Minus amplitude
            correction[i] = -self.pokeAmp
            #Post a new shape to be made
            self.sendToWfc(correction)
            #Add some delay to ensure one-to-one
            time.sleep(self.hardwareDelay)
            #Burn the first new image since we were moving the DM during the exposure
            self.signalShm.read()
            #Average out N new WFS frames
            tmp_minus = np.zeros_like(self.IM[:,i])
            for n in range(self.numItersIM):
                tmp_minus += self.signalShm.read()
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
        # corrections = np.zeros((1+self.delay, *correction.shape), dtype=correction.dtype)

        #Get an initial slope reading to set shapes
        slopes = self.nullSignal.copy()
        slopes = slopes.reshape(slopes.size,1)
        self.docrimeCross = np.zeros_like(self.docrimeCross)
        self.docrimeAuto = np.zeros_like(self.docrimeAuto)

        for i in range(self.numItersIM):
            #Compute new random shape
            correction = np.random.uniform(-self.pokeAmp,self.pokeAmp,correction.size).astype(correction.dtype).reshape(correction.shape)

            #Send our new pertubation to the WFC
            self.sendToWfc(correction.reshape(corrShapeWFC))

            add_to_buffer(self.docrimeBuffer, correction)

            #Get current WFS response
            slopes = self.signalShm.read().reshape(slopes.shape)
        
            #Correlate Current response with old correction by delay time
            self.docrimeCross += slopes@self.docrimeBuffer[0].T
            self.docrimeAuto += self.docrimeBuffer[0]@self.docrimeBuffer[0].T

        self.docrimeCross /= self.numItersIM 
        self.docrimeAuto /= self.numItersIM
        self.IM = self.docrimeCross @np.linalg.inv(self.docrimeAuto) 

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
        self.sendToWfc(self.flat)
        return
    
    def computeCM(self):
        self.numActiveModes = self.numModes-self.numDroppedModes
        if self.numActiveModes < 0:
            print("Invalid Number of Modes used in CM. Check numDroppedModes")
            return
        self.CM[:self.numActiveModes,:] = np.linalg.pinv(self.IM[:,:self.numActiveModes], rcond=0)
        self.CM[self.numActiveModes:,:] = 0
        self.gCM = self.gain*self.CM
        self.fIM = np.copy(self.IM)
        self.fIM[:,self.numActiveModes:] = 0
        return 

    def updateCorrectionPOL(self, correction=np.array([], dtype=np.float32), slopes=np.array([], dtype=np.float32)):
            
        # Compute POL Slopes s_{POL} = s_{RES} + IM*c_{n-1}
        # print(f'slopes: {slopes.shape}, IM: {self.IM.shape}, corr: {correction.shape}')
        s_pol = slopes - self.fIM@correction

        # Update Command Vector c_n = g*CM*s_{POL} + (1 − g) c_{n-1}  https://arxiv.org/pdf/1903.12124.pdf Eq 3
        return (1-self.gain)*correction - np.dot(self.gCM,s_pol)

    def standardIntegratorPOL(self):

        residual_slopes = self.signalShm.read()
        currentCorrection = self.wfcShm.read()
        # print(f'slopes: {residual_slopes.shape}, IM: {self.IM.shape}, corr: {currentCorrection.shape}')

        newCorrection = self.updateCorrectionPOL(correction=currentCorrection, 
                                                 slopes=residual_slopes)
        newCorrection[self.numActiveModes:] = 0
        self.sendToWfc(newCorrection)

        return
    
    def standardIntegrator(self):

        slopes = self.signalShm.read()
        currentCorrection = self.wfcShm.read()
        newCorrection = updateCorrection(correction=currentCorrection, 
                                        gCM=self.gCM, 
                                        slopes=slopes)
        newCorrection[self.numActiveModes:] = 0
        self.sendToWfc(newCorrection)
        return  
    
    def leakyIntegrator(self):

        #Get WFS response
        slopes = self.signalShm.read()
    
        #Compute WFC adjustment (HAS GAIN)
        adjust = self.gain*compCorrection(CM=self.CM, 
                                        slopes=slopes)
        
        #Add optical gains
        ogVec = self.opticalGainShm.read_noblock()
        adjust *= ogVec
        adjust = np.clip(adjust, 
                         -self.stabilityLimit*ogVec, 
                         self.stabilityLimit*ogVec)

        #Get current command
        currentCorrection = self.wfcShm.read_noblock()
        #Leak the current command
        currentCorrection *= (1-self.leakyGain)
        #Adjust the correction
        currentCorrection[:self.numActiveModes] -= adjust[:self.numActiveModes]

        #Send to WFC
        self.sendToWfc(currentCorrection, slopes = slopes)

        return

    def mapsIntegrator(self):

        #Get WFS response
        slopes = self.signalShm.read()
    
        #Compute WFC adjustment
        adjust = self.gain*compCorrection(CM=self.CM, 
                                        slopes=slopes)
        
        #Add optical gains
        ogVec = self.opticalGainShm.read_noblock()
        adjust *= ogVec
        adjust = np.clip(adjust, 
                         -self.stabilityLimit*ogVec, 
                         self.stabilityLimit*ogVec)

        #Get current command
        currentCorrection = self.wfcShm.read_noblock()
        #Leak the current command
        currentCorrection *= (1-self.leakyGain)
        #Adjust the correction
        currentCorrection[:self.numActiveModes] -= adjust[:self.numActiveModes]

        self.currentCorrection = np.clip(currentCorrection, *self.controlLimits)

        #Send to WFC
        self.sendToWfc(currentCorrection, slopes = slopes)

        return



    def pidIntegrator(self):

        slopes = self.signalShm.read()
        #Compute raw error term (numba accelerated)
        wfError = compCorrection(CM=self.CM, 
                                    slopes=slopes)
        
        #Apply Optical Gains
        wfError *= self.opticalGainShm.read_noblock()

        derivative = (wfError - self.previousWfError) 
        
        # Apply low-pass filter to the derivative to reduce noise
        derivative = self.derivativeFilter * derivative + (1 - self.derivativeFilter) * self.previousDerivative
        
        # Update integral (anti-windup: conditional integration)
        # notOutputLimiting = self.controlLimits[0] is None or self.controlLimits[1] is None
        isClipped = np.any(self.controlOutput == self.controlLimits[0]) or np.any(self.controlOutput == self.controlLimits[1])
        #Check to make sure we aren't actively clipping the correction
        if not isClipped:
            #Add to integral
            self.integral += wfError 
            #Clip integral term
            self.integral = np.clip(self.integral, *self.integralLimits)

        # Calculate PID output
        controlOutput = self.pGain * wfError + self.iGain * self.integral + self.dGain * derivative

        # Clip correction (force the loop to not over correct a mode)
        controlOutput[self.controlClipModeStart:] = np.clip(controlOutput[self.controlClipModeStart:], *self.controlLimits)

        #Get new correction vector from the control output
        newCorrection = self.wfcShm.read()*(1-self.leakyGain) - controlOutput #Negative control direction is convention for pyRTC

        #Remove anything in non-corrected modes (might be redundant)
        newCorrection[self.numActiveModes:] = 0
        
        #Apply new correction to mirror
        self.sendToWfc(newCorrection, slopes=slopes)

        # Save state for next iteration
        self.previousWfError = wfError
        self.previousDerivative = derivative
        self.controlOutput = controlOutput
        
        return

    def sendToWfc(self, correction, slopes=None):
        #Get an initial slope reading to set shapes

        if self.clDocrime and slopes is not None:

            slopes = slopes.reshape(slopes.size, 1)
            #Compute new random shape
            randShape = np.random.uniform(-self.pokeAmp,
                                          self.pokeAmp,
                                          correction.size).astype(self.docrimeBuffer[0].dtype).reshape(self.docrimeBuffer[0].shape)

            #Adds to end of buffer (i.e. pos -1)
            add_to_buffer(self.docrimeBuffer,randShape)

            randShape = randShape.astype(correction.dtype).reshape(correction.shape)

            #Only add randomness to active modes, otherwise it will build up
            if self.numActiveModes > 0:
                correction[:self.numActiveModes] += randShape[:self.numActiveModes]
                correction[self.numActiveModes:] = randShape[self.numActiveModes:]
            else:
                correction = randShape

            #Send our new pertubation to the WFC
            self.wfcShm.write(correction)

            #Correlate Current response with old correction by delay time
            self.docrimeCross += slopes@self.docrimeBuffer[0].T
            self.docrimeAuto += self.docrimeBuffer[0]@self.docrimeBuffer[0].T

            self.numItersDC += 1

        else:
            self.wfcShm.write(correction)
        return

    def solveDocrime(self):

        self.docrimeCross /= self.numItersDC 
        self.docrimeAuto /= self.numItersDC
        self.clDCIM = self.docrimeCross@np.linalg.inv(self.docrimeAuto)
        np.save("/home/jtaylor/pyRTC/MAPS/calib/cl_dc_tmp_IM.npy", self.clDCIM)

        return

    def plotIM(self, row=None):
        # if not (row is None):
        #     row2D = signal2D(self.IM[:,row], )
        #     plt.imshow(row2D, cmap = 'inferno')
        #     plt.colorbar()
        #     plt.show()
        # else:
        plt.imshow(self.IM, cmap = 'inferno', aspect='self.docrimeAuto')
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
    set_affinity((conf["loop"]["affinity"])%os.cpu_count()) 
    decrease_nice(pid)

    loop = Loop(conf=conf)
    
    l = Listener(loop, port= int(args.port))
    while l.running:
        l.listen()
        time.sleep(1e-3)