"""
Loop Superclass
"""
from pyRTC.Pipeline import ImageSHM, work
import threading
import os
import numpy as np
import matplotlib.pyplot as plt
import time
from numba import jit


@jit(nopython=True)
def updateCorrection(correction=np.array([], dtype=np.float32), 
                     gCM=np.array([[]], dtype=np.float32),  
                     slopes=np.array([], dtype=np.float32)):
    return correction - np.dot(gCM,slopes)

class Loop:

    def __init__(self, wfs, wfc) -> None:
        self.wfs = wfs
        self.wfc = wfc
        self.signalSize = self.wfs.signal.read_noblock_safe().size
        self.dtype = self.wfs.signal.read_noblock_safe().dtype
        self.numModes = self.wfc.M2C.shape[1]
        self.IM = np.zeros((self.signalSize, self.numModes),dtype=self.dtype)
        self.CM = np.zeros((self.numModes, self.signalSize),dtype=self.dtype)
        self.gain = 0.5

        self.wfs.start()
        self.wfc.start()

        self.alive = True
        self.running = False
        self.affinity = 11

        functionsToRun = ["standardIntegrator"]
        self.workThreads = []
        for i, functionName in enumerate(functionsToRun):
            # Launch a separate thread
            workThread = threading.Thread(target=work, args = (self,functionName), daemon=True)
            # Start the thread
            workThread.start()
            # Set CPU affinity for the thread
            # print(workThread.native_id, {self.affinity+i,})
            os.sched_setaffinity(workThread.native_id, {(self.affinity+i)%os.cpu_count(),})  
            self.workThreads.append(workThread)

        return
    
    def __del__(self):
        print("Deleeting Loop Object")
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
        self.gCM = self.gain*self.gCM
        return

    def computeIM(self, pokeAmp, N = 100, flagInd=0, hardwareDelay=1e-3):

        # Launch a separate thread
        # self.go = False
        # IMThread = threading.Thread(target=doIM, args = (self,self.wfs, self.wfc, pokeAmp), daemon=True)
        # # Start the thread
        # IMThread.start()
        # # Set CPU affinity for the thread
        # # print(workThread.native_id, {self.affinity+i,})
        # os.sched_setaffinity(IMThread.native_id, {(self.affinity)%os.cpu_count(),})  
        # self.go = True
        # IMThread.join()
        # self.go = False
        # self.wfs.read()
        # self.wfc.flatten()

        for i in range(self.IM.shape[1]):
            # print(f"IM -- Pushing Mode {i}")
            correction = np.zeros_like(self.wfc.read())
            #Plus amplitude
            correction[i] = pokeAmp
            tmp_plus = np.zeros_like(self.IM[:,i])
            #Post a new shape to be made
            self.wfc.write(correction)
            #Add some delay to ensure one-to-one
            time.sleep(hardwareDelay)
            #Burn the first new image
            self.wfs.read(flagInd=flagInd)
            for n in range(N):
                tmp_plus += self.wfs.read(flagInd=flagInd)
            tmp_plus /= N

            #Minus amplitude

            # print(f"IM -- Pulling Mode {i}")
            correction[i] = -pokeAmp
            tmp_minus = np.zeros_like(self.IM[:,i])
            
            self.wfc.write(correction)
            #Add some delay to ensure one-to-one
            time.sleep(hardwareDelay)
            #Burn the first new image
            self.wfs.read(flagInd=flagInd)
            for n in range(N):
                tmp_minus += self.wfs.read(flagInd=flagInd)
            tmp_minus /= N

            self.IM[:,i] = (tmp_plus-tmp_minus)/(2*pokeAmp)


        # self.computeCM()

        return
    
    def computeCM(self, numDropped=0):
        self.CM[:self.numModes-numDropped-1,:] = np.linalg.pinv(self.IM[:,:self.numModes-numDropped-1])
        self.CM[self.numModes-numDropped,:] = 0
        self.gCM = self.gain*self.CM
        return 
    

    
    def standardIntegrator(self,flagInd=0):

        slopes = self.wfs.read(flagInd=flagInd)
        self.wfc.write(updateCorrection(correction=self.wfc.currentCorrection, 
                                        gCM=self.gCM, 
                                        slopes=slopes))
        return

    def plotIM(self, row=None):
        if not (row is None):
            row2D = self.wfs.signal2D(self.IM[:,row])
            plt.imshow(row2D, cmap = 'inferno')
            plt.colorbar()
            plt.show()
        else:
            plt.imshow(self.IM, cmap = 'inferno', aspect='auto')
            plt.show()

