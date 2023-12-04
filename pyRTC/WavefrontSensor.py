"""
Wavefront Sensor Superclass
"""
from pyRTC.Pipeline import ImageSHM, work
import numpy as np
import matplotlib.pyplot as plt
import threading
import os
import time
from numba import jit

@jit(nopython=True)
def computeSlopesPYWFS(p1=np.array([],dtype=np.float32), 
                       p2=np.array([],dtype=np.float32),
                       p3=np.array([],dtype=np.float32), 
                       p4=np.array([],dtype=np.float32), 
                       flatNorm=True):
    # p1 = image[p1mask]
    # p2 = image[p2mask]
    # p3 = image[p3mask]
    # p4 = image[p4mask]

    x_slopes = (p1 + p2) - (p3 + p4)
    y_slopes = (p1 + p3) - (p2 + p4)

    # if flatNorm:
    norm = np.ones(x_slopes.size)*np.mean(p1+p2+p3+p4)
    # else:
        # norm = p1+p2+p3+p4
    x_slopes = np.divide(x_slopes,norm)
    y_slopes = np.divide(y_slopes,norm)

    return np.concatenate((x_slopes,
                           y_slopes))

class WavefrontSensor:

    def __init__(self, imageShape, wfsType = "PYWFS", signalType = 'slopes', flatNorm=True, layout=None) -> None:

        self.imageShape = imageShape
        self.imagedType = np.uint16
        self.signaldType = np.float32
        self.image = ImageSHM("wfs", imageShape, self.imagedType)
        self.signal = ImageSHM("signal", imageShape, self.signaldType)
        self.data = np.zeros(imageShape, dtype=self.imagedType)
        self.wfsType = wfsType
        self.signalType = signalType
        self.layout = layout
        self.affinity = 12

        if wfsType == "PYWFS":
            a, b = int(0.25*imageShape[0]), int(0.75*imageShape[0])
            c, d = int(0.25*imageShape[1]), int(0.75*imageShape[1])
            r = min(imageShape[0]-b,imageShape[1]-d)
            self.setPupils([(a,c), (a,d), (b,c), (b,d)], r)
            self.flatNorm = flatNorm

        self.alive = True
        self.running = False

        functionsToRun = ["expose","computeSignal"]
        self.workThreads = []
        for i, functionName in enumerate(functionsToRun):
            # Launch a separate thread
            workThread = threading.Thread(target=work, args = (self,functionName), daemon=True)
            # Start the thread
            workThread.start()
            # Set CPU affinity for the thread
            # print(workThread.native_id, {self.affinity+i,})
            os.sched_setaffinity(workThread.native_id, {self.affinity+i,})  
            self.workThreads.append(workThread)

        return
    
    def __del__(self):
        print("Deleeting WFS Object")
        self.alive = False
        # for workThread in self.workThreads:

        #     workThread.join()
        return

    def start(self):

        self.running = True
        return
    
    def stop(self):

        self.running = False
        return
    
    def setRoi(self, roi):

        self.roiWidth = roi[0]
        self.roiHeight = roi[1]
        self.roiLeft = roi[2]
        self.roiTop = roi[3]
        return

    def setExposure(self, exposure):
        self.exposure = exposure
        return
    
    def setBinning(self, binning):
        self.binning = binning
        return
    
    def setGain(self, gain):
        self.gain = gain
        return
    
    def setBitDepth(self, bitDepth):
        self.bitDepth = bitDepth
        return
    
    def expose(self):
        self.image.write(self.data)
        return

    def readImage(self,flagInd=0):
        return self.image.read(flagInd=flagInd)

    def read(self,flagInd=0):
        return self.signal.read(flagInd=flagInd)


   
    def computeSignal(self):
        image = self.readImage().astype(self.signaldType)

        if self.wfsType == "PYWFS":
            if self.signalType == "slopes":
                p1,p2,p3,p4 = image[self.p1mask], image[self.p2mask], image[self.p3mask], image[self.p4mask]
                self.signal.write(computeSlopesPYWFS(p1=p1,
                                                     p2=p2,
                                                     p3=p3,
                                                     p4=p4,
                                                          flatNorm=self.flatNorm))
        return

    def setPupils(self, pupilLocs, pupilRadius):
        self.pupilLocs = pupilLocs
        self.pupilRadius = pupilRadius
        self.computePupilsMask()
        if self.signalType == "slopes":
            del self.signal
            self.signal = ImageSHM("signal", (np.count_nonzero(self.pupilMask)//2,), self.signaldType)
            slopemask =  self.pupilMask[self.pupilLocs[0][1]-self.pupilRadius+1:self.pupilLocs[0][1]+self.pupilRadius, 
                                        self.pupilLocs[0][0]-self.pupilRadius+1:self.pupilLocs[0][0]+self.pupilRadius] > 0
            self.layout = np.concatenate([slopemask, slopemask], axis=1)
        return

    def computePupilsMask(self):
        pupils = []
        self.pupilMask = np.zeros(self.imageShape)
        xx,yy = np.meshgrid(np.arange(self.pupilMask.shape[0]),np.arange(self.pupilMask.shape[1]))
        for i, pupil_loc in enumerate(self.pupilLocs):
            px, py = pupil_loc
            zz = np.sqrt((xx-px)**2 + (yy-py)**2)
            pupils.append(zz < self.pupilRadius)
            self.pupilMask += pupils[-1]*(i+1)
        self.p1mask = self.pupilMask == 1
        self.p2mask = self.pupilMask == 2
        self.p3mask = self.pupilMask == 3
        self.p4mask = self.pupilMask == 4
        return

    def plotPupils(self):
        # plt.figure(figsize=(10,8))
        plt.imshow(self.pupilMask, cmap = 'inferno',origin='lower',aspect ='auto')
        plt.colorbar()
        plt.title("Pupil Mask (Value is Pupil Number)")
        plt.show()

        plt.imshow(self.pupilMask*self.readImage(), cmap = 'inferno',origin='lower',aspect ='auto')
        colors = ['g','b','orange', 'r']
        for i in range(len(self.pupilLocs)):
            px, py = self.pupilLocs[i]
            plt.axvline(x = px, color = colors[i], alpha = 0.6)
            plt.axhline(y = py, color = colors[i], alpha = 0.6)
        plt.colorbar()
        plt.title("Pupil Mask * Image ")
        plt.show()
        return

    def signal2D(self, signal, layout=None):
        if layout is None and isinstance(self.layout, np.ndarray):
            layout = self.layout
        else:
            return -1
        curSignal2D = np.zeros(layout.shape)
        slopemask = layout[:,:layout.shape[1]//2]
        curSignal2D[:,:layout.shape[1]//2][slopemask] = signal[:signal.size//2]
        curSignal2D[:,layout.shape[1]//2:][slopemask] = signal[signal.size//2:]
        return curSignal2D
    
    def plot(self):
        arr = self.readImage(flagInd=1)
        plt.imshow(arr, cmap = 'inferno', origin='lower')
        plt.colorbar()
        plt.show()

        curSignal = self.read(flagInd=1)
        if not (self.layout is None):
            curSignalPlot = self.signal2D(curSignal)
        else:
            curSignalPlot = curSignal
            
        if len(curSignalPlot.shape) == 1:
            # plt.figure(figsize=(12,5))
            plt.plot(curSignalPlot)
            plt.show()
        elif len(curSignalPlot.shape) == 2:
            # plt.figure(figsize=(10,8))
            plt.imshow(curSignalPlot, cmap = "inferno", aspect='auto', origin='lower')
            plt.colorbar()
            plt.show()
        return