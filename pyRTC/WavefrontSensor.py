"""
Wavefront Sensor Superclass
"""
from pyRTC.Pipeline import ImageSHM
import numpy as np
import matplotlib.pyplot as plt

class WavefrontSensor:

    def __init__(self, imageShape, wfsType = "PYWFS", signalType = 'slopes', flatNorm=True, layout=None) -> None:

        self.imageShape = imageShape
        self.image = ImageSHM("wfs", imageShape, np.float64)
        self.signal = ImageSHM("signal", imageShape, np.float64)
        self.data = np.zeros(imageShape)
        self.wfsType = wfsType
        self.signalType = signalType
        self.layout = layout


        if wfsType == "PYWFS":
            a, b = int(0.25*imageShape[0]), int(0.75*imageShape[0])
            c, d = int(0.25*imageShape[1]), int(0.75*imageShape[1])
            r = min(imageShape[0]-b,imageShape[1]-d)
            self.setPupils([(a,c), (a,d), (b,c), (b,d)], r)
            self.flatNorm = flatNorm

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

    def readImage(self):
        return self.image.read()

    def read(self):
        return self.signal.read()

    def computeSignal(self):
        image = self.readImage()

        if self.wfsType == "PYWFS":
            if self.signalType == "slopes":
                p1 = image[np.where(self.pupilMask == 1)]
                p2 = image[np.where(self.pupilMask == 2)]
                p3 = image[np.where(self.pupilMask == 3)]
                p4 = image[np.where(self.pupilMask == 4)]

                x_slopes = (p1 + p2) - (p3 + p4)
                y_slopes = (p1 + p3) - (p2 + p4)

                if self.flatNorm:
                    norm = np.mean(p1+p2+p3+p4)
                else:
                    norm = p1+p2+p3+p4
                
                x_slopes /= norm
                y_slopes /= norm
                signal = np.concatenate([x_slopes, y_slopes])
                self.signal.write(signal)
        return

    def setPupils(self, pupilLocs, pupilRadius):
        self.pupilLocs = pupilLocs
        self.pupilRadius = pupilRadius
        self.computePupilsMask()
        if self.signalType == "slopes":
            del self.signal
            self.signal = ImageSHM("signal", (np.count_nonzero(self.pupilMask)//2,), np.float64)
            slopemask =  self.pupilMask[self.pupilLocs[0][1]-self.pupilRadius+1:self.pupilLocs[0][1]+self.pupilRadius, 
                                        self.pupilLocs[0][0]-self.pupilRadius+1:self.pupilLocs[0][0]+self.pupilRadius] > 0
            self.layout = np.concatenate([slopemask, slopemask], axis=1)
        return

    def computePupilsMask(self):
        pupils = []
        self.pupilMask = np.zeros_like(self.readImage())
        xx,yy = np.meshgrid(np.arange(self.pupilMask.shape[0]),np.arange(self.pupilMask.shape[1]))
        for i, pupil_loc in enumerate(self.pupilLocs):
            px, py = pupil_loc
            zz = np.sqrt((xx-px)**2 + (yy-py)**2)
            pupils.append(zz < self.pupilRadius)
            self.pupilMask += pupils[-1]*(i+1)
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
        if layout is None and not self.layout is None:
            layout = self.layout
        else:
            return -1
        curSignal2D = np.zeros(layout.shape)
        slopemask = layout[:,:layout.shape[1]//2]
        curSignal2D[:,:layout.shape[1]//2][slopemask] = signal[:signal.size//2]
        curSignal2D[:,layout.shape[1]//2:][slopemask] = signal[signal.size//2:]
        return curSignal2D
    
    def plot(self):
        arr = self.readImage()
        plt.imshow(arr, cmap = 'inferno', origin='lower')
        plt.colorbar()
        plt.show()

        curSignal = self.read()
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