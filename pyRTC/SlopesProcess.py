"""
Loop Superclass
"""
from pyRTC.Pipeline import *
from pyRTC.utils import *
import threading
import argparse
import sys
import os 
import numpy as np
import matplotlib.pyplot as plt
import time
from numba import jit
from sys import platform

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

@jit(nopython=True)
def computeSlopesSHWFS(image=np.array([],dtype=np.float32), 
                       unaberratedSlopes=np.array([],dtype=np.float32), 
                       spacing=3.5, 
                       offsetX=0, 
                       offsetY=0):
    #Compute the closest integer size of the sub apertures
    intN = int(np.round(spacing,0))
    #Compute the number of sub apertures
    numRegions = unaberratedSlopes.shape[1]#image.shape[0]//intN
    #Pre compute the array to bias our centroid by
    xvals = np.arange(intN) - intN//2 
    #Initialize our slopes output
    slopes = np.zeros(unaberratedSlopes.shape, dtype=unaberratedSlopes.dtype)
    #For each regions
    for i in range(numRegions):
        for j in range(numRegions):
            #Compute where to start, all this does is account for non-integer
            #spacing between the subapertures
            start_i = int(np.round(spacing*i,0)) + offsetY
            start_j = int(np.round(spacing*j,0)) + offsetX
            #Ignore if we go off the edge of the image
            if start_j+intN < image.shape[1] and start_i+intN < image.shape[0]:
                sub_im = image[start_i:start_i+intN,
                            start_j:start_j+intN]
                norm = np.sum(sub_im)
                #Only compute centroid if we have flux
                if norm > 0: 
                    slopes[i,j] = np.sum(xvals*sub_im)/norm
                    slopes[i+numRegions,j] = np.sum(xvals*sub_im.T)/norm
    return slopes - unaberratedSlopes

class SlopesProcess:

    def __init__(self, conf) -> None:

        self.confWFS = conf["wfs"]
        self.name = "Slopes"
        self.imageShape = (self.confWFS["width"], self.confWFS["height"])

        self.conf = conf["slopes"]

        #Read wfs images's metadata and open a stream to the shared memory
        self.wfsMeta = ImageSHM("wfs_meta", (ImageSHM.METADATA_SIZE,), np.float64).read_noblock_safe()
        self.imageDType = float_to_dtype(self.wfsMeta[3])
        self.wfsShm = ImageSHM("wfs", self.imageShape, self.imageDType)

        self.signalDType = np.float32
        # self.signal = ImageSHM("signal", self.imageShape, self.signalDType)

        self.wfsType = self.conf["type"] 
        self.signalType = self.conf["signalType"] 
        self.layout = None

        if self.wfsType.lower() == "pywfs":
            #Check if we have specified a pupil layout
            # self.signal = ImageSHM("signal", self.imageShape, self.signalDType)

            if "pupils" in self.conf.keys():
                pupilLocs = [(int(x.split(',')[1]), int(x.split(',')[0])) for x in self.conf["pupils"]]
                self.setPupils(pupilLocs, self.conf["pupilsRadius"])
            else: #Default Pupil layout
                a, b = int(0.25*self.imageShape[0]), int(0.75*self.imageShape[0])
                c, d = int(0.25*self.imageShape[1]), int(0.75*self.imageShape[1])
                r = min(self.imageShape[0]-b,self.imageShape[1]-d)
                self.setPupils([(a,c), (a,d), (b,c), (b,d)], r)
            if self.signalType == 'slopes':
                #Set normalization
                self.flatNorm = self.conf["flatNorm"]

        elif self.wfsType.lower() == "shwfs":

            self.shwfsContrast = 4
            self.subApSpacing = self.conf["subApSpacing"]
            self.numRegions = self.imageShape[0]//int(np.round(self.subApSpacing,0))
            self.offsetX = self.conf["subApOffsetX"]
            self.offsetY = self.conf["subApOffsetY"]
            # del self.signal
            self.signalSize = int(2*self.numRegions**2)
            self.signalShape = (2*self.numRegions,self.numRegions)

            print(f'subApSpacing: {self.subApSpacing}')
            print(f'numRegions: {self.numRegions}')
            print(f'offsetX: {self.offsetX}')
            print(f'offsetY: {self.offsetY}')
            print(f'signalSize: {self.signalSize}')
            print(f'signalShape: {self.signalShape}')
            print(f'signalDType: {self.signalDType}')

            self.signal = ImageSHM("signal", self.signalShape, self.signalDType)

            #Initialize Valid Subaperture Mask
            self.validSubApsFile = setFromConfig(self.conf, "validSubApsFile", "")
            self.validSubAps = np.ones(self.signalShape, dtype=self.signalDType)
            self.loadValidSubAps()
            #Initialize the reference slopes
            self.refSlopesFile = setFromConfig(self.conf, "refSlopesFile", "")
            self.refSlopes = np.zeros(self.signalShape, dtype=self.signalDType)
            self.loadRefSlopes()

        self.affinity = self.conf["affinity"]
        self.alive = True
        self.running = False
        functionsToRun = self.conf["functions"]
        self.workThreads = []
        for i, functionName in enumerate(functionsToRun):
            # Launch a separate thread
            workThread = threading.Thread(target=work, args = (self,functionName), daemon=True)
            # Start the thread
            workThread.start()
            # Set CPU affinity for the thread
            set_affinity((self.affinity+i)%os.cpu_count())
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
    
    def read(self):
        return self.signal.read()
    
    def readImage(self):
        return self.wfsShm.read()

    def setValidSubAps(self, validSubAps):
        self.validSubAps = validSubAps.astype(self.validSubAps)
        return
    
    def saveValidSubAps(self,filename=''):
        if filename == '':
            filename = self.validSubApsFile
        np.save(filename, self.validSubAps)
        return

    def loadValidSubAps(self,filename=''):
        #If no file given, first try reference slopes file
        if filename == '':
            filename = self.validSubApsFile
        #If we are still without a file, set zeros
        if filename == '':
            self.validSubAps = np.ones_like(self.validSubAps)
        else: #If we have a filename
            self.validSubAps = np.load(filename)
        return


    def takeRefSlopes(self):
        #Reset reference slopes to zero
        self.setRefSlopes(np.zeros_like(self.refSlopes))
        refSlopes = np.zeros_like(self.refSlopes)
        #Average 1000 slopes measurements
        for i in range(1000):
            refSlopes += self.read().astype(refSlopes.dtype)
        refSlopes /= 1000
        self.setRefSlopes(refSlopes)        
        return 

    def setRefSlopes(self, refSlopes):
        self.refSlopes = refSlopes.astype(self.signalDType)
        return
    
    def saveRefSlopes(self,filename=''):
        if filename == '':
            filename = self.refSlopesFile
        np.save(filename, self.refSlopes)
        return

    def loadRefSlopes(self,filename=''):
        #If no file given, first try reference slopes file
        if filename == '':
            filename = self.refSlopesFile
        #If we are still without a file, set zeros
        if filename == '':
            self.refSlopes = np.zeros_like(self.refSlopes)
        else: #If we have a filename
            self.refSlopes = np.load(filename)
        return
    
    def computeSignal(self):
        image = self.readImage().astype(self.signalDType)

        if self.wfsType == "PYWFS":
            if self.signalType == "slopes":
                p1,p2,p3,p4 = image[self.p1mask], image[self.p2mask], image[self.p3mask], image[self.p4mask]
                slope_signal = computeSlopesPYWFS(p1=p1,
                                                    p2=p2,
                                                    p3=p3,
                                                    p4=p4,
                                                    flatNorm=self.flatNorm)
                self.signal.write(slope_signal)
                self.signal2D.write(self.computeSlopeMap(slope_signal))
                
        elif self.wfsType == "SHWFS":
            if self.signalType == "slopes":
                threshold = np.mean(image)*self.shwfsContrast
                image[image < threshold] = 0
                slopes = computeSlopesSHWFS(image, 
                                                     self.refSlopes, 
                                                     self.subApSpacing,
                                                     self.offsetX,
                                                     self.offsetY)
                self.signal.write(slopes*self.validSubAps)
        return
    
    def setPupils(self, pupilLocs, pupilRadius):
        self.pupilLocs = pupilLocs
        self.pupilRadius = pupilRadius
        self.computePupilsMask()
        if self.signalType == "slopes":
            self.signalSize = np.count_nonzero(self.pupilMask)//2
            slopemask =  self.pupilMask[self.pupilLocs[0][1]-self.pupilRadius+1:self.pupilLocs[0][1]+self.pupilRadius, 
                                        self.pupilLocs[0][0]-self.pupilRadius+1:self.pupilLocs[0][0]+self.pupilRadius] > 0
            self.layout = np.concatenate([slopemask, slopemask], axis=1)
            self.signal = ImageSHM("signal", (self.signalSize,), self.signalDType)
            self.signal2D = ImageSHM("signal2D", (self.layout.shape[0], self.layout.shape[1]), self.signalDType)
            
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

    def computeSlopeMap(self, signal, layout=None):
        if layout is None and isinstance(self.layout, np.ndarray):
            layout = self.layout
        else:
            return -1
        curSignal2D = np.zeros(layout.shape)
        slopemask = layout[:,:layout.shape[1]//2]
        curSignal2D[:,:layout.shape[1]//2][slopemask] = signal[:signal.size//2]
        curSignal2D[:,layout.shape[1]//2:][slopemask] = signal[signal.size//2:]
        return curSignal2D
    
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

    slopes = SlopesProcess(conf=conf)
    slopes.start()

    l = Listener(slopes, port= int(args.port))
    while l.running:
        l.listen()
        time.sleep(1e-3)