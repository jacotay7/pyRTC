"""
Slopes Superclass
"""
from pyRTC.Pipeline import *
from pyRTC.utils import *
from pyRTC.pyRTCComponent import *
import argparse
import os 
import numpy as np
import matplotlib.pyplot as plt
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
            if start_j+intN <= image.shape[1] and start_i+intN <= image.shape[0]:
                sub_im = image[start_i:start_i+intN,
                            start_j:start_j+intN]
                norm = np.sum(sub_im)
                #Only compute centroid if we have flux
                if norm > 0: 
                    slopes[i,j] = np.sum(xvals*sub_im)/norm
                    slopes[i+numRegions,j] = np.sum(xvals*sub_im.T)/norm
    return slopes - unaberratedSlopes

class SlopesProcess(pyRTCComponent):
    """
    A class to handle real-time slope computation for wavefront sensors.

    Config
    ------
    type : str
        Type of the WFS ("PYWFS" or "SHWFS").
    signalType : str
        Type of signal ("slopes").
    imageNoise : float, optional
        Image noise. Default is 0.0.
    centralObscurationRatio : float, optional
        Central obscuration ratio. Default is 0.0.
    flatNorm : float, optional
        Normalization factor for the flat. Required for "PYWFS" with "slopes" signalType.
    pupils : list of str, optional
        List of pupil locations in "x,y" format. Required for "PYWFS".
    pupilsRadius : int, optional
        Radius of the pupils. Required for "PYWFS".
    contrast : float, optional
        Contrast for "SHWFS". Default is 0.
    subApSpacing : float, optional
        Sub-aperture spacing for "SHWFS".
    subApOffsetX : float, optional
        Sub-aperture offset in X direction for "SHWFS".
    subApOffsetY : float, optional
        Sub-aperture offset in Y direction for "SHWFS".
    refSlopeCount : int, optional
        Number of reference slopes for averaging. Default is 1000.
    validSubApsFile : str, optional
        File containing valid sub-aperture mask. Default is "".
    refSlopesFile : str, optional
        File containing reference slopes. Default is "".

    Attributes
    ----------
    confWFS : dict
        Wavefront sensor configuration.
    name : str
        Name of the process.
    imageShape : tuple
        Shape of the WFS image.
    conf : dict
        Slopes configuration.
    wfsMeta : numpy.ndarray
        Metadata of the WFS image.
    imageDType : type
        Data type of the WFS image.
    wfsShm : ImageSHM
        Shared memory object for the WFS image.
    signalDType : type
        Data type of the signal.
    imageNoise : float
        Image noise.
    centralObscurationRatio : float
        Central obscuration ratio.
    wfsType : str
        Type of the WFS.
    signalType : str
        Type of signal.
    validSubAps : numpy.ndarray or None
        Valid sub-aperture mask.
    shwfsContrast : float
        Contrast for "SHWFS".
    subApSpacing : float
        Sub-aperture spacing for "SHWFS".
    numRegions : int
        Number of regions for "SHWFS".
    offsetX : float
        Sub-aperture offset in X direction for "SHWFS".
    offsetY : float
        Sub-aperture offset in Y direction for "SHWFS".
    refSlopeCount : int
        Number of reference slopes for averaging.
    signal2DSize : int
        Size of the 2D signal.
    signal2DShape : tuple
        Shape of the 2D signal.
    validSubApsFile : str
        File containing valid sub-aperture mask.
    signalSize : int
        Size of the signal.
    signalShape : tuple
        Shape of the signal.
    signal : ImageSHM
        Shared memory object for the signal.
    signal2D : ImageSHM
        Shared memory object for the 2D signal.
    refSlopesFile : str
        File containing reference slopes.
    refSlopes : numpy.ndarray
        Reference slopes.
    flatNorm : float
        Normalization factor for the flat.
    pupilLocs : list of tuple
        List of pupil locations.
    pupilRadius : int
        Radius of the pupils.
    pupilMask : numpy.ndarray
        Mask of the pupils.
    p1mask : numpy.ndarray
        Mask for pupil 1.
    p2mask : numpy.ndarray
        Mask for pupil 2.
    p3mask : numpy.ndarray
        Mask for pupil 3.
    p4mask : numpy.ndarray
        Mask for pupil 4.
    """
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
        self.imageNoise = setFromConfig(self.conf,"imageNoise", 0.0)
        self.centralObscurationRatio = setFromConfig(self.conf,"centralObscurationRatio", 0.0)

        self.wfsType = self.conf["type"].lower() 
        self.signalType = self.conf["signalType"] 
        self.validSubAps = None

        if self.wfsType == "pywfs":
            #Check if we have specified a pupil validSubAps
            # self.signal = ImageSHM("signal", self.imageShape, self.signalDType)

            if "pupils" in self.conf.keys():
                pupilLocs = [(int(x.split(',')[1]), int(x.split(',')[0])) for x in self.conf["pupils"]]
                self.setPupils(pupilLocs, self.conf["pupilsRadius"])
            else: #Default Pupil validSubAps
                a, b = int(0.25*self.imageShape[0]), int(0.75*self.imageShape[0])
                c, d = int(0.25*self.imageShape[1]), int(0.75*self.imageShape[1])
                r = min(self.imageShape[0]-b,self.imageShape[1]-d)
                self.setPupils([(a,c), (a,d), (b,c), (b,d)], r)
            if self.signalType == 'slopes':
                #Set normalization
                self.flatNorm = self.conf["flatNorm"]

        elif self.wfsType == "shwfs":

            self.shwfsContrast = setFromConfig(self.conf, "contrast", 0)
            self.subApSpacing = self.conf["subApSpacing"]
            self.numRegions = self.imageShape[0]//int(np.round(self.subApSpacing,0))
            self.offsetX = self.conf["subApOffsetX"]
            self.offsetY = self.conf["subApOffsetY"]
            self.refSlopeCount = setFromConfig(self.conf, "refSlopeCount", 1000)
            self.signal2DSize = int(2*self.numRegions**2)
            self.signal2DShape = (2*self.numRegions,self.numRegions)

            #Initialize Valid Subaperture Mask
            self.validSubApsFile = setFromConfig(self.conf, "validSubApsFile", "")
            self.validSubAps = np.ones(self.signal2DShape, dtype=bool)
            self.loadValidSubAps()

            self.signalSize = np.sum(self.validSubAps)
            self.signalShape = (self.signalSize,)

            print(f'subApSpacing: {self.subApSpacing}')
            print(f'numRegions: {self.numRegions}')
            print(f'offsetX: {self.offsetX}')
            print(f'offsetY: {self.offsetY}')
            print(f'signalSize: {self.signalSize}')
            print(f'signalShape: {self.signalShape}')
            print(f'signalDType: {self.signalDType}')

            self.signal = ImageSHM("signal", self.signalShape, self.signalDType)
            self.signal2D = ImageSHM("signal2D", self.signal2DShape, self.signalDType)
            
            #Initialize the reference slopes
            self.refSlopesFile = setFromConfig(self.conf, "refSlopesFile", "")
            self.refSlopes = np.zeros(self.signal2DShape, dtype=self.signalDType)
            self.loadRefSlopes()

        super().__init__(self.conf)
    
    def read(self, block = True):
        """
        Read the current signal.

        Returns
        -------
        numpy.ndarray
            Current signal.
        """
        if block:
            return self.signal.read()
        return self.signal.read_noblock()
    
    def readImage(self, block=True):
        """
        Read the current WFS image.

        Returns
        -------
        numpy.ndarray
            Current WFS image.
        """
        if block:
            return self.wfsShm.read()
        return self.wfsShm.read_noblock()

    def setValidSubAps(self, validSubAps):
        """
        Set the valid sub-aperture mask. Converts to boolean if not already

        Parameters
        ----------
        validSubAps : numpy.ndarray
            Valid sub-aperture mask.
        """
        self.validSubAps = validSubAps.astype(bool)
        return
    
    def saveValidSubAps(self,filename=''):
        """
        Save the valid sub-aperture mask to a file.

        Parameters
        ----------
        filename : str, optional
            File to save the valid sub-aperture mask to. If not specified, uses the configured validSubApsFile.
        """
        if filename == '':
            filename = self.validSubApsFile
        np.save(filename, self.validSubAps)
        return

    def loadValidSubAps(self,filename=''):
        """
        Load the valid sub-aperture mask from a file.

        Parameters
        ----------
        filename : str, optional
            File to load the valid sub-aperture mask from. If not specified, uses the configured validSubApsFile.
        """
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
        """
        Take reference slopes by averaging multiple slope measurements. Number of measurements
        set by refSlopeCount variable.
        """
        #Reset reference slopes to zero
        self.setRefSlopes(np.zeros_like(self.refSlopes))
        refSlopes = np.zeros_like(self.refSlopes)
        #Average self.refSlopeCount slopes measurements
        for i in range(self.refSlopeCount):
            cur_slopes = self.read().astype(refSlopes.dtype)
            refSlopes += self.computeSignal2D(cur_slopes)
        refSlopes /= self.refSlopeCount
        self.setRefSlopes(refSlopes)        
        return 

    def setRefSlopes(self, refSlopes):
        """
        Set the reference slopes.

        Parameters
        ----------
        refSlopes : numpy.ndarray
            Reference slopes.
        """
        self.refSlopes = refSlopes.astype(self.signalDType)
        return
    
    def saveRefSlopes(self,filename=''):
        """
        Save the reference slopes to a file.

        Parameters
        ----------
        filename : str, optional
            File to save the reference slopes to. If not specified, uses the configured refSlopesFile.
        """
        if filename == '':
            filename = self.refSlopesFile
        np.save(filename, self.refSlopes)
        return

    def loadRefSlopes(self,filename=''):
        """
        Load the reference slopes from a file.

        Parameters
        ----------
        filename : str, optional
            File to load the reference slopes from. If not specified, uses the configured refSlopesFile.
        """
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
        """
        Compute the signal from the WFS image.
        """
        image = self.readImage().astype(self.signalDType)
        if self.signalType == "slopes":
            if self.wfsType == "pywfs":
                p1,p2,p3,p4 = image[self.p1mask], image[self.p2mask], image[self.p3mask], image[self.p4mask]
                slope_signal = computeSlopesPYWFS(p1=p1,
                                                    p2=p2,
                                                    p3=p3,
                                                    p4=p4,
                                                    flatNorm=self.flatNorm)
                
                
                    
            elif self.wfsType == "shwfs":
                
                # threshold = np.std(image[image < np.mean(image)])*self.shwfsContrast
                threshold = self.imageNoise*self.shwfsContrast
                image[image < threshold] = 0
                slopes = computeSlopesSHWFS(image, 
                                                    self.refSlopes, 
                                                    self.subApSpacing,
                                                    self.offsetX,
                                                    self.offsetY)
                slope_signal = slopes[self.validSubAps]

            self.signal.write(slope_signal)
            self.signal2D.write(self.computeSignal2D(slope_signal))
        return
    
    def computeImageNoise(self):
        """
        Compute the image noise. Useful to set a good SNR cutoff for SHWFS
        """
        img = self.readImage()
        if img[img < 0].size > 0:
            self.imageNoise = compute_fwhm_dark_subtracted_image(img)/2
        else:
            print("Image is not dark subtracted")
        return

    def setPupils(self, pupilLocs, pupilRadius):
        """
        Set the pupils' locations and radius. First computes a Pupil Mask, then generates slope mask 
        and sets up SHMS of the correct sizes.

        Parameters
        ----------
        pupilLocs : list of tuple
            List of pupil locations.
        pupilRadius : int
            Radius of the pupils.
        """
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
        """
        Compute the mask for the pupils. Assumes circular aperture with obstruction ratio 
        set by the centralObscurationRatio parameter.
        """
        self.pupilMask = np.zeros(self.imageShape)

        pupilTemplate = generate_circular_aperture_mask(int(np.ceil(2*self.pupilRadius)),
                                                        self.pupilRadius, 
                                                        self.centralObscurationRatio)        
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

    def plotPupils(self):
        """
        Plot the pupil mask to see if its right.
        """
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

    def computeSignal2D(self, signal, validSubAps=None):
        """
        Compute the 2D signal from the valid sub-aperture mask.

        Parameters
        ----------
        signal : numpy.ndarray
            Signal to process.
        validSubAps : numpy.ndarray, optional
            Valid sub-aperture mask. If not provided, uses the current valid sub-aperture mask.

        Returns
        -------
        numpy.ndarray
            2D signal.
        """
        if validSubAps is None and isinstance(self.validSubAps, np.ndarray):
            validSubAps = self.validSubAps
        else:
            return -1
        curSignal2D = np.zeros(validSubAps.shape)
        if self.wfsType == "pywfs":
            slopemask = validSubAps[:,:validSubAps.shape[1]//2]
            curSignal2D[:,:validSubAps.shape[1]//2][slopemask] = signal[:signal.size//2]
            curSignal2D[:,validSubAps.shape[1]//2:][slopemask] = signal[signal.size//2:]
        else:
            curSignal2D[validSubAps] = signal
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