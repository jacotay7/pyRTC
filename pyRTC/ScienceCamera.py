"""
Science Camera Superclass
"""
from pyRTC.Pipeline import ImageSHM
from pyRTC.pyRTCComponent import *
from pyRTC.utils import *
import numpy as np
import matplotlib.pyplot as plt

class ScienceCamera(pyRTCComponent):
    """
    A pyRTCComponent which represents a Science Camera. This is a general class which is 
    reponsible for all components of imaging which are common to all Science Cameras. This
    class should be used by defining a child class held in pyRTC.hardware, which overwrites
    the relevant functions which actual hardware connectivity code. The child class can call its parent
    implementations in order to make use of the code which sets the relevant parameters, write to shared
    memory, etc... or they can overwrite them completely. See hardware/SpinnakerScienceCam.py for an example.

    Config
    ------
    name : str
        Name of the camera.
    width : int
        Width of the image. Required.
    height : int
        Height of the image. Required.
    darkCount : int
        Number of dark frames to average. Required.
    integration : int
        Integration length. Required.
    darkFile : str, optional
        File to save the dark frames. Default is "".
    modelFile : str, optional
        File to save the model PSF. Default is "".

    Attributes
    ----------
    name : str
        Name of the camera.
    imageShape : tuple
        Shape of the image.
    imageRawDType : type
        Data type of the raw image.
    imageDType : type
        Data type of the image.
    psfLongDtype : type
        Data type of the long exposure PSF.
    psfShort : ImageSHM
        Shared memory object for the short exposure PSF.
    psfLong : ImageSHM
        Shared memory object for the long exposure PSF.
    strehlShm : ImageSHM
        Shared memory object for the Strehl ratio.
    tipTiltShm : ImageSHM
        Shared memory object for the tip-tilt.
    data : numpy.ndarray
        Data array for the image.
    dark : numpy.ndarray
        Dark frame.
    darkCount : int
        Number of dark frames to average.
    darkFile : str
        File to save the dark frames.
    model : numpy.ndarray
        Model PSF.
    modelFile : str
        File to save the model PSF.
    strehl_ratio : float
        Strehl ratio.
    peak_dist : float
        Peak distance.
    integrationLength : int
        Integration length.
    roiWidth : int
        Width of the region of interest.
    roiHeight : int
        Height of the region of interest.
    roiLeft : int
        Left coordinate of the region of interest.
    roiTop : int
        Top coordinate of the region of interest.
    exposure : int
        Exposure time.
    binning : int
        Binning factor.
    gain : int
        Gain setting.
    bitDepth : int
        Bit depth setting.
    """
    def __init__(self, conf) -> None:

        self.name = conf["name"]
        self.imageShape = (conf["width"], conf["height"])
        self.imageRawDType = np.uint16
        self.imageDType = np.int32
        self.psfLongDtype = np.float64
        
        self.psfShort = ImageSHM("psfShort", self.imageShape, self.imageDType)
        self.psfLong = ImageSHM("psfLong", self.imageShape, self.psfLongDtype)
        self.strehlShm = ImageSHM("strehl", (1,), float)
        self.tipTiltShm = ImageSHM("tiptilt", (1,), float)

        self.data = np.zeros(self.imageShape, dtype=self.imageRawDType)
        self.dark = np.zeros(self.imageShape, dtype=self.imageDType)
        self.darkCount = conf["darkCount"]
        self.darkFile = setFromConfig(conf, "darkFile", "")
        self.model = np.zeros(self.imageShape, dtype=self.psfLongDtype)
        self.modelFile = setFromConfig(conf, "modelFile", "")
        self.strehl_ratio = 0
        self.peak_dist = 0

        self.loadDark()
        self.loadModelPSF()

        self.integrationLength = conf["integration"]
        super().__init__(conf)
    
    def setRoi(self, roi):
        """
        Set the region of interest (ROI).

        Parameters
        ----------
        roi : tuple
            Tuple containing (width, height, left, top) of the ROI.
        """
        self.roiWidth = roi[0]
        self.roiHeight = roi[1]
        self.roiLeft = roi[2]
        self.roiTop = roi[3]
        return

    def setExposure(self, exposure):
        """
        Set the exposure time.

        Parameters
        ----------
        exposure : int
            Exposure time to set.
        """
        self.exposure = exposure
        return
    
    def setBinning(self, binning):
        """
        Set the binning factor.

        Parameters
        ----------
        binning : int
            Binning factor to set.
        """
        self.binning = binning
        return
    
    def setGain(self, gain):
        """
        Set the gain.

        Parameters
        ----------
        gain : int
            Gain to set.
        """
        self.gain = gain
        return
    
    def setGamma(self, gamma):
        """
        Set the gamma.

        Parameters
        ----------
        gamma : float
            Gamma to set.
        """
        self.gamma = gamma
        return
    
    def setBitDepth(self, bitDepth):
        """
        Set the bit depth.

        Parameters
        ----------
        bitDepth : int
            Bit depth to set.
        """
        self.bitDepth = bitDepth
        return
    
    def setIntegrationLength(self, integrationLength):
        """
        Set the integration length.

        Parameters
        ----------
        integrationLength : int
            Integration length to set.
        """
        self.integrationLength = integrationLength
        return
    
    def expose(self):
        """
        Perform a single exposure.
        """
        self.psfShort.write(self.data.astype(self.imageDType) - self.dark)
        return

    def integrate(self):
        """
        Perform multiple exposures and integrate the results. Number of frames set by integrationLength.
        """
        x = np.zeros(self.data.shape)
        for i in range(self.integrationLength):
            x += self.read().astype(x.dtype)
        self.psfLong.write(x/self.integrationLength)
        return 

    def read(self, block = True):
        """
        Read the current short exposure PSF.

        Returns
        -------
        numpy.ndarray
            Current short exposure PSF.
        """
        if block:
            return self.psfShort.read()
        return self.psfShort.read_noblock()
    
    def readLong(self):
        """
        Read the current long exposure PSF.

        Returns
        -------
        numpy.ndarray
            Current long exposure PSF.
        """
        return self.psfLong.read()
    
    def takeDark(self):
        """
        Take dark frames and average them to create a dark frame. 
        Number of exposures to average set by darkCount parameter.
        """
        self.setDark(np.zeros_like(self.dark))
        dark = np.zeros(self.imageShape, dtype=np.float64)
        for i in range(self.darkCount):
            dark += self.read().astype(np.float64)
        dark /= self.darkCount
        self.setDark(dark)        
        return 

    def setDark(self, dark):
        """
        Set the dark frame.

        Parameters
        ----------
        dark : numpy.ndarray
            Dark frame to set.
        """
        self.dark = dark.astype(self.imageDType)
        return
    
    def saveDark(self,filename=''):
        """
        Save the dark frame to a file.

        Parameters
        ----------
        filename : str, optional
            File to save the dark frame to. If not specified, uses the configured darkFile.
        """
        if filename == '':
            filename = self.darkFile
        np.save(filename, self.dark)
        return
    
    def loadDark(self,filename=''):
        """
        Load the dark frame from a file.

        Parameters
        ----------
        filename : str, optional
            File to load the dark frame from. If not specified, uses the configured darkFile.
        """
        #If no file given, first try dark file
        if filename == '':
            filename = self.darkFile
        #If we are still without a file, set zeros
        if filename == '':
            self.dark = np.zeros_like(self.dark)
        else: #If we have a filename
            self.dark = np.load(filename)
        return
    
    def takeModelPSF(self):
        """
        Capture the current long exposure PSF as the model PSF.
        """
        self.model = self.readLong()
        return

    def setModelPSF(self, model):
        """
        Set the model PSF.

        Parameters
        ----------
        model : numpy.ndarray
            Model PSF to set.
        """
        self.model = model.astype(self.psfLongDtype)
        return
    
    def saveModelPSF(self,filename=''):
        """
        Save the model PSF to a file.

        Parameters
        ----------
        filename : str, optional
            File to save the model PSF to. If not specified, uses the configured modelFile.
        """
        if filename == '':
            filename = self.modelFile
        np.save(filename, self.model)
        return
    
    def loadModelPSF(self,filename=''):
        """
        Load the model PSF from a file.

        Parameters
        ----------
        filename : str, optional
            File to load the model PSF from. If not specified, uses the configured modelFile.
        """
        #If no file given, first try dark file
        if filename == '':
            filename = self.modelFile
        #If we are still without a file, set zeros
        if filename == '':
            self.model = np.zeros_like(self.model)
        else: #If we have a filename
            self.model = np.load(filename)
        return

    def computeStrehl(self, median_filter_size = 1, gaussian_sigma = 0):
        """
        Compute the rough Strehl ratio and tip tilt offset. These values are reference to the modelPSF.
        If your model PSF is taken empirically, then the Strehl ratio is not absolute, and should only be
        used as a relative measurement for focal plane feedback.

        Parameters
        ----------
        median_filter_size : int, optional
            Size of the median filter to apply. Default is 1.
        gaussian_sigma : float, optional
            Sigma for the Gaussian filter. Default is 0.

        Returns
        -------
        float
            Strehl ratio.
        """

        model = clean_image_for_strehl(self.model, 
                                       median_filter_size = median_filter_size, 
                                       gaussian_sigma = gaussian_sigma)

        current = clean_image_for_strehl(self.readLong(), 
                                         median_filter_size = median_filter_size, 
                                         gaussian_sigma = gaussian_sigma)

        self.strehl_ratio = np.max(current) / np.max(model)
        self.peak_dist = np.linalg.norm(centroid(current) - centroid(self.model))

        self.strehlShm.write(np.array([self.strehl_ratio], dtype=float))
        self.tipTiltShm.write(np.array([self.peak_dist], dtype=float))

        return self.strehl_ratio

    def plot(self):
        """
        Plot the current short exposure PSF.
        """
        arr = self.read()
        plt.imshow(arr, cmap = 'inferno', origin='lower')
        plt.colorbar()
        plt.show()
        return