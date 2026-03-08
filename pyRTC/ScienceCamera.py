"""Science-camera abstractions and common image-quality telemetry.

This module defines the base class used by pyRTC science-camera adapters. It
handles the shared-memory products that downstream tools expect, including short
and long exposure PSFs, Strehl ratio estimates, and tip-tilt telemetry, while
leaving camera-specific acquisition details to hardware subclasses.
"""

import matplotlib.pyplot as plt
import numpy as np

from pyRTC.logging_utils import ensure_logging_configured, get_logger
from pyRTC.Pipeline import ImageSHM
from pyRTC.Pipeline import launchComponent
from pyRTC.pyRTCComponent import pyRTCComponent
from pyRTC.utils import centroid, clean_image_for_strehl, setFromConfig


logger = get_logger(__name__)

class ScienceCamera(pyRTCComponent):
    """
    Base class for cameras that produce science images and image-quality metrics.

    ``ScienceCamera`` centralizes the parts of imaging that are shared across
    real and synthetic science-camera backends: SHM publication, dark/model PSF
    handling, long-exposure accumulation, and simple Strehl/tip-tilt telemetry.
    Subclasses are expected to implement the device-facing acquisition logic and
    then call the parent methods so the standard pyRTC products stay updated.

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
        try:
            ensure_logging_configured(app_name="pyrtc", component_name=self.__class__.__name__)
            self.logger = get_logger(f"{self.__class__.__module__}.{self.__class__.__name__}")
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
            self.logger.info(
                "Initialized science camera name=%s image_shape=%s integration=%s",
                self.name,
                self.imageShape,
                self.integrationLength,
            )
        except Exception:
            logger.exception("Failed to initialize science camera")
            raise
    
    def setRoi(self, roi):
        """
        Set the region of interest (ROI).

        Parameters
        ----------
        roi : tuple
            Tuple containing (width, height, left, top) of the ROI.
        """
        try:
            self.roiWidth = roi[0]
            self.roiHeight = roi[1]
            self.roiLeft = roi[2]
            self.roiTop = roi[3]
            self.logger.info("Set ROI width=%s height=%s left=%s top=%s", *roi)
        except Exception:
            logger.exception("Failed to set ROI from %s", roi)
            raise
        return

    def setExposure(self, exposure):
        """
        Set the exposure time.

        Parameters
        ----------
        exposure : int
            Exposure time to set.
        """
        try:
            self.exposure = exposure
            self.logger.info("Set exposure to %s", exposure)
        except Exception:
            logger.exception("Failed to set exposure to %s", exposure)
            raise
        return
    
    def setBinning(self, binning):
        """
        Set the binning factor.

        Parameters
        ----------
        binning : int
            Binning factor to set.
        """
        try:
            self.binning = binning
            self.logger.info("Set binning to %s", binning)
        except Exception:
            logger.exception("Failed to set binning to %s", binning)
            raise
        return
    
    def setGain(self, gain):
        """
        Set the gain.

        Parameters
        ----------
        gain : int
            Gain to set.
        """
        try:
            self.gain = gain
            self.logger.info("Set gain to %s", gain)
        except Exception:
            logger.exception("Failed to set gain to %s", gain)
            raise
        return
    
    def setGamma(self, gamma):
        """
        Set the gamma.

        Parameters
        ----------
        gamma : float
            Gamma to set.
        """
        try:
            self.gamma = gamma
            self.logger.info("Set gamma to %s", gamma)
        except Exception:
            logger.exception("Failed to set gamma to %s", gamma)
            raise
        return
    
    def setBitDepth(self, bitDepth):
        """
        Set the bit depth.

        Parameters
        ----------
        bitDepth : int
            Bit depth to set.
        """
        try:
            self.bitDepth = bitDepth
            self.logger.info("Set bit depth to %s", bitDepth)
        except Exception:
            logger.exception("Failed to set bit depth to %s", bitDepth)
            raise
        return
    
    def setIntegrationLength(self, integrationLength):
        """
        Set the integration length.

        Parameters
        ----------
        integrationLength : int
            Integration length to set.
        """
        try:
            self.integrationLength = integrationLength
            self.logger.info("Set integration length to %s", integrationLength)
        except Exception:
            logger.exception("Failed to set integration length to %s", integrationLength)
            raise
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
            return self.psfShort.read(RELEASE_GIL = True)
        return self.psfShort.read_noblock()
    
    def readLong(self):
        """
        Read the current long exposure PSF.

        Returns
        -------
        numpy.ndarray
            Current long exposure PSF.
        """
        return self.psfLong.read(RELEASE_GIL = True)
    
    def takeDark(self):
        """
        Take dark frames and average them to create a dark frame. 
        Number of exposures to average set by darkCount parameter.
        """
        try:
            if self.darkCount < 1:
                raise ValueError("darkCount must be at least 1 to acquire a dark frame")
            self.logger.info("Taking science camera dark frame using %s exposures", self.darkCount)
            self.setDark(np.zeros_like(self.dark))
            dark = np.zeros(self.imageShape, dtype=np.float64)
            for _ in range(self.darkCount):
                dark += self.read().astype(np.float64)
            dark /= self.darkCount
            self.setDark(dark)
            self.logger.info("Completed science camera dark frame acquisition")
        except Exception:
            logger.exception("Failed to acquire science camera dark frame")
            raise
        return 

    def setDark(self, dark):
        """
        Set the dark frame.

        Parameters
        ----------
        dark : numpy.ndarray
            Dark frame to set.
        """
        try:
            self.dark = dark.astype(self.imageDType)
            self.logger.info("Updated science camera dark frame")
        except Exception:
            logger.exception("Failed to update science camera dark frame")
            raise
        return
    
    def saveDark(self,filename=''):
        """
        Save the dark frame to a file.

        Parameters
        ----------
        filename : str, optional
            File to save the dark frame to. If not specified, uses the configured darkFile.
        """
        try:
            if filename == '':
                filename = self.darkFile
            if filename == '':
                raise ValueError("No dark frame filename provided")
            np.save(filename, self.dark)
            self.logger.info("Saved science camera dark frame to %s", filename)
        except Exception:
            logger.exception("Failed to save science camera dark frame to %s", filename or self.darkFile)
            raise
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
        try:
            if filename == '':
                filename = self.darkFile
            if filename == '':
                self.dark = np.zeros_like(self.dark)
                logger.info("No science camera dark frame file configured; using zeros")
            else:
                self.dark = np.load(filename)
                self.logger.info("Loaded science camera dark frame from %s", filename)
        except Exception:
            logger.exception("Failed to load science camera dark frame from %s", filename or self.darkFile)
            raise
        return
    
    def takeModelPSF(self):
        """
        Capture the current long exposure PSF as the model PSF.
        """
        try:
            self.model = self.readLong()
            self.logger.info("Captured model PSF from current long-exposure image")
        except Exception:
            logger.exception("Failed to capture model PSF")
            raise
        return

    def setModelPSF(self, model):
        """
        Set the model PSF.

        Parameters
        ----------
        model : numpy.ndarray
            Model PSF to set.
        """
        try:
            self.model = model.astype(self.psfLongDtype)
            self.logger.info("Updated model PSF")
        except Exception:
            logger.exception("Failed to update model PSF")
            raise
        return
    
    def saveModelPSF(self,filename=''):
        """
        Save the model PSF to a file.

        Parameters
        ----------
        filename : str, optional
            File to save the model PSF to. If not specified, uses the configured modelFile.
        """
        try:
            if filename == '':
                filename = self.modelFile
            if filename == '':
                raise ValueError("No model PSF filename provided")
            np.save(filename, self.model)
            self.logger.info("Saved model PSF to %s", filename)
        except Exception:
            logger.exception("Failed to save model PSF to %s", filename or self.modelFile)
            raise
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
        try:
            if filename == '':
                filename = self.modelFile
            if filename == '':
                self.model = np.zeros_like(self.model)
                logger.info("No model PSF file configured; using zeros")
            else:
                self.model = np.load(filename)
                self.logger.info("Loaded model PSF from %s", filename)
        except Exception:
            logger.exception("Failed to load model PSF from %s", filename or self.modelFile)
            raise
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
        try:
            arr = self.read()
            plt.imshow(arr, cmap = 'inferno', origin='lower')
            plt.colorbar()
            plt.show()
            self.logger.info("Plotted science camera image")
        except Exception:
            logger.exception("Failed to plot science camera image")
            raise
        return

if __name__ == "__main__":

    launchComponent(ScienceCamera, "psf", start = True)