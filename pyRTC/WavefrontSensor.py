"""Wavefront-sensor abstractions and common image pre-processing kernels.

This module defines the base class used by pyRTC wavefront-sensor adapters and
includes small image-processing helpers that are hot enough to warrant Numba
acceleration. Hardware-specific sensors subclass ``WavefrontSensor`` and reuse
its SHM publication, dark handling, and optional geometric pre-processing.
"""

import matplotlib.pyplot as plt
import numpy as np
from numba import jit, prange

from pyRTC.logging_utils import get_logger
from pyRTC.Pipeline import ImageSHM, launchComponent
from pyRTC.pyRTCComponent import pyRTCComponent
from pyRTC.utils import setFromConfig


logger = get_logger(__name__)

@jit(nopython=True, nogil=True, cache=True, fastmath=True)
def downsample_int32_image_jit(image, N):
    """
    Numba-optimized function to downsample a 2D int32 NumPy array by a factor N, returning int32 output.

    Parameters:
    - image: 2D NumPy array of int32 with shape (H, W)
    - N: int, downsampling factor

    Returns:
    - downsampled_image: 2D NumPy array of int32 with shape (H//N, W//N)
    """
    H, W = image.shape

    # Calculate padding sizes if H or W is not divisible by N
    pad_H = (-H) % N
    pad_W = (-W) % N

    # Pad the image if necessary to make dimensions divisible by N
    if pad_H > 0 or pad_W > 0:
        # Create a new array with zeros
        H_padded = H + pad_H
        W_padded = W + pad_W
        image_padded = np.zeros((H_padded, W_padded), dtype=np.int32)
        image_padded[:H, :W] = image
    else:
        image_padded = image
        H_padded, W_padded = H, W

    # Initialize the output array
    out_H = H_padded // N
    out_W = W_padded // N
    downsampled_image = np.zeros((out_H, out_W), dtype=np.int32)

    # Loop over the output array indices with Numba's parallel loops
    for i in range(out_H):
        for j in range(out_W):
            # Compute the sum over the N x N block
            sum_block = 0
            for di in range(N):
                for dj in range(N):
                    sum_block += image_padded[i*N + di, j*N + dj]
            # Compute the mean
            mean_value = sum_block / (N * N)
            # Round and cast to int32
            downsampled_image[i, j] = np.int32(round(mean_value))

    return downsampled_image

@jit(nopython=True, nogil=True, cache=True, fastmath=True, parallel=True)
def rotate_image_jit(image, angle_rad):
    """
    Numba-optimized parallel bilinear interpolation rotation.
    
    Parameters:
    - image: 2D NumPy array (int32 or float) with shape (H, W)
    - angle_rad: float, rotation angle in radians (positive = counter-clockwise)
    
    Returns:
    - rotated_image: 2D NumPy array with same shape and dtype as input
    """
    h, w = image.shape
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)
    
    # Center of rotation
    cx, cy = w / 2.0, h / 2.0
    
    # Output image (same size as input)
    rotated = np.zeros_like(image)
    
    for y in prange(h):
        for x in range(w):
            # Translate to center
            x_centered = x - cx
            y_centered = y - cy
            
            # Rotate (inverse transformation)
            x_orig = x_centered * cos_angle + y_centered * sin_angle + cx
            y_orig = -x_centered * sin_angle + y_centered * cos_angle + cy
            
            # Check if the source coordinates are within bounds
            if 0 <= x_orig < w-1 and 0 <= y_orig < h-1:
                # Bilinear interpolation
                x0, x1 = int(np.floor(x_orig)), int(np.ceil(x_orig))
                y0, y1 = int(np.floor(y_orig)), int(np.ceil(y_orig))
                
                # Ensure indices are within bounds
                if x1 >= w:
                    x1 = w - 1
                if y1 >= h:
                    y1 = h - 1
                
                # Interpolation weights
                wx = x_orig - x0
                wy = y_orig - y0
                
                # Bilinear interpolation
                val = (image[y0, x0] * (1 - wx) * (1 - wy) +
                       image[y0, x1] * wx * (1 - wy) +
                       image[y1, x0] * (1 - wx) * wy +
                       image[y1, x1] * wx * wy)
                
                rotated[y, x] = val
    
    return rotated

class WavefrontSensor(pyRTCComponent):
    """
    Base class for cameras that feed the wavefront-sensing pipeline.

    The class owns the common control-plane behavior for wavefront-sensor image
    sources: configuration, dark subtraction, optional downsampling and
    rotation, and publication of both raw and processed frames. Concrete sensor
    adapters in ``pyRTC.hardware`` are responsible for talking to vendor SDKs
    and filling ``self.data`` before delegating back to the base implementation.

    Config
    ------
    name : str
        The name of the wavefront sensor. Default "wavefrontSensor"
    width : int
        The width of the wavefront sensor image. Required.
    height : int
        The width of the wavefront sensor image.  Required.      
    darkCount : int
        Number of dark frames to average. Default 1000.
    darkFile : str
        Path to the dark frame file. Default, empty string.

    Attributes
    ----------
    imageShape : tuple
        The shape of the image (width, height).
    imageRawDType : data-type
        The data type for raw image.
    imageDType : data-type
        The data type for processed image.
    imageRaw : ImageSHM
        Shared memory object for raw image.
    image : ImageSHM
        Shared memory object for processed image.
    data : ndarray
        Array to store raw image data.
    dark : ndarray
        Array to store dark frame data.
    affinity : int
        The affinity configuration.
    roiWidth : int
        Width of the region of interest.
    roiHeight : int
        Height of the region of interest.
    roiLeft : int
        Left coordinate of the region of interest.
    roiTop : int
        Top coordinate of the region of interest.
    exposure : float
        Exposure time.
    binning : int
        Binning factor.
    gain : float
        Gain setting.
    bitDepth : int
        Bit depth of the image.

    Methods
    -------
    setRoi(roi)
        Sets the region of interest.
    setExposure(exposure)
        Sets the exposure time.
    setBinning(binning)
        Sets the binning factor.
    setGain(gain)
        Sets the gain.
    setBitDepth(bitDepth)
        Sets the bit depth.
    expose()
        Writes the current image data to shared memory.
    read()
        Reads the processed image data from shared memory.
    takeDark()
        Captures and sets the dark frame.
    setDark(dark)
        Sets the dark frame.
    saveDark(filename='')
        Saves the dark frame to a file.
    loadDark(filename='')
        Loads the dark frame from a file.
    plot()
        Plots the current image data.
    rotateImage(angle_deg)
        Rotates the current image data by the specified angle in degrees.
    """

    def __init__(self, conf: dict) -> None:
        """
        Constructs all the necessary attributes for the wavefront sensor object.

        Parameters
        ----------
        conf : dict
            Configuration dictionary for the wavefront sensor. Typically it will just be
            the "wfs" section of a pyRTC config.
        """
        try:
            super().__init__(conf)

            self.name = setFromConfig(conf, "name", "wavefrontSensor")
            self.width = setFromConfig(conf, "width", 1)
            self.height = setFromConfig(conf, "height", 1)
            self.darkCount = setFromConfig(conf, "darkCount", 1000)
            self.darkFile = setFromConfig(conf, "darkFile", "")
            self.downsampleFactor = setFromConfig(conf, "downsampleFactor", 0)
            self.rotationAngle = setFromConfig(conf, "rotationAngle", 0.0)

            self.imageRawShape = [self.width, self.height]
            self.imageRawDType = np.uint16
            self.imageDType = np.int32
            self.imageShape = [self.width, self.height]
            if self.downsampleFactor > 0:
                self.imageShape[0] = self.imageShape[0] // self.downsampleFactor
                self.imageShape[1] = self.imageShape[1] // self.downsampleFactor
            self.imageRaw = ImageSHM("wfsRaw", self.imageRawShape, self.imageRawDType, gpuDevice=self.gpuDevice, consumer=False)
            self.image = ImageSHM("wfs", self.imageShape, self.imageDType, gpuDevice=self.gpuDevice, consumer=False)
            self.register_output_stream("wfsRaw", self.imageRaw)
            self.register_output_stream("wfs", self.image, source_streams=["wfsRaw"], lineage_source="wfsRaw")

            self.data = np.zeros(self.imageShape, dtype=self.imageRawDType)
            self.dark = np.zeros(self.imageRawShape, dtype=self.imageDType)

            self.loadDark()
            self.logger.info(
                "Initialized wavefront sensor name=%s raw_shape=%s image_shape=%s downsample=%s rotation=%s",
                self.name,
                self.imageRawShape,
                self.imageShape,
                self.downsampleFactor,
                self.rotationAngle,
            )
        except Exception:
            logger.exception("Failed to initialize wavefront sensor")
            raise

        return
    
    def setRoi(self, roi):
        """
        Sets the region of interest (ROI) for the sensor.

        Parameters
        ----------
        roi : tuple
            A tuple containing (width, height, left, top) of the ROI.
        """
        try:
            self.roiWidth = roi[0]
            self.roiHeight = roi[1]
            self.roiLeft = roi[2]
            self.roiTop = roi[3]
            self.logger.info("Set ROI width=%s height=%s left=%s top=%s", *roi)
        except Exception:
            self.logger.exception("Failed to set ROI from %s", roi)
            raise

        return

    def setExposure(self, exposure: float) -> None:
        """
        Sets the exposure time for the sensor.

        Parameters
        ----------
        exposure : float
            Exposure time in whatever unit your camera uses.
        """
        try:
            self.exposure = exposure
            self.logger.info("Set exposure to %s", exposure)
        except Exception:
            self.logger.exception("Failed to set exposure to %s", exposure)
            raise

        return
    
    def setBinning(self, binning: int) -> None:
        """
        Sets the binning factor for the sensor.

        Parameters
        ----------
        binning : int
            Binning factor.
        """
        try:
            self.binning = binning
            self.logger.info("Set binning to %s", binning)
        except Exception:
            self.logger.exception("Failed to set binning to %s", binning)
            raise

        return
    
    def setGain(self, gain: float) -> None:
        """
        Sets the gain for the sensor.

        Parameters
        ----------
        gain : float
            Gain value.
        """
        try:
            self.gain = gain
            self.logger.info("Set gain to %s", gain)
        except Exception:
            self.logger.exception("Failed to set gain to %s", gain)
            raise
        return
    
    def setBitDepth(self, bitDepth: int) -> None:
        """
        Sets the bit depth for the sensor.

        Parameters
        ----------
        bitDepth : int
            Bit depth. pyRTC convention is this is the number of bits in the ADC,
            e.g., 8, 16, 12, 10.
        """
        try:
            self.bitDepth = bitDepth
            self.logger.info("Set bit depth to %s", bitDepth)
        except Exception:
            self.logger.exception("Failed to set bit depth to %s", bitDepth)
            raise
        return
    
    def expose(self) -> None:
        """
        Writes the current image data to shared memory. Both raw, and dark subtracted.
        
        Parameters
        ----------
        """
        self.write_stream("wfsRaw", self.data)
        img = self.data.astype(self.imageDType)
        
        # Apply dark subtraction
        processed_image = img - self.dark
        
        # Apply downsampling if configured
        if self.downsampleFactor > 0:
            processed_image = downsample_int32_image_jit(processed_image, self.downsampleFactor)
        
        # Apply rotation if specified
        if self.rotationAngle != 0.0:
            angle_rad = np.radians(self.rotationAngle)
            processed_image = rotate_image_jit(processed_image, angle_rad)
        
        # Write the processed image to shared memory
        self.write_stream("wfs", processed_image)
        return

    def read(self, block = True) -> None:
        """
        Reads the dark subtracted image data from shared memory.

        Returns
        -------
        ndarray
            Processed image data.
        """
        if block:
            return self.read_stream("wfs", RELEASE_GIL=self.RELEASE_GIL)
        else:
            return self.read_stream("wfs", block=False)
    
    def takeDark(self) -> None:
        """
        Captures and sets the dark frame.
        """
        try:
            if self.darkCount < 1:
                raise ValueError("darkCount must be at least 1 to acquire a dark frame")
            self.logger.info("Taking dark frame using %s exposures", self.darkCount)
            self.setDark(np.zeros_like(self.dark))
            dark = np.zeros(self.imageShape, dtype=np.float64)
            for _ in range(self.darkCount):
                dark += self.read().astype(np.float64)
            dark /= self.darkCount
            self.setDark(dark)
            self.logger.info("Completed dark frame acquisition")
        except Exception:
            self.logger.exception("Failed to acquire dark frame")
            raise
        return 

    def setDark(self, dark) -> None:
        """
        Sets the dark frame.

        Parameters
        ----------
        dark : ndarray
            Dark frame data.
        """
        try:
            self.dark = dark.astype(self.imageDType)
            self.logger.info("Updated dark frame")
        except Exception:
            self.logger.exception("Failed to update dark frame")
            raise
        return
    
    def saveDark(self,filename=''):
        """
        Saves the dark frame to a file.

        Parameters
        ----------
        filename : str, optional
            Filename to save the dark frame to. If not specified, uses the dark file path from the configuration.
        """
        try:
            if filename == '':
                filename = self.darkFile
            if filename == '':
                raise ValueError("No dark frame filename provided")
            np.save(filename, self.dark)
            self.logger.info("Saved dark frame to %s", filename)
        except Exception:
            self.logger.exception("Failed to save dark frame to %s", filename or self.darkFile)
            raise
        return
    
    def loadDark(self,filename=''):
        """
        Loads the dark frame from a file.

        Parameters
        ----------
        filename : str, optional
            Filename to load the dark frame from. If not specified, uses the dark file path from the configuration.
        """
        #If no file given, first try dark file
        try:
            if filename == '':
                filename = self.darkFile
            if filename == '':
                self.dark = np.zeros_like(self.dark)
                self.logger.info("No dark frame file configured; using zeros")
            else:
                self.dark = np.load(filename)
                self.logger.info("Loaded dark frame from %s", filename)
        except Exception:
            self.logger.exception("Failed to load dark frame from %s", filename or self.darkFile)
            raise
        return
    
    def plot(self) -> None:
        """
        Plots the current image data.
        """
        try:
            arr = self.read(block=False)
            plt.figure(figsize=(8,8))
            plt.imshow(arr, cmap = 'inferno', origin='lower')
            plt.colorbar()
            plt.show()
            self.logger.info("Plotted wavefront sensor image")
        except Exception:
            self.logger.exception("Failed to plot wavefront sensor image")
            raise
        return
    
    def rotateImage(self, angle_deg: float) -> np.ndarray:
        """
        Rotates the current image data by the specified angle.
        
        This method uses a high-performance numba JIT-compiled bilinear interpolation
        rotation algorithm that is significantly faster than scipy or opencv implementations
        while maintaining good image quality.

        Parameters
        ----------
        angle_deg : float
            Rotation angle in degrees. Positive values rotate counter-clockwise.

        Returns
        -------
        ndarray
            Rotated image data with the same shape and dtype as the original.
            
        Examples
        --------
        >>> wfs = WavefrontSensor(config)
        >>> rotated_img = wfs.rotateImage(45.0)  # Rotate 45 degrees counter-clockwise
        >>> rotated_img = wfs.rotateImage(-90.0) # Rotate 90 degrees clockwise
        """
        # Get the current image data
        try:
            current_image = self.read(block=False)
            angle_rad = np.radians(angle_deg)
            rotated_image = rotate_image_jit(current_image, angle_rad)
            self.logger.info("Rotated image by %s degrees", angle_deg)
            return rotated_image
        except Exception:
            self.logger.exception("Failed to rotate image by %s degrees", angle_deg)
            raise
    
if __name__ == "__main__":

    launchComponent(WavefrontSensor, "wfs", start = True)