"""Slope-processing kernels and the pyRTC slope extraction component.

This module turns wavefront-sensor camera frames into the residual slope or
signal vectors consumed by the AO loop. It includes optimized CPU and GPU
helpers for pyramid and Shack-Hartmann processing plus the ``SlopesProcess``
component that manages calibration data and SHM publication.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ['NUMBA_NUM_THREADS'] = '1'

import matplotlib.pyplot as plt
import numpy as np
from typing import Any
from numba import jit

from pyRTC.logging_utils import get_logger
from pyRTC.Pipeline import ImageSHM, clear_shms, gpu_torch_available, initExistingShm, launchComponent
from pyRTC.pyRTCComponent import pyRTCComponent
from pyRTC.utils import (
    compute_fwhm_dark_subtracted_image,
    generate_circular_aperture_mask,
    setFromConfig,
)

logger = get_logger(__name__)

def computeSlopesPYWFSTorch(image: Any,
                            p1Mask: Any,
                            p2Mask: Any,
                            p3Mask: Any,
                            p4Mask: Any,
                            numPixelsInPupils: int, 
                            slopes: Any,
                            refSlopes: Any):
    """Compute normalized pyramid-WFS slopes on a torch device.

    The function extracts the four pupil images selected by the provided masks,
    forms differential x/y slope channels, normalizes by the mean total pupil
    flux, and subtracts the stored reference slopes.
    """

    if not gpu_torch_available():
        raise ImportError("computeSlopesPYWFSTorch requires PyTorch. Install with 'pip install pyRTC[gpu]' or 'pip install torch'.")

    import torch

    # Ensure the image is in float format
    image = image.to(torch.float32)
    
    # Mask pupils out of the image
    p1 = image[p1Mask]
    p2 = image[p2Mask]
    p3 = image[p3Mask]
    p4 = image[p4Mask]
    
    # Sum pupils, saving partial sums to avoid recomputing later
    tmp1 = p1 + p2
    tmp2 = p3 + p4
    
    # Compute X slopes
    slopes[:numPixelsInPupils] = tmp1 - tmp2
    
    # Compute Y slopes
    slopes[numPixelsInPupils:] = (p1 + p3) - (p2 + p4)
    
    # Normalize slopes
    slopes = slopes / torch.mean(tmp1 + tmp2)
    
    # Subtract reference slopes
    return slopes - refSlopes

"""
Optimized for best performance with numpy only
All memory is preallocated.
"""
def computeSlopesPYWFSOptimNumpy(image:np.ndarray,
                            p1Mask:np.ndarray, 
                            p2Mask:np.ndarray,
                            p3Mask:np.ndarray, 
                            p4Mask:np.ndarray,
                            p1:np.ndarray, 
                            p2:np.ndarray,
                            p3:np.ndarray, 
                            p4:np.ndarray,
                            tmp1:np.ndarray,
                            tmp2:np.ndarray,
                            numPixelsInPupils:int, 
                            slopes:np.ndarray,
                            refSlopes:np.ndarray,
                        ):
    # Mask Pupils out of image and convert to floats
    p1 = image[p1Mask].astype(np.float32)
    p2 = image[p2Mask].astype(np.float32)
    p3 = image[p3Mask].astype(np.float32)
    p4 = image[p4Mask].astype(np.float32)
    # Sum Pupils, Saving partial sums to avoid recomputing later
    tmp1 = np.add(p1,p2)
    tmp2 = np.add(p3,p4)
    # Compute X slopes
    slopes[:numPixelsInPupils] = np.subtract(tmp1,tmp2)
    # Compute Y slopes
    slopes[numPixelsInPupils:] = np.subtract(np.add(p1,p3),np.add(p2,p4))
    # Normalize slopes
    slopes = np.divide(slopes, np.mean(np.add(tmp1,tmp2)))
    # Subtract reference slopes
    return slopes - refSlopes


"""
Optimized for best performance.
Works very well with numba JIT compilation.
Performed better compared to a numpy only implementation
"""
@jit(nopython=True, nogil=True, cache=False, fastmath=True)
def computeSlopesPYWFSOptimNumba(image:np.ndarray,
                            p1Mask:np.ndarray, 
                            p2Mask:np.ndarray,
                            p3Mask:np.ndarray, 
                            p4Mask:np.ndarray,
                            p1:np.ndarray, 
                            p2:np.ndarray,
                            p3:np.ndarray, 
                            p4:np.ndarray,
                            tmp1:np.ndarray,
                            tmp2:np.ndarray,
                            numPixelsInPupils:int, 
                            slopes:np.ndarray,
                            refSlopes:np.ndarray,
                        ):
    """Compute pyramid-WFS slopes using a Numba-optimized CPU kernel."""

    # Mask Pupils out of image and convert to floats
    p1_count, p2_count, p3_count, p4_count = 0, 0, 0, 0
    for i in range(len(image)):
        if p1Mask[i]:
            p1[p1_count] = np.float32(image[i])
            p1_count += 1
        if p2Mask[i]:
            p2[p2_count] = np.float32(image[i])
            p2_count += 1
        if p3Mask[i]:
            p3[p3_count] = np.float32(image[i])
            p3_count += 1
        if p4Mask[i]:
            p4[p4_count] = np.float32(image[i])
            p4_count += 1

    # Sum Pupils, Saving partial sums to avoid recomputing later
    total_sum = 0.0
    for i in range(numPixelsInPupils):  # Assuming all counts are equal
        tmp1[i] = p1[i] + p2[i]
        tmp2[i] = p3[i] + p4[i]
        total_sum += tmp1[i] + tmp2[i]
    mean_value = total_sum / p1_count

    for i in range(numPixelsInPupils):
        # Compute Y slopes
        slopes[i] = (tmp1[i] - tmp2[i])/mean_value - refSlopes[i]
        # Compute X slopes
        slopes[numPixelsInPupils + i] = ((p1[i] + p3[i]) - (p2[i] + p4[i]))/mean_value \
            - refSlopes[numPixelsInPupils + i]

    return slopes

"""
Optimized for best performance.
Works very well with numba JIT compilation.
Performed better compared to a numpy only implementation, while also
allowing for non-integer spacing.
"""
@jit(nopython=True, nogil=True, cache=False)
def computeSlopesSHWFSOptimNumba(image:np.ndarray, 
                                 slopes:np.ndarray, 
                                 unaberratedSlopes:np.ndarray, 
                                 threshold:np.float32, 
                                 spacing:np.float32,
                                 xvals:np.ndarray,
                                 offsetX:int, 
                                 offsetY:int,
                                 intN:int,
                                 ):
    """Compute Shack-Hartmann centroid slopes with a Numba kernel.

    The image is traversed lenslet by lenslet, thresholded locally, and reduced
    into x/y centroid offsets relative to the unaberrated reference slopes.
    """

    
    # Convert image to the same dtype as unaberratedSlopes
    image = image.astype(np.float32)
    
    # Compute the number of sub-apertures
    numRegions = unaberratedSlopes.shape[1]

    # Loop over all regions
    for i in range(numRegions):
        for j in range(numRegions):
            # Compute where to start
            start_i = int(round(spacing * i)) + offsetY
            start_j = int(round(spacing * j)) + offsetX
            
            # Ensure we stay within the bounds of the image
            if start_j + intN <= image.shape[1] and start_i + intN <= image.shape[0]:
                #Create a local subimage around the lenslet spot
                sub_im = image[start_i:start_i + intN, start_j:start_j + intN]

                #loop through the sub image
                norm = np.float32(0)
                weightX = np.float32(0)
                weightY = np.float32(0)
                for m in range(intN):
                    for n in range(intN):
                        #If we are counting the pixel
                        if sub_im[m,n] > threshold:
                            #Add it to the normalization
                            norm += sub_im[m,n]
                            #Compute the X and Y centroids (before normalization)
                            weightX += xvals[m,n] * sub_im[m,n]
                            weightY += xvals[n,m] * sub_im[m,n]

                #If we have flux in the sub aperture
                if norm > 0:
                    #Normalize the centroids and remove the reference slope
                    slopes[i, j] = weightX/norm - unaberratedSlopes[i, j]
                    slopes[i + numRegions, j] = weightY/norm - unaberratedSlopes[i + numRegions, j]
                #If we have no flux slopes should be zero
    
    return slopes

"""
Optimized for best performance with numpy only.
Does not allow for non-integer spacing.
"""
def computeSlopesSHWFSOptimNumpy(image:np.ndarray, 
                                 slopes:np.ndarray, 
                                 unaberratedSlopes:np.ndarray, 
                                 threshold:float, 
                                 spacing:int, 
                                 xvals:np.array):

    #Only works for integer spacings
    spacing = int(spacing)

    # Convert the image to floats and threshold in one operation
    image = np.where(image > threshold, image.astype(np.float32), 0.0)

    # Reshape the image into blocks of size spacing X spacing
    reshaped_image = image.reshape(image.shape[0] // spacing, spacing, image.shape[1] // spacing, spacing)

    # Compute the sum of pixel values in each MxM region
    region_sums = np.sum(reshaped_image, axis=(1, 3))

    # Precompute the dot products instead of tensordot (which is more general but slower)
    weighted_sum_x = np.einsum('ijkl,jl->ik', reshaped_image, xvals)
    weighted_sum_y = np.einsum('ijkl,jl->ik', reshaped_image, xvals.T)
    
    # Get mask for non-zero value sums
    mask = region_sums > 0.0

    # Compute the centroids directly on the valid regions
    valid_region_sums = region_sums[mask]
    slopes[:slopes.shape[1]][mask] = weighted_sum_x[mask] / valid_region_sums - unaberratedSlopes[:slopes.shape[1]][mask]
    slopes[slopes.shape[1]:][mask] = weighted_sum_y[mask] / valid_region_sums - unaberratedSlopes[slopes.shape[1]:][mask]

    # Return the difference with reference slopes
    return slopes

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
    gpuDevice : str
        Default device if using GPU
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
        try:
            super().__init__(conf)
            self.conf = conf
            self.name = "Slopes"

            self.wfsShm, self.imageShape, self.imageDType = initExistingShm(self.input_stream_name("wfs"), gpuDevice=self.gpuDevice)
            self.register_input_stream("wfs", self.wfsShm)

            self.signalDType = np.float32
            self.imageNoise = setFromConfig(self.conf, "imageNoise", 0.0)
            self.centralObscurationRatio = setFromConfig(self.conf, "centralObscurationRatio", 0.0)

            self.wfsType = self.conf["type"].lower()
            self.signalType = self.conf["signalType"]
            self.validSubAps = None
            self.validSubApsFile = setFromConfig(self.conf, "validSubApsFile", "")

            self.refSlopesFile = setFromConfig(self.conf, "refSlopesFile", "")
            self.refSlopeCount = setFromConfig(self.conf, "refSlopeCount", 1000)

            if self.wfsType == "pywfs":
                if "pupils" in self.conf.keys():
                    pupilLocs = [(int(x.split(',')[1]), int(x.split(',')[0])) for x in self.conf["pupils"]]
                    self.setPupils(pupilLocs, self.conf["pupilsRadius"])
                else:
                    a, b = int(0.25 * self.imageShape[0]), int(0.75 * self.imageShape[0])
                    c, d = int(0.25 * self.imageShape[1]), int(0.75 * self.imageShape[1])
                    r = min(self.imageShape[0] - b, self.imageShape[1] - d)
                    self.setPupils([(a, c), (a, d), (b, c), (b, d)], r)
                if self.signalType == 'slopes':
                    self.flatNorm = setFromConfig(self.conf, "flatNorm", True)

                self.refSlopes = np.zeros(self.signal2DShape, dtype=self.signalDType)
                self.refSlopes1D = np.zeros_like(self.signal.read_noblock())
                self.slopesArr1D = np.zeros_like(self.refSlopes1D)
                self.numPixelsInPupils = np.count_nonzero(self.p1mask)
                self.p1 = np.empty(self.numPixelsInPupils, dtype=self.signalDType)
                self.p2 = np.empty_like(self.p1)
                self.p3 = np.empty_like(self.p1)
                self.p4 = np.empty_like(self.p1)
                self.tmp1, self.tmp2 = np.empty_like(self.p1), np.empty_like(self.p1)

            elif self.wfsType == "shwfs":
                self.shwfsContrast = setFromConfig(self.conf, "contrast", 0)
                self.subApSpacing = self.conf["subApSpacing"]
                self.regionSize = int(np.round(self.subApSpacing, 0))
                self.numRegions = self.imageShape[0] // self.regionSize
                self.offsetX = self.conf["subApOffsetX"]
                self.offsetY = self.conf["subApOffsetY"]
                xvals = np.arange(self.regionSize).astype(int) - self.regionSize // 2
                self.xvals = np.meshgrid(xvals, xvals)[0].astype(self.signalDType)

                self.signal2DSize = int(2 * self.numRegions**2)
                self.signal2DShape = (2 * self.numRegions, self.numRegions)

                self.validSubAps = np.ones(self.signal2DShape, dtype=bool)
                self.loadValidSubAps()

                self.signalSize = np.sum(self.validSubAps)
                self.signalShape = (self.signalSize,)

                logger.info(
                    "SHWFS slopes configured subApSpacing=%s numRegions=%s offsetX=%s offsetY=%s signalSize=%s signalShape=%s signalDType=%s",
                    self.subApSpacing,
                    self.numRegions,
                    self.offsetX,
                    self.offsetY,
                    self.signalSize,
                    self.signalShape,
                    self.signalDType,
                )

                self._configure_signal_streams(self.signalShape, self.signal2DShape)

                self.refSlopes = np.zeros(self.signal2DShape, dtype=self.signalDType)

            self.loadRefSlopes()
            self.logger.info("Initialized slopes process wfsType=%s signalType=%s imageShape=%s", self.wfsType, self.signalType, self.imageShape)
        except Exception:
            logger.exception("Failed to initialize slopes process")
            raise

    def _close_signal_streams(self) -> None:
        """Close any currently attached signal output streams."""

        for attribute_name in ("signal", "signal2D"):
            stream = getattr(self, attribute_name, None)
            if stream is None or not hasattr(stream, "close"):
                continue
            try:
                stream.close()
            except Exception:
                logger.debug("Failed closing %s stream during rebuild", attribute_name, exc_info=True)

    def _existing_output_stream_matches(self, stream_name: str, expected_shape: tuple[int, ...]) -> bool:
        """Return ``True`` when an existing SHM matches the expected output shape."""

        try:
            stream, shape, dtype = initExistingShm(self.output_stream_name(stream_name), gpuDevice=self.gpuDevice)
        except Exception:
            return False

        try:
            return tuple(shape) == tuple(expected_shape) and np.dtype(dtype) == np.dtype(self.signalDType)
        finally:
            try:
                stream.close()
            except Exception:
                logger.debug("Failed closing temporary stream probe for %s", stream_name, exc_info=True)

    def _configure_signal_streams(
        self,
        signal_shape: tuple[int, ...],
        signal2d_shape: tuple[int, ...],
        *,
        rebuild: bool = False,
    ) -> None:
        """Create or rebuild the signal and signal2D output streams.

        Parameters
        ----------
        signal_shape : tuple of int
            Desired one-dimensional signal shape.
        signal2d_shape : tuple of int
            Desired two-dimensional display shape.
        rebuild : bool, optional
            When ``True`` force a rebuild of the underlying SHMs. This is used
            when pupil geometry changes can resize the signal outputs.
        """

        self.signalShape = tuple(int(axis) for axis in signal_shape)
        self.signal2DShape = tuple(int(axis) for axis in signal2d_shape)

        needs_rebuild = rebuild
        if not needs_rebuild:
            needs_rebuild = not self._existing_output_stream_matches("signal", self.signalShape)
        if not needs_rebuild:
            needs_rebuild = not self._existing_output_stream_matches("signal2D", self.signal2DShape)

        if needs_rebuild:
            self._close_signal_streams()
            clear_shms([self.output_stream_name("signal"), self.output_stream_name("signal2D")])

        self.signal = ImageSHM(self.output_stream_name("signal"), self.signalShape, self.signalDType, gpuDevice=self.gpuDevice, consumer=False)
        self.signal2D = ImageSHM(self.output_stream_name("signal2D"), self.signal2DShape, self.signalDType, gpuDevice=self.gpuDevice, consumer=False)
        self.register_output_stream("signal", self.signal, source_streams=["wfs"], lineage_source="wfs")
        self.register_output_stream("signal2D", self.signal2D, source_streams=["signal"], lineage_source="signal")

    
    def read(self, block = True, SAFE=True, GPU=False):
        """
        Read the current signal.

        Returns
        -------
        numpy.ndarray
            Current signal.
        """
        if block:
            return self.read_stream("signal", SAFE=SAFE, GPU=GPU, RELEASE_GIL=self.RELEASE_GIL)
        return self.read_stream("signal", block=False, SAFE=SAFE, GPU=GPU)
    
    def readImage(self, SAFE=True, GPU=False, block=True):
        """
        Read the current WFS image.

        Returns
        -------
        numpy.ndarray
            Current WFS image.
        """
        if block:
            return self.read_stream("wfs", SAFE=SAFE, GPU=GPU, RELEASE_GIL=self.RELEASE_GIL)
        return self.read_stream("wfs", block=False, SAFE=SAFE, GPU=GPU)

    def setValidSubAps(self, validSubAps):
        """
        Set the valid sub-aperture mask. Converts to boolean if not already

        Parameters
        ----------
        validSubAps : numpy.ndarray
            Valid sub-aperture mask.
        """
        component_logger = getattr(self, "logger", logger)
        try:
            self.validSubAps = validSubAps.astype(bool)
            self.curSignal2D = np.zeros(validSubAps.shape)
            component_logger.info("Set valid sub-aperture mask shape=%s", validSubAps.shape)
        except Exception:
            component_logger.exception("Failed to set valid sub-aperture mask")
            raise
        return
    
    def saveValidSubAps(self,filename=''):
        """
        Save the valid sub-aperture mask to a file.

        Parameters
        ----------
        filename : str, optional
            File to save the valid sub-aperture mask to. If not specified, uses the configured validSubApsFile.
        """
        component_logger = getattr(self, "logger", logger)
        try:
            if filename == '':
                filename = self.validSubApsFile
            if filename == '':
                raise ValueError("No validSubAps filename provided")
            np.save(filename, self.validSubAps)
            component_logger.info("Saved valid sub-aperture mask to %s", filename)
        except Exception:
            component_logger.exception("Failed to save valid sub-aperture mask to %s", filename or getattr(self, "validSubApsFile", ""))
            raise
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
        component_logger = getattr(self, "logger", logger)
        try:
            if filename == '':
                filename = self.validSubApsFile
            if filename == '':
                validSubAps = np.ones_like(self.validSubAps)
                component_logger.info("No validSubAps file configured; using all-true mask")
            else:
                validSubAps = np.load(filename)
                component_logger.info("Loaded valid sub-aperture mask from %s", filename)

            self.setValidSubAps(validSubAps)
        except Exception:
            component_logger.exception("Failed to load valid sub-aperture mask from %s", filename or getattr(self, "validSubApsFile", ""))
            raise

        return


    def takeRefSlopes(self):
        """
        Take reference slopes by averaging multiple slope measurements. Number of measurements
        set by refSlopeCount variable.
        """
        component_logger = getattr(self, "logger", logger)
        try:
            if self.refSlopeCount < 1:
                raise ValueError("refSlopeCount must be at least 1")
            component_logger.info("Taking reference slopes using %s frames", self.refSlopeCount)
            self.setRefSlopes(np.zeros_like(self.refSlopes))
            refSlopes = np.zeros_like(self.refSlopes)
            for _ in range(self.refSlopeCount):
                cur_slopes = self.read().astype(refSlopes.dtype)
                refSlopes += self.computeSignal2D(cur_slopes)
            refSlopes /= self.refSlopeCount
            self.setRefSlopes(refSlopes)
            component_logger.info("Completed reference slope acquisition")
        except Exception:
            component_logger.exception("Failed to take reference slopes")
            raise
        return 

    def setRefSlopes(self, refSlopes):
        """
        Set the reference slopes.

        Parameters
        ----------
        refSlopes : numpy.ndarray
            Reference slopes.
        """
        component_logger = getattr(self, "logger", logger)
        try:
            self.refSlopes = refSlopes.astype(self.signalDType)
            if self.wfsType == 'pywfs':
                slopemask = self.validSubAps[:, :self.validSubAps.shape[1] // 2]
                self.refSlopes1D = np.zeros_like(self.signal.read_noblock())
                self.refSlopes1D[:self.refSlopes1D.size // 2] = self.refSlopes[:, :self.refSlopes.shape[1] // 2][slopemask]
                self.refSlopes1D[self.refSlopes1D.size // 2:] = self.refSlopes[:, self.refSlopes.shape[1] // 2:][slopemask]
            component_logger.info("Updated reference slopes")
        except Exception:
            component_logger.exception("Failed to update reference slopes")
            raise
            
        return
    
    def saveRefSlopes(self,filename=''):
        """
        Save the reference slopes to a file.

        Parameters
        ----------
        filename : str, optional
            File to save the reference slopes to. If not specified, uses the configured refSlopesFile.
        """
        component_logger = getattr(self, "logger", logger)
        try:
            if filename == '':
                filename = self.refSlopesFile
            if filename == '':
                raise ValueError("No reference slopes filename provided")
            np.save(filename, self.refSlopes)
            component_logger.info("Saved reference slopes to %s", filename)
        except Exception:
            component_logger.exception("Failed to save reference slopes to %s", filename or getattr(self, "refSlopesFile", ""))
            raise
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
        component_logger = getattr(self, "logger", logger)
        try:
            if filename == '':
                filename = self.refSlopesFile
            if filename == '':
                refSlopes = np.zeros_like(self.refSlopes)
                component_logger.info("No reference slopes file configured; using zeros")
            else:
                refSlopes = np.load(filename)
                component_logger.info("Loaded reference slopes from %s", filename)

            self.setRefSlopes(refSlopes)
        except Exception:
            component_logger.exception("Failed to load reference slopes from %s", filename or getattr(self, "refSlopesFile", ""))
            raise
        return
    
    def computeSignal(self):
        """
        Compute the signal from the WFS image.
        """
        image = self.readImage(SAFE=False, GPU = False)
        if self.signalType == "slopes":
            if self.wfsType == "pywfs":
                if self.gpuDevice is not None and gpu_torch_available():
                    import torch

                    slope_signal = computeSlopesPYWFSTorch(image.ravel(),
                                p1Mask=torch.from_numpy(self.p1mask.ravel()).to(self.gpuDevice), 
                                p2Mask=torch.from_numpy(self.p2mask.ravel()).to(self.gpuDevice),
                                p3Mask=torch.from_numpy(self.p3mask.ravel()).to(self.gpuDevice), 
                                p4Mask=torch.from_numpy(self.p4mask.ravel()).to(self.gpuDevice),
                                numPixelsInPupils = self.numPixelsInPupils, 
                                slopes = torch.from_numpy(self.slopesArr1D).to(self.gpuDevice),
                                refSlopes=torch.from_numpy(self.refSlopes1D).to(self.gpuDevice)).cpu().numpy()
                else:
                    slope_signal = computeSlopesPYWFSOptimNumba(image=image.ravel(),
                                p1Mask=self.p1mask.ravel(), 
                                p2Mask=self.p2mask.ravel(),
                                p3Mask=self.p3mask.ravel(), 
                                p4Mask=self.p4mask.ravel(),
                                p1=self.p1, 
                                p2=self.p2,
                                p3=self.p3, 
                                p4=self.p4,
                                tmp1=self.tmp1,
                                tmp2=self.tmp2,
                                numPixelsInPupils = self.numPixelsInPupils, 
                                slopes = self.slopesArr1D,
                                refSlopes=self.refSlopes1D)
                
                
            elif self.wfsType == "shwfs":

                slopes = computeSlopesSHWFSOptimNumba(image = image,
                                            slopes = np.zeros_like(self.refSlopes), 
                                            unaberratedSlopes = self.refSlopes,
                                            threshold = self.imageNoise*self.shwfsContrast, 
                                            spacing = self.subApSpacing,
                                            xvals = self.xvals,
                                            offsetX = self.offsetX,
                                            offsetY = self.offsetY,
                                            intN = self.regionSize)
                slope_signal = slopes[self.validSubAps]
                # self.signal.write(slopes[self.validSubAps])
                # self.signal2D.write(slopes*self.validSubAps)
                # slopes = np.zeros_like(self.refSlopes)
                # self.signal.write(self.refSlopes.flatten()[:np.prod(self.signalShape)].reshape(self.signalShape))
            self.write_stream("signal", slope_signal, source_streams=["wfs"], lineage_source="wfs")
            self.write_stream("signal2D", self.computeSignal2D(slope_signal), source_streams=["signal"], lineage_source="signal")
        
        return
    
    def computeImageNoise(self):
        """
        Compute the image noise. Useful to set a good SNR cutoff for SHWFS
        """
        component_logger = getattr(self, "logger", logger)
        try:
            img = self.readImage()
            if img[img < 0].size > 0:
                self.imageNoise = compute_fwhm_dark_subtracted_image(img) / 2
                component_logger.info("Computed image noise=%s", self.imageNoise)
            else:
                logger.warning("Image is not dark subtracted")
        except Exception:
            component_logger.exception("Failed to compute image noise")
            raise
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
        component_logger = getattr(self, "logger", logger)
        try:
            self.pupilLocs = pupilLocs
            self.pupilRadius = pupilRadius
            self.computePupilsMask()
            if self.signalType == "slopes":
                self.signalSize = np.count_nonzero(self.pupilMask) // 2
                slopemask = self.pupilMask[
                    self.pupilLocs[0][1]-self.pupilRadius:self.pupilLocs[0][1]+self.pupilRadius,
                    self.pupilLocs[0][0]-self.pupilRadius:self.pupilLocs[0][0]+self.pupilRadius,
                ] > 0
                self.setValidSubAps(np.concatenate([slopemask, slopemask], axis=1))
                if self.validSubApsFile != "":
                    self.saveValidSubAps()
                self._configure_signal_streams(
                    (self.signalSize,),
                    (self.validSubAps.shape[0], self.validSubAps.shape[1]),
                    rebuild=True,
                )
            component_logger.info("Configured pupils locs=%s radius=%s", pupilLocs, pupilRadius)
        except Exception:
            component_logger.exception("Failed to set pupils locs=%s radius=%s", pupilLocs, pupilRadius)
            raise
            
        return

    def computePupilsMask(self):
        """
        Compute the mask for the pupils. Assumes circular aperture with obstruction ratio 
        set by the centralObscurationRatio parameter.
        """
        component_logger = getattr(self, "logger", logger)
        try:
            self.pupilMask = np.zeros(self.imageShape)

            pupilTemplate = generate_circular_aperture_mask(int(np.ceil(2*self.pupilRadius)),
                                                            self.pupilRadius, 
                                                            self.centralObscurationRatio)
            N = self.pupilMask.shape[0]
            n = pupilTemplate.shape[0]
            half_n = n // 2

            for i, pupil_loc in enumerate(self.pupilLocs):
                px, py = pupil_loc

                x_start = px - half_n
                x_end = px + half_n + (n % 2)
                y_start = py - half_n
                y_end = py + half_n + (n % 2)
            
                if x_start < 0 or y_start < 0 or x_end > N or y_end > N:
                    raise ValueError("The subimage exceeds the bounds of the larger array.")

                self.pupilMask[y_start:y_end, x_start:x_end] += pupilTemplate*(i+1)

            self.p1mask = self.pupilMask == 1
            self.p2mask = self.pupilMask == 2
            self.p3mask = self.pupilMask == 3
            self.p4mask = self.pupilMask == 4
            component_logger.info("Computed pupil masks for %s pupils", len(self.pupilLocs))
        except Exception:
            component_logger.exception("Failed to compute pupil mask")
            raise
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
        
        if self.wfsType == "pywfs":
            slopemask = validSubAps[:,:validSubAps.shape[1]//2]
            self.curSignal2D[:,:validSubAps.shape[1]//2][slopemask] = signal[:signal.size//2]
            self.curSignal2D[:,validSubAps.shape[1]//2:][slopemask] = signal[signal.size//2:]
        else:
            self.curSignal2D[self.validSubAps] = signal
        return self.curSignal2D
    
if __name__ == "__main__":

    launchComponent(SlopesProcess, "slopes", start = True)