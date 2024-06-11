"""
Wavefront Sensor Superclass
"""
from pyRTC.Pipeline import ImageSHM, work
from pyRTC.utils import *
from pyRTC.pyRTCComponent import *
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from sys import platform

class WavefrontSensor(pyRTCComponent):
    """
    A pyRTCComponent which represents a Wavefront Sensor (camera). This is a general class which is 
    reponsible for all components of wavefront sensing which are common to all wavefront sensors. This
    class should be used by defining a child class held in pyRTC.hardware, which overwrites
    the relevant functions which actual hardware connectivity code. The child class can call its parent
    implementations in order to make use of the code which sets the relevant parameters, write to shared
    memory, etc... or they can overwrite them completely. See hardware/ximeaWFS.py for an example.

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

        self.name = setFromConfig(conf, "name", "wavefrontSensor")
        self.width = conf["width"]
        self.height = conf["height"]
        self.darkCount = setFromConfig(conf, "darkCount", 1000)
        self.darkFile = setFromConfig(conf, "darkFile", "")

        self.imageShape = (self.width, self.height)
        self.imageRawDType = np.uint16
        self.imageDType = np.int32

        self.imageRaw = ImageSHM("wfsRaw", self.imageShape, self.imageRawDType)
        self.image = ImageSHM("wfs", self.imageShape, self.imageDType)

        self.data = np.zeros(self.imageShape, dtype=self.imageRawDType)
        self.dark = np.zeros(self.imageShape, dtype=self.imageDType)

        self.loadDark()

        super().__init__(conf)

        return
    
    def setRoi(self, roi):
        """
        Sets the region of interest (ROI) for the sensor.

        Parameters
        ----------
        roi : tuple
            A tuple containing (width, height, left, top) of the ROI.
        """
        self.roiWidth = roi[0]
        self.roiHeight = roi[1]
        self.roiLeft = roi[2]
        self.roiTop = roi[3]

        return

    def setExposure(self, exposure: float) -> None:
        """
        Sets the exposure time for the sensor.

        Parameters
        ----------
        exposure : float
            Exposure time in whatever unit your camera uses.
        """
        self.exposure = exposure

        return
    
    def setBinning(self, binning: int) -> None:
        """
        Sets the binning factor for the sensor.

        Parameters
        ----------
        binning : int
            Binning factor.
        """
        self.binning = binning

        return
    
    def setGain(self, gain: float) -> None:
        """
        Sets the gain for the sensor.

        Parameters
        ----------
        gain : float
            Gain value.
        """
        self.gain = gain
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
        self.bitDepth = bitDepth
        return
    
    def expose(self) -> None:
        """
        Writes the current image data to shared memory. Both raw, and dark subtracted.
        """
        self.imageRaw.write(self.data)
        self.image.write(self.data.astype(self.imageDType) - self.dark)
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
            return self.image.read()
        else:
            return self.image.read_noblock()
    
    def takeDark(self) -> None:
        """
        Captures and sets the dark frame.
        """
        self.setDark(np.zeros_like(self.dark))
        dark = np.zeros(self.imageShape, dtype=np.float64)
        for i in range(self.darkCount):
            dark += self.read().astype(np.float64)
        dark /= self.darkCount
        self.setDark(dark)        
        return 

    def setDark(self, dark) -> None:
        """
        Sets the dark frame.

        Parameters
        ----------
        dark : ndarray
            Dark frame data.
        """
        self.dark = dark.astype(self.imageDType)
        return
    
    def saveDark(self,filename=''):
        """
        Saves the dark frame to a file.

        Parameters
        ----------
        filename : str, optional
            Filename to save the dark frame to. If not specified, uses the dark file path from the configuration.
        """
        if filename == '':
            filename = self.darkFile
        np.save(filename, self.dark)
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
        if filename == '':
            filename = self.darkFile
        #If we are still without a file, set zeros
        if filename == '':
            self.dark = np.zeros_like(self.dark)
        else: #If we have a filename
            self.dark = np.load(filename)
        return
    
    def plot(self) -> None:
        """
        Plots the current image data.
        """
        arr = self.read(block=False)
        plt.figure(figsize=(8,8))
        plt.imshow(arr, cmap = 'inferno', origin='lower')
        plt.colorbar()
        plt.show()
        return
    
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
    set_affinity((conf["wfs"]["affinity"])%os.cpu_count()) 
    decrease_nice(pid)

    confWFS = conf["wfs"]
    wfs = WavefrontSensor(conf=confWFS)

    wfs.start()
    
    l = Listener(wfs, port= int(args.port))
    while l.running:
        l.listen()
        time.sleep(1e-3)