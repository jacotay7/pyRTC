"""Science-camera abstractions and common image-quality telemetry.

This module defines the base class used by pyrtc science-camera adapters. It
handles the shared-memory products that downstream tools expect, including short
and long exposure PSFs, Strehl ratio estimates, and tip-tilt telemetry, while
leaving camera-specific acquisition details to hardware subclasses.
"""

import matplotlib.pyplot as plt
import numpy as np

from pyrtc.logging_utils import ensure_logging_configured, get_logger
from pyrtc.streams import create_stream
from pyrtc.manager import launch_component
from pyrtc.component import Component
from pyrtc.utils import centroid, clean_image_for_strehl, set_from_config


logger = get_logger(__name__)


class ScienceCamera(Component):
    """
    Base class for cameras that produce science images and image-quality metrics.

    ``ScienceCamera`` centralizes the parts of imaging that are shared across
    real and synthetic science-camera backends: SHM publication, dark/model PSF
    handling, long-exposure accumulation, and simple Strehl/tip-tilt telemetry.
    Subclasses are expected to implement the device-facing acquisition logic and
    then call the parent methods so the standard pyrtc products stay updated.

    Config
    ------
    name : str
        Name of the camera.
    width : int
        Width of the image. Required.
    height : int
        Height of the image. Required.
    dark_count : int
        Number of dark frames to average. Required.
    integration : int
        Integration length. Required.
    dark_file : str, optional
        File to save the dark frames. Default is "".
    model_file : str, optional
        File to save the model PSF. Default is "".

    Attributes
    ----------
    name : str
        Name of the camera.
    image_shape : tuple
        Shape of the image.
    image_raw_dtype : type
        Data type of the raw image.
    image_dtype : type
        Data type of the image.
    psf_long_dtype : type
        Data type of the long exposure PSF.
    psf_short : pyshmem.SharedMemory
        Shared memory object for the short exposure PSF.
    psf_long : pyshmem.SharedMemory
        Shared memory object for the long exposure PSF.
    strehl_shm : pyshmem.SharedMemory
        Shared memory object for the Strehl ratio.
    tip_tilt_shm : pyshmem.SharedMemory
        Shared memory object for the tip-tilt.
    data : numpy.ndarray
        Data array for the image.
    dark : numpy.ndarray
        Dark frame.
    dark_count : int
        Number of dark frames to average.
    dark_file : str
        File to save the dark frames.
    model : numpy.ndarray
        Model PSF.
    model_file : str
        File to save the model PSF.
    strehl_ratio : float
        Strehl ratio.
    peak_dist : float
        Peak distance.
    integration_length : int
        Integration length.
    roi_width : int
        Width of the region of interest.
    roi_height : int
        Height of the region of interest.
    roi_left : int
        Left coordinate of the region of interest.
    roi_top : int
        Top coordinate of the region of interest.
    exposure : int
        Exposure time.
    binning : int
        Binning factor.
    gain : int
        Gain setting.
    bit_depth : int
        Bit depth setting.
    """

    def __init__(self, conf) -> None:
        try:
            output_streams = (
                conf.get("output_streams", {})
                if isinstance(conf.get("output_streams"), dict)
                else {}
            )

            def _output_name(stream_name: str) -> str:
                value = output_streams.get(stream_name, stream_name)
                if isinstance(value, dict):
                    value = value.get("shm", value.get("name", stream_name))
                return str(value)

            ensure_logging_configured(app_name="pyrtc", component_name=self.__class__.__name__)
            self.logger = get_logger(f"{self.__class__.__module__}.{self.__class__.__name__}")
            self.name = conf["name"]
            self.image_shape = (conf["width"], conf["height"])
            self.image_raw_dtype = np.uint16
            self.image_dtype = np.int32
            self.psf_long_dtype = np.float64

            self.psf_short = create_stream(
                _output_name("psf_short"), self.image_shape, self.image_dtype
            )
            self.psf_long = create_stream(
                _output_name("psf_long"), self.image_shape, self.psf_long_dtype
            )
            self.strehl_shm = create_stream(_output_name("strehl"), (1,), float)
            self.tip_tilt_shm = create_stream(_output_name("tiptilt"), (1,), float)

            self.data = np.zeros(self.image_shape, dtype=self.image_raw_dtype)
            self.dark = np.zeros(self.image_shape, dtype=self.image_dtype)
            self.dark_count = conf["dark_count"]
            self.dark_file = set_from_config(conf, "dark_file", "")
            self.model = np.zeros(self.image_shape, dtype=self.psf_long_dtype)
            self.model_file = set_from_config(conf, "model_file", "")
            self.strehl_ratio = 0
            self.peak_dist = 0

            self.load_dark()
            self.load_model_psf()

            self.integration_length = conf["integration"]
            super().__init__(conf)
            self.register_output_stream("psf_short", self.psf_short)
            self.register_output_stream("psf_long", self.psf_long)
            self.register_output_stream("strehl", self.strehl_shm)
            self.register_output_stream("tiptilt", self.tip_tilt_shm)
            self.logger.info(
                "Initialized science camera name=%s image_shape=%s integration=%s",
                self.name,
                self.image_shape,
                self.integration_length,
            )
        except Exception:
            logger.exception("Failed to initialize science camera")
            raise

    def set_roi(self, roi):
        """
        Set the region of interest (ROI).

        Parameters
        ----------
        roi : tuple
            Tuple containing (width, height, left, top) of the ROI.
        """
        try:
            self.roi_width = roi[0]
            self.roi_height = roi[1]
            self.roi_left = roi[2]
            self.roi_top = roi[3]
            self.logger.info("Set ROI width=%s height=%s left=%s top=%s", *roi)
        except Exception:
            logger.exception("Failed to set ROI from %s", roi)
            raise
        return

    def set_exposure(self, exposure):
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

    def set_binning(self, binning):
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

    def set_gain(self, gain):
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

    def set_gamma(self, gamma):
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

    def set_bit_depth(self, bit_depth):
        """
        Set the bit depth.

        Parameters
        ----------
        bit_depth : int
            Bit depth to set.
        """
        try:
            self.bit_depth = bit_depth
            self.logger.info("Set bit depth to %s", bit_depth)
        except Exception:
            logger.exception("Failed to set bit depth to %s", bit_depth)
            raise
        return

    def set_integration_length(self, integration_length):
        """
        Set the integration length.

        Parameters
        ----------
        integration_length : int
            Integration length to set.
        """
        try:
            self.integration_length = integration_length
            self.logger.info("Set integration length to %s", integration_length)
        except Exception:
            logger.exception("Failed to set integration length to %s", integration_length)
            raise
        return

    def expose(self):
        """
        Perform a single exposure.
        """
        self.write_stream("psf_short", self.data.astype(self.image_dtype) - self.dark)
        return

    def integrate(self):
        """
        Perform multiple exposures and integrate the results. Number of frames set by integration_length.
        """
        x = np.zeros(self.data.shape)
        for i in range(self.integration_length):
            x += self.read().astype(x.dtype)
        self.write_stream("psf_long", x / self.integration_length)
        return

    def read(self, block=True):
        """
        Read the current short exposure PSF.

        Returns
        -------
        numpy.ndarray
            Current short exposure PSF.
        """
        if block:
            return self.read_stream("psf_short")
        return self.read_stream("psf_short", block=False)

    def read_long(self):
        """
        Read the current long exposure PSF.

        Returns
        -------
        numpy.ndarray
            Current long exposure PSF.
        """
        return self.read_stream("psf_long")

    def take_dark(self):
        """
        Take dark frames and average them to create a dark frame.
        Number of exposures to average set by dark_count parameter.
        """
        try:
            if self.dark_count < 1:
                raise ValueError("dark_count must be at least 1 to acquire a dark frame")
            self.logger.info("Taking science camera dark frame using %s exposures", self.dark_count)
            self.set_dark(np.zeros_like(self.dark))
            dark = np.zeros(self.image_shape, dtype=np.float64)
            for _ in range(self.dark_count):
                dark += self.read().astype(np.float64)
            dark /= self.dark_count
            self.set_dark(dark)
            self.logger.info("Completed science camera dark frame acquisition")
        except Exception:
            logger.exception("Failed to acquire science camera dark frame")
            raise
        return

    def set_dark(self, dark):
        """
        Set the dark frame.

        Parameters
        ----------
        dark : numpy.ndarray
            Dark frame to set.
        """
        try:
            self.dark = dark.astype(self.image_dtype)
            self.logger.info("Updated science camera dark frame")
        except Exception:
            logger.exception("Failed to update science camera dark frame")
            raise
        return

    def save_dark(self, filename=""):
        """
        Save the dark frame to a file.

        Parameters
        ----------
        filename : str, optional
            File to save the dark frame to. If not specified, uses the configured dark_file.
        """
        try:
            if filename == "":
                filename = self.dark_file
            if filename == "":
                raise ValueError("No dark frame filename provided")
            np.save(filename, self.dark)
            self.logger.info("Saved science camera dark frame to %s", filename)
        except Exception:
            logger.exception(
                "Failed to save science camera dark frame to %s", filename or self.dark_file
            )
            raise
        return

    def load_dark(self, filename=""):
        """
        Load the dark frame from a file.

        Parameters
        ----------
        filename : str, optional
            File to load the dark frame from. If not specified, uses the configured dark_file.
        """
        # If no file given, first try dark file
        try:
            if filename == "":
                filename = self.dark_file
            if filename == "":
                self.dark = np.zeros_like(self.dark)
                logger.info("No science camera dark frame file configured; using zeros")
            else:
                self.dark = np.load(filename)
                self.logger.info("Loaded science camera dark frame from %s", filename)
        except Exception:
            logger.exception(
                "Failed to load science camera dark frame from %s", filename or self.dark_file
            )
            raise
        return

    def take_model_psf(self):
        """
        Capture the current long exposure PSF as the model PSF.
        """
        try:
            self.model = self.read_long()
            self.logger.info("Captured model PSF from current long-exposure image")
        except Exception:
            logger.exception("Failed to capture model PSF")
            raise
        return

    def set_model_psf(self, model):
        """
        Set the model PSF.

        Parameters
        ----------
        model : numpy.ndarray
            Model PSF to set.
        """
        try:
            self.model = model.astype(self.psf_long_dtype)
            self.logger.info("Updated model PSF")
        except Exception:
            logger.exception("Failed to update model PSF")
            raise
        return

    def save_model_psf(self, filename=""):
        """
        Save the model PSF to a file.

        Parameters
        ----------
        filename : str, optional
            File to save the model PSF to. If not specified, uses the configured model_file.
        """
        try:
            if filename == "":
                filename = self.model_file
            if filename == "":
                raise ValueError("No model PSF filename provided")
            np.save(filename, self.model)
            self.logger.info("Saved model PSF to %s", filename)
        except Exception:
            logger.exception("Failed to save model PSF to %s", filename or self.model_file)
            raise
        return

    def load_model_psf(self, filename=""):
        """
        Load the model PSF from a file.

        Parameters
        ----------
        filename : str, optional
            File to load the model PSF from. If not specified, uses the configured model_file.
        """
        # If no file given, first try dark file
        try:
            if filename == "":
                filename = self.model_file
            if filename == "":
                self.model = np.zeros_like(self.model)
                logger.info("No model PSF file configured; using zeros")
            else:
                self.model = np.load(filename)
                self.logger.info("Loaded model PSF from %s", filename)
        except Exception:
            logger.exception("Failed to load model PSF from %s", filename or self.model_file)
            raise
        return

    def compute_strehl(self, median_filter_size=1, gaussian_sigma=0):
        """
        Compute the rough Strehl ratio and tip tilt offset. These values are reference to the model_psf.
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

        model = clean_image_for_strehl(
            self.model, median_filter_size=median_filter_size, gaussian_sigma=gaussian_sigma
        )

        current = clean_image_for_strehl(
            self.read_long(), median_filter_size=median_filter_size, gaussian_sigma=gaussian_sigma
        )

        self.strehl_ratio = np.max(current) / np.max(model)
        self.peak_dist = np.linalg.norm(centroid(current) - centroid(self.model))

        self.write_stream("strehl", np.array([self.strehl_ratio], dtype=float))
        self.write_stream("tiptilt", np.array([self.peak_dist], dtype=float))

        return self.strehl_ratio

    def plot(self):
        """
        Plot the current short exposure PSF.
        """
        try:
            arr = self.read()
            plt.imshow(arr, cmap="inferno", origin="lower")
            plt.colorbar()
            plt.show()
            self.logger.info("Plotted science camera image")
        except Exception:
            logger.exception("Failed to plot science camera image")
            raise
        return


if __name__ == "__main__":
    launch_component(ScienceCamera, "psf", start=True)
