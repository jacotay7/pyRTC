"""Slope-processing kernels and the pyrtc slope extraction component.

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
os.environ["NUMBA_NUM_THREADS"] = "1"

import matplotlib.pyplot as plt
import numpy as np
from typing import Any
from numba import jit

from pyrtc.logging_utils import get_logger
from pyrtc.manager import launch_component
from pyrtc.streams import clear_shms, create_stream, gpu_torch_available, open_stream
from pyrtc.component import Component
from pyrtc.utils import (
    compute_fwhm_dark_subtracted_image,
    generate_circular_aperture_mask,
    set_from_config,
)

logger = get_logger(__name__)


PYWFS_NORMALIZATION_EPS = np.float32(1e-12)


def compute_slopes_pywfs_torch(
    image: Any,
    p1_mask: Any,
    p2_mask: Any,
    p3_mask: Any,
    p4_mask: Any,
    num_pixels_in_pupils: int,
    slopes: Any,
    ref_slopes: Any,
):
    """Compute normalized pyramid-WFS slopes on a torch device.

    The function extracts the four pupil images selected by the provided masks,
    forms differential x/y slope channels, normalizes by the mean total pupil
    flux, and subtracts the stored reference slopes.
    """

    if not gpu_torch_available():
        raise ImportError(
            "compute_slopes_pywfs_torch requires PyTorch. Install with 'pip install pyrtc[gpu]' or 'pip install torch'."
        )

    import torch

    # Ensure the image is in float format
    image = image.to(torch.float32)

    # Mask pupils out of the image
    p1 = image[p1_mask]
    p2 = image[p2_mask]
    p3 = image[p3_mask]
    p4 = image[p4_mask]

    # Sum pupils, saving partial sums to avoid recomputing later
    tmp1 = p1 + p2
    tmp2 = p3 + p4

    # Compute X slopes
    slopes[:num_pixels_in_pupils] = tmp1 - tmp2

    # Compute Y slopes
    slopes[num_pixels_in_pupils:] = (p1 + p3) - (p2 + p4)

    # Normalize slopes only when there is measurable pupil flux.
    mean_flux = torch.mean(tmp1 + tmp2)
    if torch.abs(mean_flux) <= PYWFS_NORMALIZATION_EPS:
        return torch.zeros_like(slopes)
    slopes = slopes / mean_flux

    # Subtract reference slopes
    return slopes - ref_slopes


"""
Optimized for best performance with numpy only
All memory is preallocated.
"""


def compute_slopes_pywfs_optim_numpy(
    image: np.ndarray,
    p1_mask: np.ndarray,
    p2_mask: np.ndarray,
    p3_mask: np.ndarray,
    p4_mask: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    p4: np.ndarray,
    tmp1: np.ndarray,
    tmp2: np.ndarray,
    num_pixels_in_pupils: int,
    slopes: np.ndarray,
    ref_slopes: np.ndarray,
):
    # Mask Pupils out of image and convert to floats
    p1 = image[p1_mask].astype(np.float32)
    p2 = image[p2_mask].astype(np.float32)
    p3 = image[p3_mask].astype(np.float32)
    p4 = image[p4_mask].astype(np.float32)
    # Sum Pupils, Saving partial sums to avoid recomputing later
    tmp1 = np.add(p1, p2)
    tmp2 = np.add(p3, p4)
    # Compute X slopes
    slopes[:num_pixels_in_pupils] = np.subtract(tmp1, tmp2)
    # Compute Y slopes
    slopes[num_pixels_in_pupils:] = np.subtract(np.add(p1, p3), np.add(p2, p4))
    # Normalize slopes only when there is measurable pupil flux.
    mean_value = np.mean(np.add(tmp1, tmp2))
    if np.abs(mean_value) <= PYWFS_NORMALIZATION_EPS:
        slopes.fill(0.0)
        return slopes
    slopes = np.divide(slopes, mean_value)
    # Subtract reference slopes
    return slopes - ref_slopes


"""
Optimized for best performance.
Works very well with numba JIT compilation.
Performed better compared to a numpy only implementation
"""


@jit(nopython=True, nogil=True, cache=False, fastmath=True)
def compute_slopes_pywfs_optim_numba(
    image: np.ndarray,
    p1_mask: np.ndarray,
    p2_mask: np.ndarray,
    p3_mask: np.ndarray,
    p4_mask: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    p4: np.ndarray,
    tmp1: np.ndarray,
    tmp2: np.ndarray,
    num_pixels_in_pupils: int,
    slopes: np.ndarray,
    ref_slopes: np.ndarray,
):
    """Compute pyramid-WFS slopes using a Numba-optimized CPU kernel."""

    # Mask Pupils out of image and convert to floats
    p1_count, p2_count, p3_count, p4_count = 0, 0, 0, 0
    for i in range(len(image)):
        if p1_mask[i]:
            p1[p1_count] = np.float32(image[i])
            p1_count += 1
        if p2_mask[i]:
            p2[p2_count] = np.float32(image[i])
            p2_count += 1
        if p3_mask[i]:
            p3[p3_count] = np.float32(image[i])
            p3_count += 1
        if p4_mask[i]:
            p4[p4_count] = np.float32(image[i])
            p4_count += 1

    # Sum Pupils, Saving partial sums to avoid recomputing later
    total_sum = 0.0
    for i in range(num_pixels_in_pupils):  # Assuming all counts are equal
        tmp1[i] = p1[i] + p2[i]
        tmp2[i] = p3[i] + p4[i]
        total_sum += tmp1[i] + tmp2[i]
    if p1_count == 0:
        for i in range(2 * num_pixels_in_pupils):
            slopes[i] = 0.0
        return slopes

    mean_value = total_sum / p1_count
    if np.abs(mean_value) <= PYWFS_NORMALIZATION_EPS:
        for i in range(2 * num_pixels_in_pupils):
            slopes[i] = 0.0
        return slopes

    for i in range(num_pixels_in_pupils):
        # Compute Y slopes
        slopes[i] = (tmp1[i] - tmp2[i]) / mean_value - ref_slopes[i]
        # Compute X slopes
        slopes[num_pixels_in_pupils + i] = (
            (p1[i] + p3[i]) - (p2[i] + p4[i])
        ) / mean_value - ref_slopes[num_pixels_in_pupils + i]

    return slopes


"""
Optimized for best performance.
Works very well with numba JIT compilation.
Performed better compared to a numpy only implementation, while also
allowing for non-integer spacing.
"""


@jit(nopython=True, nogil=True, cache=False)
def compute_slopes_shwfs_optim_numba(
    image: np.ndarray,
    slopes: np.ndarray,
    unaberrated_slopes: np.ndarray,
    threshold: np.float32,
    spacing: np.float32,
    xvals: np.ndarray,
    offset_x: int,
    offset_y: int,
    int_n: int,
):
    """Compute Shack-Hartmann centroid slopes with a Numba kernel.

    The image is traversed lenslet by lenslet, thresholded locally, and reduced
    into x/y centroid offsets relative to the unaberrated reference slopes.
    """

    # Convert image to the same dtype as unaberrated_slopes
    image = image.astype(np.float32)

    # Compute the number of sub-apertures
    num_regions = unaberrated_slopes.shape[1]

    # Loop over all regions
    for i in range(num_regions):
        for j in range(num_regions):
            # Compute where to start
            start_i = int(round(spacing * i)) + offset_y
            start_j = int(round(spacing * j)) + offset_x

            # Ensure we stay within the bounds of the image
            if start_j + int_n <= image.shape[1] and start_i + int_n <= image.shape[0]:
                # Create a local subimage around the lenslet spot
                sub_im = image[start_i : start_i + int_n, start_j : start_j + int_n]

                # loop through the sub image
                norm = np.float32(0)
                weight_x = np.float32(0)
                weight_y = np.float32(0)
                for m in range(int_n):
                    for n in range(int_n):
                        # If we are counting the pixel
                        if sub_im[m, n] > threshold:
                            # Add it to the normalization
                            norm += sub_im[m, n]
                            # Compute the X and Y centroids (before normalization)
                            weight_x += xvals[m, n] * sub_im[m, n]
                            weight_y += xvals[n, m] * sub_im[m, n]

                # If we have flux in the sub aperture
                if norm > 0:
                    # Normalize the centroids and remove the reference slope
                    slopes[i, j] = weight_x / norm - unaberrated_slopes[i, j]
                    slopes[i + num_regions, j] = (
                        weight_y / norm - unaberrated_slopes[i + num_regions, j]
                    )
                # If we have no flux slopes should be zero

    return slopes


"""
Optimized for best performance with numpy only.
Does not allow for non-integer spacing.
"""


def compute_slopes_shwfs_optim_numpy(
    image: np.ndarray,
    slopes: np.ndarray,
    unaberrated_slopes: np.ndarray,
    threshold: float,
    spacing: int,
    xvals: np.array,
):

    # Only works for integer spacings
    spacing = int(spacing)

    # Convert the image to floats and threshold in one operation
    image = np.where(image > threshold, image.astype(np.float32), 0.0)

    # Reshape the image into blocks of size spacing X spacing
    reshaped_image = image.reshape(
        image.shape[0] // spacing, spacing, image.shape[1] // spacing, spacing
    )

    # Compute the sum of pixel values in each MxM region
    region_sums = np.sum(reshaped_image, axis=(1, 3))

    # Precompute the dot products instead of tensordot (which is more general but slower)
    weighted_sum_x = np.einsum("ijkl,jl->ik", reshaped_image, xvals)
    weighted_sum_y = np.einsum("ijkl,jl->ik", reshaped_image, xvals.T)

    # Get mask for non-zero value sums
    mask = region_sums > 0.0

    # Compute the centroids directly on the valid regions
    valid_region_sums = region_sums[mask]
    slopes[: slopes.shape[1]][mask] = (
        weighted_sum_x[mask] / valid_region_sums - unaberrated_slopes[: slopes.shape[1]][mask]
    )
    slopes[slopes.shape[1] :][mask] = (
        weighted_sum_y[mask] / valid_region_sums - unaberrated_slopes[slopes.shape[1] :][mask]
    )

    # Return the difference with reference slopes
    return slopes


class SlopesProcess(Component):
    """
    A class to handle real-time slope computation for wavefront sensors.

    Config
    ------
    type : str
        Type of the WFS ("PYWFS" or "SHWFS").
    signal_type : str
        Type of signal ("slopes").
    image_noise : float, optional
        Image noise. Default is 0.0.
    central_obscuration_ratio : float, optional
        Central obscuration ratio. Default is 0.0.
    flat_norm : float, optional
        Normalization factor for the flat. Required for "PYWFS" with "slopes" signal_type.
    pupils : list of str, optional
        List of pupil locations in "x,y" format. Required for "PYWFS".
    pupils_radius : int, optional
        Radius of the pupils. Required for "PYWFS".
    contrast : float, optional
        Contrast for "SHWFS". Default is 0.
    sub_ap_spacing : float, optional
        Sub-aperture spacing for "SHWFS".
    sub_ap_offset_x : float, optional
        Sub-aperture offset in X direction for "SHWFS".
    sub_ap_offset_y : float, optional
        Sub-aperture offset in Y direction for "SHWFS".
    ref_slope_count : int, optional
        Number of reference slopes for averaging. Default is 1000.
    valid_sub_aps_file : str, optional
        File containing valid sub-aperture mask. Default is "".
    ref_slopes_file : str, optional
        File containing reference slopes. Default is "".

    Attributes
    ----------
    conf_wfs : dict
        Wavefront sensor configuration.
    name : str
        Name of the process.
    image_shape : tuple
        Shape of the WFS image.
    conf : dict
        Slopes configuration.
    wfs_meta : numpy.ndarray
        Metadata of the WFS image.
    image_dtype : type
        Data type of the WFS image.
    wfs_shm : pyshmem.SharedMemory
        Shared memory object for the WFS image.
    signal_dtype : type
        Data type of the signal.
    image_noise : float
        Image noise.
    central_obscuration_ratio : float
        Central obscuration ratio.
    wfs_type : str
        Type of the WFS.
    signal_type : str
        Type of signal.
    valid_sub_aps : numpy.ndarray or None
        Valid sub-aperture mask.
    shwfs_contrast : float
        Contrast for "SHWFS".
    sub_ap_spacing : float
        Sub-aperture spacing for "SHWFS".
    num_regions : int
        Number of regions for "SHWFS".
    offset_x : float
        Sub-aperture offset in X direction for "SHWFS".
    offset_y : float
        Sub-aperture offset in Y direction for "SHWFS".
    ref_slope_count : int
        Number of reference slopes for averaging.
    signal_2d_size : int
        Size of the 2D signal.
    signal_2d_shape : tuple
        Shape of the 2D signal.
    valid_sub_aps_file : str
        File containing valid sub-aperture mask.
    signal_size : int
        Size of the signal.
    signal_shape : tuple
        Shape of the signal.
    signal : pyshmem.SharedMemory
        Shared memory object for the signal.
    signal_2d : pyshmem.SharedMemory
        Shared memory object for the 2D signal.
    ref_slopes_file : str
        File containing reference slopes.
    ref_slopes : numpy.ndarray
        Reference slopes.
    gpu_device : str
        Default device if using GPU
    flat_norm : float
        Normalization factor for the flat.
    pupil_locs : list of tuple
        List of pupil locations.
    pupil_radius : int
        Radius of the pupils.
    pupil_mask : numpy.ndarray
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

            self.wfs_shm = open_stream(self.input_stream_name("wfs"), gpu_device=self.gpu_device)
            self.image_shape = tuple(self.wfs_shm.shape)
            self.image_dtype = np.dtype(self.wfs_shm.dtype)
            # Pre-allocated hot-path read buffer (ignored for GPU streams).
            self._image_buffer = np.empty(self.image_shape, dtype=self.image_dtype)
            self.register_input_stream("wfs", self.wfs_shm)

            self.signal_dtype = np.float32
            self.image_noise = set_from_config(self.conf, "image_noise", 0.0)
            self.central_obscuration_ratio = set_from_config(
                self.conf, "central_obscuration_ratio", 0.0
            )

            self.wfs_type = self.conf["type"].lower()
            self.signal_type = self.conf["signal_type"]
            self.valid_sub_aps = None
            self.valid_sub_aps_file = set_from_config(self.conf, "valid_sub_aps_file", "")

            self.ref_slopes_file = set_from_config(self.conf, "ref_slopes_file", "")
            self.ref_slope_count = set_from_config(self.conf, "ref_slope_count", 1000)

            if self.wfs_type == "pywfs":
                if "pupils" in self.conf.keys():
                    pupil_locs = [
                        (int(x.split(",")[1]), int(x.split(",")[0])) for x in self.conf["pupils"]
                    ]
                    self.set_pupils(pupil_locs, self.conf["pupils_radius"])
                else:
                    a, b = int(0.25 * self.image_shape[0]), int(0.75 * self.image_shape[0])
                    c, d = int(0.25 * self.image_shape[1]), int(0.75 * self.image_shape[1])
                    r = min(self.image_shape[0] - b, self.image_shape[1] - d)
                    self.set_pupils([(a, c), (a, d), (b, c), (b, d)], r)
                if self.signal_type == "slopes":
                    self.flat_norm = set_from_config(self.conf, "flat_norm", True)

                self.ref_slopes = np.zeros(self.signal_2d_shape, dtype=self.signal_dtype)
                self.ref_slopes_1d = np.zeros_like(self.signal.read())
                self.slopes_arr_1d = np.zeros_like(self.ref_slopes_1d)
                self.num_pixels_in_pupils = np.count_nonzero(self.p1mask)
                self.p1 = np.empty(self.num_pixels_in_pupils, dtype=self.signal_dtype)
                self.p2 = np.empty_like(self.p1)
                self.p3 = np.empty_like(self.p1)
                self.p4 = np.empty_like(self.p1)
                self.tmp1, self.tmp2 = np.empty_like(self.p1), np.empty_like(self.p1)

            elif self.wfs_type == "shwfs":
                self.shwfs_contrast = set_from_config(self.conf, "contrast", 0.0)
                self.sub_ap_spacing = self.conf["sub_ap_spacing"]
                self.region_size = int(np.round(self.sub_ap_spacing, 0))
                self.num_regions = self.image_shape[0] // self.region_size
                self.offset_x = self.conf["sub_ap_offset_x"]
                self.offset_y = self.conf["sub_ap_offset_y"]
                xvals = np.arange(self.region_size).astype(int) - self.region_size // 2
                self.xvals = np.meshgrid(xvals, xvals)[0].astype(self.signal_dtype)

                self.signal_2d_size = int(2 * self.num_regions**2)
                self.signal_2d_shape = (2 * self.num_regions, self.num_regions)

                self.valid_sub_aps = np.ones(self.signal_2d_shape, dtype=bool)
                self.load_valid_sub_aps()

                self.signal_size = np.sum(self.valid_sub_aps)
                self.signal_shape = (self.signal_size,)

                logger.info(
                    "SHWFS slopes configured sub_ap_spacing=%s num_regions=%s offset_x=%s offset_y=%s signal_size=%s signal_shape=%s signal_dtype=%s",
                    self.sub_ap_spacing,
                    self.num_regions,
                    self.offset_x,
                    self.offset_y,
                    self.signal_size,
                    self.signal_shape,
                    self.signal_dtype,
                )

                self._configure_signal_streams(self.signal_shape, self.signal_2d_shape)

                self.ref_slopes = np.zeros(self.signal_2d_shape, dtype=self.signal_dtype)

            self.load_ref_slopes()
            self.logger.info(
                "Initialized slopes process wfs_type=%s signal_type=%s image_shape=%s",
                self.wfs_type,
                self.signal_type,
                self.image_shape,
            )
        except Exception:
            logger.exception("Failed to initialize slopes process")
            raise

    def _close_signal_streams(self) -> None:
        """Close any currently attached signal output streams."""

        for attribute_name in ("signal", "signal_2d"):
            stream = getattr(self, attribute_name, None)
            if stream is None or not hasattr(stream, "close"):
                continue
            try:
                stream.close()
            except Exception:
                logger.debug(
                    "Failed closing %s stream during rebuild", attribute_name, exc_info=True
                )

    def _existing_output_stream_matches(
        self, stream_name: str, expected_shape: tuple[int, ...]
    ) -> bool:
        """Return ``True`` when an existing SHM matches the expected output shape."""

        try:
            stream = open_stream(self.output_stream_name(stream_name), gpu_device=self.gpu_device)
        except Exception:
            return False

        try:
            return tuple(stream.shape) == tuple(expected_shape) and np.dtype(
                stream.dtype
            ) == np.dtype(self.signal_dtype)
        finally:
            try:
                stream.close()
            except Exception:
                logger.debug(
                    "Failed closing temporary stream probe for %s", stream_name, exc_info=True
                )

    def _configure_signal_streams(
        self,
        signal_shape: tuple[int, ...],
        signal2d_shape: tuple[int, ...],
        *,
        rebuild: bool = False,
    ) -> None:
        """Create or rebuild the signal and signal_2d output streams.

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

        self.signal_shape = tuple(int(axis) for axis in signal_shape)
        self.signal_2d_shape = tuple(int(axis) for axis in signal2d_shape)

        needs_rebuild = rebuild
        if not needs_rebuild:
            needs_rebuild = not self._existing_output_stream_matches("signal", self.signal_shape)
        if not needs_rebuild:
            needs_rebuild = not self._existing_output_stream_matches(
                "signal_2d", self.signal_2d_shape
            )

        if needs_rebuild:
            self._close_signal_streams()
            clear_shms([self.output_stream_name("signal"), self.output_stream_name("signal_2d")])

        self.signal = create_stream(
            self.output_stream_name("signal"),
            self.signal_shape,
            self.signal_dtype,
            gpu_device=self.gpu_device,
        )
        self.signal_2d = create_stream(
            self.output_stream_name("signal_2d"),
            self.signal_2d_shape,
            self.signal_dtype,
            gpu_device=self.gpu_device,
        )
        self.register_output_stream("signal", self.signal)
        self.register_output_stream("signal_2d", self.signal_2d)

    def read(self, block=True):
        """
        Read the current signal.

        Returns
        -------
        numpy.ndarray
            Current signal.
        """
        if block:
            return self.read_stream("signal")
        return self.read_stream("signal", block=False)

    def read_image(self, block=True):
        """
        Read the current WFS image.

        Returns
        -------
        numpy.ndarray
            Current WFS image.
        """
        if block:
            return self.read_stream("wfs")
        return self.read_stream("wfs", block=False)

    def set_valid_sub_aps(self, valid_sub_aps):
        """
        Set the valid sub-aperture mask. Converts to boolean if not already

        Parameters
        ----------
        valid_sub_aps : numpy.ndarray
            Valid sub-aperture mask.
        """
        component_logger = getattr(self, "logger", logger)
        try:
            self.valid_sub_aps = valid_sub_aps.astype(bool)
            self.cur_signal_2d = np.zeros(valid_sub_aps.shape)
            component_logger.info("Set valid sub-aperture mask shape=%s", valid_sub_aps.shape)
        except Exception:
            component_logger.exception("Failed to set valid sub-aperture mask")
            raise
        return

    def save_valid_sub_aps(self, filename=""):
        """
        Save the valid sub-aperture mask to a file.

        Parameters
        ----------
        filename : str, optional
            File to save the valid sub-aperture mask to. If not specified, uses the configured valid_sub_aps_file.
        """
        component_logger = getattr(self, "logger", logger)
        try:
            if filename == "":
                filename = self.valid_sub_aps_file
            if filename == "":
                raise ValueError("No valid_sub_aps filename provided")
            np.save(filename, self.valid_sub_aps)
            component_logger.info("Saved valid sub-aperture mask to %s", filename)
        except Exception:
            component_logger.exception(
                "Failed to save valid sub-aperture mask to %s",
                filename or getattr(self, "valid_sub_aps_file", ""),
            )
            raise
        return

    def load_valid_sub_aps(self, filename=""):
        """
        Load the valid sub-aperture mask from a file.

        Parameters
        ----------
        filename : str, optional
            File to load the valid sub-aperture mask from. If not specified, uses the configured valid_sub_aps_file.
        """
        # If no file given, first try reference slopes file
        component_logger = getattr(self, "logger", logger)
        try:
            if filename == "":
                filename = self.valid_sub_aps_file
            if filename == "":
                valid_sub_aps = np.ones_like(self.valid_sub_aps)
                component_logger.info("No valid_sub_aps file configured; using all-true mask")
            else:
                valid_sub_aps = np.load(filename)
                component_logger.info("Loaded valid sub-aperture mask from %s", filename)

            self.set_valid_sub_aps(valid_sub_aps)
        except Exception:
            component_logger.exception(
                "Failed to load valid sub-aperture mask from %s",
                filename or getattr(self, "valid_sub_aps_file", ""),
            )
            raise

        return

    def take_ref_slopes(self):
        """
        Take reference slopes by averaging multiple slope measurements. Number of measurements
        set by ref_slope_count variable.
        """
        component_logger = getattr(self, "logger", logger)
        try:
            if self.ref_slope_count < 1:
                raise ValueError("ref_slope_count must be at least 1")
            component_logger.info("Taking reference slopes using %s frames", self.ref_slope_count)
            self.set_ref_slopes(np.zeros_like(self.ref_slopes))
            ref_slopes = np.zeros_like(self.ref_slopes)
            for _ in range(self.ref_slope_count):
                cur_slopes = self.read().astype(ref_slopes.dtype)
                ref_slopes += self.compute_signal_2d(cur_slopes)
            ref_slopes /= self.ref_slope_count
            self.set_ref_slopes(ref_slopes)
            component_logger.info("Completed reference slope acquisition")
        except Exception:
            component_logger.exception("Failed to take reference slopes")
            raise
        return

    def set_ref_slopes(self, ref_slopes):
        """
        Set the reference slopes.

        Parameters
        ----------
        ref_slopes : numpy.ndarray
            Reference slopes.
        """
        component_logger = getattr(self, "logger", logger)
        try:
            self.ref_slopes = ref_slopes.astype(self.signal_dtype)
            if self.wfs_type == "pywfs":
                slopemask = self.valid_sub_aps[:, : self.valid_sub_aps.shape[1] // 2]
                self.ref_slopes_1d = np.zeros_like(self.signal.read())
                self.ref_slopes_1d[: self.ref_slopes_1d.size // 2] = self.ref_slopes[
                    :, : self.ref_slopes.shape[1] // 2
                ][slopemask]
                self.ref_slopes_1d[self.ref_slopes_1d.size // 2 :] = self.ref_slopes[
                    :, self.ref_slopes.shape[1] // 2 :
                ][slopemask]
            component_logger.info("Updated reference slopes")
        except Exception:
            component_logger.exception("Failed to update reference slopes")
            raise

        return

    def save_ref_slopes(self, filename=""):
        """
        Save the reference slopes to a file.

        Parameters
        ----------
        filename : str, optional
            File to save the reference slopes to. If not specified, uses the configured ref_slopes_file.
        """
        component_logger = getattr(self, "logger", logger)
        try:
            if filename == "":
                filename = self.ref_slopes_file
            if filename == "":
                raise ValueError("No reference slopes filename provided")
            np.save(filename, self.ref_slopes)
            component_logger.info("Saved reference slopes to %s", filename)
        except Exception:
            component_logger.exception(
                "Failed to save reference slopes to %s",
                filename or getattr(self, "ref_slopes_file", ""),
            )
            raise
        return

    def load_ref_slopes(self, filename=""):
        """
        Load the reference slopes from a file.

        Parameters
        ----------
        filename : str, optional
            File to load the reference slopes from. If not specified, uses the configured ref_slopes_file.
        """
        # If no file given, first try reference slopes file
        component_logger = getattr(self, "logger", logger)
        try:
            if filename == "":
                filename = self.ref_slopes_file
            if filename == "":
                ref_slopes = np.zeros_like(self.ref_slopes)
                component_logger.info("No reference slopes file configured; using zeros")
            else:
                ref_slopes = np.load(filename)
                component_logger.info("Loaded reference slopes from %s", filename)

            self.set_ref_slopes(ref_slopes)
        except Exception:
            component_logger.exception(
                "Failed to load reference slopes from %s",
                filename or getattr(self, "ref_slopes_file", ""),
            )
            raise
        return

    def compute_signal(self):
        """
        Compute the signal from the WFS image.
        """
        image = self.read_stream("wfs", out=self._image_buffer)
        if self.signal_type == "slopes":
            if self.wfs_type == "pywfs":
                if self.gpu_device is not None and gpu_torch_available():
                    import torch

                    slope_signal = (
                        compute_slopes_pywfs_torch(
                            image.ravel(),
                            p1_mask=torch.from_numpy(self.p1mask.ravel()).to(self.gpu_device),
                            p2_mask=torch.from_numpy(self.p2mask.ravel()).to(self.gpu_device),
                            p3_mask=torch.from_numpy(self.p3mask.ravel()).to(self.gpu_device),
                            p4_mask=torch.from_numpy(self.p4mask.ravel()).to(self.gpu_device),
                            num_pixels_in_pupils=self.num_pixels_in_pupils,
                            slopes=torch.from_numpy(self.slopes_arr_1d).to(self.gpu_device),
                            ref_slopes=torch.from_numpy(self.ref_slopes_1d).to(self.gpu_device),
                        )
                        .cpu()
                        .numpy()
                    )
                else:
                    slope_signal = compute_slopes_pywfs_optim_numba(
                        image=image.ravel(),
                        p1_mask=self.p1mask.ravel(),
                        p2_mask=self.p2mask.ravel(),
                        p3_mask=self.p3mask.ravel(),
                        p4_mask=self.p4mask.ravel(),
                        p1=self.p1,
                        p2=self.p2,
                        p3=self.p3,
                        p4=self.p4,
                        tmp1=self.tmp1,
                        tmp2=self.tmp2,
                        num_pixels_in_pupils=self.num_pixels_in_pupils,
                        slopes=self.slopes_arr_1d,
                        ref_slopes=self.ref_slopes_1d,
                    )

            elif self.wfs_type == "shwfs":
                slopes = compute_slopes_shwfs_optim_numba(
                    image=image,
                    slopes=np.zeros_like(self.ref_slopes),
                    unaberrated_slopes=self.ref_slopes,
                    threshold=self.image_noise * self.shwfs_contrast,
                    spacing=self.sub_ap_spacing,
                    xvals=self.xvals,
                    offset_x=self.offset_x,
                    offset_y=self.offset_y,
                    int_n=self.region_size,
                )
                slope_signal = slopes[self.valid_sub_aps]
                # self.signal.write(slopes[self.valid_sub_aps])
                # self.signal_2d.write(slopes*self.valid_sub_aps)
                # slopes = np.zeros_like(self.ref_slopes)
                # self.signal.write(self.ref_slopes.flatten()[:np.prod(self.signal_shape)].reshape(self.signal_shape))
            self.write_stream("signal", slope_signal)
            self.write_stream("signal_2d", self.compute_signal_2d(slope_signal))

        return

    def compute_image_noise(self):
        """
        Compute the image noise. Useful to set a good SNR cutoff for SHWFS
        """
        component_logger = getattr(self, "logger", logger)
        try:
            img = self.read_image()
            if img[img < 0].size > 0:
                self.image_noise = compute_fwhm_dark_subtracted_image(img) / 2
                component_logger.info("Computed image noise=%s", self.image_noise)
            else:
                logger.warning("Image is not dark subtracted")
        except Exception:
            component_logger.exception("Failed to compute image noise")
            raise
        return

    def set_pupils(self, pupil_locs, pupil_radius):
        """
        Set the pupils' locations and radius. First computes a Pupil Mask, then generates slope mask
        and sets up SHMS of the correct sizes.

        Parameters
        ----------
        pupil_locs : list of tuple
            List of pupil locations.
        pupil_radius : int
            Radius of the pupils.
        """
        component_logger = getattr(self, "logger", logger)
        try:
            self.pupil_locs = pupil_locs
            self.pupil_radius = pupil_radius
            self.compute_pupils_mask()
            if self.signal_type == "slopes":
                self.signal_size = np.count_nonzero(self.pupil_mask) // 2
                slopemask = (
                    self.pupil_mask[
                        self.pupil_locs[0][1] - self.pupil_radius : self.pupil_locs[0][1]
                        + self.pupil_radius,
                        self.pupil_locs[0][0] - self.pupil_radius : self.pupil_locs[0][0]
                        + self.pupil_radius,
                    ]
                    > 0
                )
                self.set_valid_sub_aps(np.concatenate([slopemask, slopemask], axis=1))
                if self.valid_sub_aps_file != "":
                    self.save_valid_sub_aps()
                self._configure_signal_streams(
                    (self.signal_size,),
                    (self.valid_sub_aps.shape[0], self.valid_sub_aps.shape[1]),
                    rebuild=True,
                )
            component_logger.info("Configured pupils locs=%s radius=%s", pupil_locs, pupil_radius)
        except Exception:
            component_logger.exception(
                "Failed to set pupils locs=%s radius=%s", pupil_locs, pupil_radius
            )
            raise

        return

    def compute_pupils_mask(self):
        """
        Compute the mask for the pupils. Assumes circular aperture with obstruction ratio
        set by the central_obscuration_ratio parameter.
        """
        component_logger = getattr(self, "logger", logger)
        try:
            self.pupil_mask = np.zeros(self.image_shape)

            pupil_template = generate_circular_aperture_mask(
                int(np.ceil(2 * self.pupil_radius)),
                self.pupil_radius,
                self.central_obscuration_ratio,
            )
            N = self.pupil_mask.shape[0]
            n = pupil_template.shape[0]
            half_n = n // 2

            for i, pupil_loc in enumerate(self.pupil_locs):
                px, py = pupil_loc

                x_start = px - half_n
                x_end = px + half_n + (n % 2)
                y_start = py - half_n
                y_end = py + half_n + (n % 2)

                if x_start < 0 or y_start < 0 or x_end > N or y_end > N:
                    raise ValueError("The subimage exceeds the bounds of the larger array.")

                self.pupil_mask[y_start:y_end, x_start:x_end] += pupil_template * (i + 1)

            self.p1mask = self.pupil_mask == 1
            self.p2mask = self.pupil_mask == 2
            self.p3mask = self.pupil_mask == 3
            self.p4mask = self.pupil_mask == 4
            component_logger.info("Computed pupil masks for %s pupils", len(self.pupil_locs))
        except Exception:
            component_logger.exception("Failed to compute pupil mask")
            raise
        return

    def plot_pupils(self):
        """
        Plot the pupil mask to see if its right.
        """
        # plt.figure(figsize=(10,8))
        plt.imshow(self.pupil_mask, cmap="inferno", origin="lower", aspect="auto")
        plt.colorbar()
        plt.title("Pupil Mask (Value is Pupil Number)")
        plt.show()

        plt.imshow(
            self.pupil_mask * self.read_image(), cmap="inferno", origin="lower", aspect="auto"
        )
        colors = ["g", "b", "orange", "r"]
        for i in range(len(self.pupil_locs)):
            px, py = self.pupil_locs[i]
            plt.axvline(x=px, color=colors[i], alpha=0.6)
            plt.axhline(y=py, color=colors[i], alpha=0.6)
        plt.colorbar()
        plt.title("Pupil Mask * Image ")
        plt.show()
        return

    def compute_signal_2d(self, signal, valid_sub_aps=None):
        """
        Compute the 2D signal from the valid sub-aperture mask.

        Parameters
        ----------
        signal : numpy.ndarray
            Signal to process.
        valid_sub_aps : numpy.ndarray, optional
            Valid sub-aperture mask. If not provided, uses the current valid sub-aperture mask.

        Returns
        -------
        numpy.ndarray
            2D signal.
        """
        if valid_sub_aps is None and isinstance(self.valid_sub_aps, np.ndarray):
            valid_sub_aps = self.valid_sub_aps
        else:
            return -1

        if self.wfs_type == "pywfs":
            slopemask = valid_sub_aps[:, : valid_sub_aps.shape[1] // 2]
            self.cur_signal_2d[:, : valid_sub_aps.shape[1] // 2][slopemask] = signal[
                : signal.size // 2
            ]
            self.cur_signal_2d[:, valid_sub_aps.shape[1] // 2 :][slopemask] = signal[
                signal.size // 2 :
            ]
        else:
            self.cur_signal_2d[self.valid_sub_aps] = signal
        return self.cur_signal_2d


if __name__ == "__main__":
    launch_component(SlopesProcess, "slopes", start=True)
