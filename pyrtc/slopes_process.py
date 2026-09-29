"""Slope-processing kernels and the pyrtc slope extraction component.

This module turns wavefront-sensor camera frames into the residual slope or
signal vectors consumed by the AO loop. It includes optimized CPU and GPU
helpers for pyramid and Shack-Hartmann processing plus the ``SlopesProcess``
component that manages calibration data and SHM publication.
"""

import threading

import numpy as np
from typing import Any
from numba import jit

from pyrtc.logging_utils import get_logger
from pyrtc.manager import launch_component
from pyrtc.streams import clear_shms, create_stream, gpu_torch_available, open_stream
from pyrtc.component import Component
from pyrtc.utils import (
    pyplot,
    compute_fwhm_dark_subtracted_image,
    generate_circular_aperture_mask,
    set_from_config,
)

logger = get_logger(__name__)


PYWFS_NORMALIZATION_EPS = np.float32(1e-12)


def _host_array(value) -> np.ndarray:
    """Return ``value`` as a NumPy array, copying torch tensors to the host."""

    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


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

    Each ``p*_mask`` may be a boolean mask over the flattened image or an
    integer index tensor of the selected pixels (row-major order, as
    ``torch.nonzero`` returns). Index tensors avoid the per-frame
    boolean-mask compaction and are what ``SlopesProcess`` caches.
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

    # Normalize slopes only when there is measurable pupil flux. Select on the
    # device rather than branching on the value, which would force a host sync.
    mean_flux = torch.mean(tmp1 + tmp2)
    dark = torch.abs(mean_flux) <= float(PYWFS_NORMALIZATION_EPS)
    normalized = slopes / torch.where(dark, torch.ones_like(mean_flux), mean_flux) - ref_slopes
    return torch.where(dark, torch.zeros_like(normalized), normalized)


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


@jit(nopython=True, nogil=True, cache=True, fastmath=True)
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


@jit(nopython=True, nogil=True, cache=True)
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

    Pixels are converted to float32 as they are read, so the kernel allocates
    nothing and ``slopes`` can be reused across frames. Every entry of
    ``slopes`` is written: sub-apertures without flux above threshold, or
    falling outside the image, are set to 0.
    """

    # Compute the number of sub-apertures
    num_regions = unaberrated_slopes.shape[1]

    # Loop over all regions
    for i in range(num_regions):
        for j in range(num_regions):
            # Compute where to start
            start_i = int(round(spacing * i)) + offset_y
            start_j = int(round(spacing * j)) + offset_x

            # Sub-apertures without flux (or outside the image) read 0
            slopes[i, j] = 0.0
            slopes[i + num_regions, j] = 0.0

            # Ensure we stay within the bounds of the image
            if start_j + int_n > image.shape[1] or start_i + int_n > image.shape[0]:
                continue

            # A view of the lenslet's sub-image (no copy)
            sub_im = image[start_i : start_i + int_n, start_j : start_j + int_n]

            # loop through the sub image
            norm = np.float32(0)
            weight_x = np.float32(0)
            weight_y = np.float32(0)
            for m in range(int_n):
                for n in range(int_n):
                    value = np.float32(sub_im[m, n])
                    # If we are counting the pixel
                    if value > threshold:
                        # Add it to the normalization
                        norm += value
                        # Compute the X and Y centroids (before normalization)
                        weight_x += xvals[m, n] * value
                        weight_y += xvals[n, m] * value

            # If we have flux in the sub aperture
            if norm > 0:
                # Normalize the centroids and remove the reference slope
                slopes[i, j] = weight_x / norm - unaberrated_slopes[i, j]
                slopes[i + num_regions, j] = (
                    weight_y / norm - unaberrated_slopes[i + num_regions, j]
                )

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


SHWFS_CENTROIDERS = ("cog", "wcog", "correlation")
FWHM_PER_SIGMA = float(2.0 * np.sqrt(2.0 * np.log(2.0)))


def shwfs_subaperture_coords(int_n: int) -> np.ndarray:
    """Return the 1D pixel coordinates used by every SHWFS centroider.

    Pixel ``k`` of a sub-aperture sits at ``k - int_n // 2``, the same
    convention as the ``xvals`` grid of the thresholded CoG kernel, so all
    centroiders report spot positions in the same frame.
    """

    return (np.arange(int(int_n)) - int(int_n) // 2).astype(np.float32)


def wcog_gain_correction(weight_fwhm: float, spot_fwhm: float) -> float:
    """Return the factor that undoes the WCoG gain loss for a Gaussian spot.

    For a Gaussian spot of standard deviation ``s`` weighted by a Gaussian of
    standard deviation ``w`` centred on the reference, the weighted CoG
    measures ``w**2 / (w**2 + s**2)`` of the true displacement. The inverse,
    ``1 + (spot_fwhm / weight_fwhm) ** 2``, is exact only for Gaussian spots
    close to the weight centre; returns ``1.0`` when ``spot_fwhm <= 0``.
    """

    if spot_fwhm <= 0:
        return 1.0
    if weight_fwhm <= 0:
        raise ValueError("weight_fwhm must be > 0")
    return float(1.0 + (float(spot_fwhm) / float(weight_fwhm)) ** 2)


@jit(nopython=True, nogil=True, cache=True)
def build_shwfs_wcog_weights_numba(
    coords: np.ndarray,
    weight_centers: np.ndarray,
    inv_two_sigma2: np.float32,
    weights_x: np.ndarray,
    weights_y: np.ndarray,
):
    """Fill the separable WCoG Gaussian weights of every sub-aperture.

    ``weights_x[i, j, n]`` and ``weights_y[i, j, m]`` receive
    ``exp(-(coords - c)**2 * inv_two_sigma2)`` for the weight centre ``c`` of
    sub-aperture ``(i, j)`` (x in ``weight_centers[:N]``, y in
    ``weight_centers[N:]``). The weights only change with the centres or the
    FWHM, so they are built once rather than per frame.
    """

    num_regions = weights_x.shape[0]
    for i in range(num_regions):
        for j in range(num_regions):
            cx = weight_centers[i, j]
            cy = weight_centers[i + num_regions, j]
            for k in range(coords.shape[0]):
                dx = coords[k] - cx
                dy = coords[k] - cy
                weights_x[i, j, k] = np.exp(-dx * dx * inv_two_sigma2)
                weights_y[i, j, k] = np.exp(-dy * dy * inv_two_sigma2)
    return weights_x, weights_y


@jit(nopython=True, nogil=True, cache=True)
def compute_slopes_shwfs_wcog_numba(
    image: np.ndarray,
    slopes: np.ndarray,
    unaberrated_slopes: np.ndarray,
    threshold: np.float32,
    spacing: np.float32,
    coords: np.ndarray,
    offset_x: int,
    offset_y: int,
    int_n: int,
    weight_centers: np.ndarray,
    weights_x: np.ndarray,
    weights_y: np.ndarray,
    gain: np.float32,
):
    """Compute Shack-Hartmann slopes with a Gaussian-weighted CoG.

    Pixels above ``threshold`` are multiplied by the separable Gaussian
    weights ``weights_x[i, j] x weights_y[i, j]`` (see
    :func:`build_shwfs_wcog_weights_numba`) centred on ``weight_centers``
    (pixel offsets in the ``coords`` frame, x in rows ``[:N]`` and y in rows
    ``[N:]``). The displacement from the weight centre is multiplied by
    ``gain`` (``1`` leaves the WCoG gain loss uncorrected, see
    :func:`wcog_gain_correction`) and added back to the centre, so the result
    is a spot position in the same frame as the thresholded CoG. The
    reference ``unaberrated_slopes`` are then subtracted.

    Every entry of ``slopes`` is written: sub-apertures without flux above
    threshold, or falling outside the image, are set to 0.
    """

    num_regions = unaberrated_slopes.shape[1]
    for i in range(num_regions):
        for j in range(num_regions):
            start_i = int(round(spacing * i)) + offset_y
            start_j = int(round(spacing * j)) + offset_x
            slopes[i, j] = 0.0
            slopes[i + num_regions, j] = 0.0
            if start_j + int_n > image.shape[1] or start_i + int_n > image.shape[0]:
                continue

            cx = weight_centers[i, j]
            cy = weight_centers[i + num_regions, j]
            norm = 0.0
            sum_x = 0.0
            sum_y = 0.0
            for m in range(int_n):
                row_norm = 0.0
                row_x = 0.0
                for n in range(int_n):
                    value = np.float32(image[start_i + m, start_j + n])
                    if value > threshold:
                        weighted = value * weights_x[i, j, n]
                        row_norm += weighted
                        row_x += weighted * coords[n]
                row_weight = weights_y[i, j, m]
                norm += row_weight * row_norm
                sum_x += row_weight * row_x
                sum_y += row_weight * row_norm * coords[m]

            if norm > 0:
                slopes[i, j] = cx + gain * (sum_x / norm - cx) - unaberrated_slopes[i, j]
                slopes[i + num_regions, j] = (
                    cy + gain * (sum_y / norm - cy) - unaberrated_slopes[i + num_regions, j]
                )

    return slopes


@jit(nopython=True, nogil=True, cache=True)
def build_shwfs_correlation_templates_numba(
    reference_image: np.ndarray,
    threshold: np.float32,
    spacing: np.float32,
    offset_x: int,
    offset_y: int,
    int_n: int,
    search_radius: int,
    templates: np.ndarray,
    template_flux: np.ndarray,
):
    """Fill the per-sub-aperture correlation templates from a reference frame.

    ``templates[i, j]`` receives the thresholded central
    ``int_n - 2 * search_radius`` square of sub-aperture ``(i, j)`` and
    ``template_flux[i, j]`` the thresholded flux of the whole sub-aperture
    (0 for sub-apertures without flux or outside the image).
    """

    num_regions = template_flux.shape[0]
    core = int_n - 2 * search_radius
    for i in range(num_regions):
        for j in range(num_regions):
            start_i = int(round(spacing * i)) + offset_y
            start_j = int(round(spacing * j)) + offset_x
            template_flux[i, j] = 0.0
            for a in range(core):
                for b in range(core):
                    templates[i, j, a, b] = 0.0
            if (
                start_j + int_n > reference_image.shape[1]
                or start_i + int_n > reference_image.shape[0]
            ):
                continue
            flux = 0.0
            for m in range(int_n):
                for n in range(int_n):
                    value = np.float32(reference_image[start_i + m, start_j + n])
                    if value > threshold:
                        flux += value
                        a = m - search_radius
                        b = n - search_radius
                        if 0 <= a < core and 0 <= b < core:
                            templates[i, j, a, b] = value
            template_flux[i, j] = flux
    return templates, template_flux


@jit(nopython=True, nogil=True, cache=True)
def _subpixel_minimum_3x3(scores, best_i, best_j, num_shifts):
    """Return the sub-pixel ``(dx, dy)`` offset of a score minimum.

    Uses the 2D quadratic interpolation (2QI) of Löfdahl (2010, A&A 524,
    A90) over the 3x3 neighbourhood, which unlike two 1D parabolas accounts
    for the cross term of non-separable (extended) scenes. On the edge of the
    search window, or when the quadratic has no minimum, it falls back to a
    three-point parabola along each axis that has both neighbours, and to 0
    otherwise. Offsets are clamped to +/- 1 pixel.
    """

    s0 = scores[best_i, best_j]
    has_x = 0 < best_j < num_shifts - 1
    has_y = 0 < best_i < num_shifts - 1
    if has_x and has_y:
        a2 = 0.5 * (scores[best_i, best_j + 1] - scores[best_i, best_j - 1])
        a3 = 0.5 * (scores[best_i, best_j + 1] - 2.0 * s0 + scores[best_i, best_j - 1])
        a4 = 0.5 * (scores[best_i + 1, best_j] - scores[best_i - 1, best_j])
        a5 = 0.5 * (scores[best_i + 1, best_j] - 2.0 * s0 + scores[best_i - 1, best_j])
        a6 = 0.25 * (
            scores[best_i + 1, best_j + 1]
            - scores[best_i + 1, best_j - 1]
            - scores[best_i - 1, best_j + 1]
            + scores[best_i - 1, best_j - 1]
        )
        det = a6 * a6 - 4.0 * a3 * a5
        if a3 > 0 and a5 > 0 and det < 0:
            dx = (2.0 * a2 * a5 - a4 * a6) / det
            dy = (2.0 * a3 * a4 - a2 * a6) / det
            return min(1.0, max(-1.0, dx)), min(1.0, max(-1.0, dy))
    dx = 0.0
    dy = 0.0
    if has_x:
        left = scores[best_i, best_j - 1]
        right = scores[best_i, best_j + 1]
        denom = left - 2.0 * s0 + right
        if denom > 0:
            dx = min(1.0, max(-1.0, 0.5 * (left - right) / denom))
    if has_y:
        up = scores[best_i - 1, best_j]
        down = scores[best_i + 1, best_j]
        denom = up - 2.0 * s0 + down
        if denom > 0:
            dy = min(1.0, max(-1.0, 0.5 * (up - down) / denom))
    return dx, dy


@jit(nopython=True, nogil=True, cache=True, fastmath=True)
def compute_slopes_shwfs_correlation_numba(
    image: np.ndarray,
    slopes: np.ndarray,
    unaberrated_slopes: np.ndarray,
    threshold: np.float32,
    spacing: np.float32,
    offset_x: int,
    offset_y: int,
    int_n: int,
    templates: np.ndarray,
    template_flux: np.ndarray,
    ref_positions: np.ndarray,
    search_radius: int,
    window: np.ndarray,
    scores: np.ndarray,
):
    """Compute Shack-Hartmann slopes by correlating against reference templates.

    For each sub-aperture the thresholded live image is flux-normalised to the
    template's flux, and the squared-difference function between the
    template (the central ``int_n - 2 * search_radius`` square of the
    reference, see :func:`build_shwfs_correlation_templates_numba`) and the
    live image is evaluated for every integer shift within
    ``+/- search_radius`` pixels. The template always overlaps the live
    window fully, so there is no overlap bias. The minimum is refined to
    sub-pixel precision with a three-point parabola along each axis; a
    minimum on the edge of the search window is not refined (the shift is
    clamped to the window).

    The measured shift is added to ``ref_positions`` (the thresholded CoG of
    the reference frame) so the result is a spot position in the same frame
    as the CoG centroiders, and ``unaberrated_slopes`` is subtracted.

    ``window`` (``int_n x int_n``) and ``scores``
    (``(2R+1) x (2R+1)``) are float scratch buffers. Every entry of
    ``slopes`` is written: sub-apertures without flux above threshold (live or
    reference), or outside the image, are set to 0.
    """

    num_regions = unaberrated_slopes.shape[1]
    num_shifts = 2 * search_radius + 1
    core = int_n - 2 * search_radius
    for i in range(num_regions):
        for j in range(num_regions):
            start_i = int(round(spacing * i)) + offset_y
            start_j = int(round(spacing * j)) + offset_x
            slopes[i, j] = 0.0
            slopes[i + num_regions, j] = 0.0
            if start_j + int_n > image.shape[1] or start_i + int_n > image.shape[0]:
                continue
            ref_flux = template_flux[i, j]
            if ref_flux <= 0:
                continue

            flux = 0.0
            for m in range(int_n):
                for n in range(int_n):
                    value = np.float32(image[start_i + m, start_j + n])
                    if value > threshold:
                        window[m, n] = value
                        flux += value
                    else:
                        window[m, n] = 0.0
            if flux <= 0:
                continue
            scale = np.float32(ref_flux / flux)
            for m in range(int_n):
                for n in range(int_n):
                    window[m, n] *= scale

            template = templates[i, j]
            best = np.inf
            best_i = search_radius
            best_j = search_radius
            for di in range(num_shifts):
                for dj in range(num_shifts):
                    score = np.float32(0.0)
                    for a in range(core):
                        row = window[di + a]
                        template_row = template[a]
                        for b in range(core):
                            diff = row[dj + b] - template_row[b]
                            score += diff * diff
                    scores[di, dj] = score
                    if score < best:
                        best = score
                        best_i = di
                        best_j = dj

            shift_x, shift_y = _subpixel_minimum_3x3(scores, best_i, best_j, num_shifts)
            shift_x += best_j - search_radius
            shift_y += best_i - search_radius

            slopes[i, j] = ref_positions[i, j] + shift_x - unaberrated_slopes[i, j]
            slopes[i + num_regions, j] = (
                ref_positions[i + num_regions, j] + shift_y - unaberrated_slopes[i + num_regions, j]
            )

    return slopes


class SlopesProcess(Component):
    """
    A class to handle real-time slope computation for wavefront sensors.

    Config
    ------
    type : str
        Type of the WFS ("PYWFS" or "SHWFS", case-insensitive).
    signal_type : str
        Type of signal. Only "slopes" (case-insensitive) is supported; any
        other value raises ``ValueError`` at construction.
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
        Contrast for "SHWFS". Default is 0. Pixels at or below
        ``image_noise * contrast`` are ignored by every centroider.
    centroider : str, optional
        SHWFS centroiding algorithm: ``"cog"`` (thresholded centre of
        gravity, default), ``"wcog"`` (Gaussian-weighted CoG) or
        ``"correlation"`` (correlation against a reference image).
    wcog_fwhm : float, optional
        FWHM in pixels of the WCoG Gaussian weight. Default is half the
        sub-aperture size.
    wcog_spot_fwhm : float, optional
        Spot FWHM in pixels used to correct the WCoG gain loss. Default is 0
        (no correction; the interaction matrix absorbs the gain).
    correlation_search_radius : int, optional
        Correlation search half-width in pixels. Default is a quarter of the
        sub-aperture size (at least 1).
    reference_image_file : str, optional
        File holding the SHWFS reference image used by ``"correlation"``
        (templates) and ``"wcog"`` (weight centres). Default is "".
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
    centroider : str
        SHWFS centroiding algorithm.
    reference_image : numpy.ndarray or None
        SHWFS reference image for the ``"wcog"`` and ``"correlation"``
        centroiders.
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

    SUPPORTED_WFS_TYPES = ("SHWFS", "PYWFS")
    SUPPORTED_SIGNAL_TYPES = ("slopes",)

    @property
    def _signal_lock(self) -> threading.RLock:
        """Lock serializing frame processing against geometry/reference changes.

        ``compute_signal`` holds it while it computes and publishes one frame
        (not while it waits for the WFS image), and ``set_pupils`` /
        ``set_ref_slopes`` hold it while they swap buffers and streams, so the
        worker never sees a half-updated configuration. Created on first use
        so instances built with ``__new__`` (tests) work too.
        """

        lock = self.__dict__.get("_signal_lock_obj")
        if lock is None:
            lock = self.__dict__.setdefault("_signal_lock_obj", threading.RLock())
        return lock

    @classmethod
    def normalize_signal_types(cls, conf) -> tuple[str, str]:
        """Return the lower-cased ``(type, signal_type)`` of a slopes config.

        Raises ``ValueError`` for missing or unsupported values, so a bad
        config fails at construction instead of leaving ``compute_signal``
        silently publishing nothing.
        """

        normalized = []
        for key, supported in (
            ("type", cls.SUPPORTED_WFS_TYPES),
            ("signal_type", cls.SUPPORTED_SIGNAL_TYPES),
        ):
            value = conf.get(key) if hasattr(conf, "get") else None
            lowered = value.strip().lower() if isinstance(value, str) else None
            if lowered not in {choice.lower() for choice in supported}:
                raise ValueError(
                    f"slopes: unsupported {key} {value!r}; supported values are "
                    f"{', '.join(supported)} (case-insensitive)"
                )
            normalized.append(lowered)
        return normalized[0], normalized[1]

    def __init__(self, conf) -> None:
        try:
            # Validate before Component.__init__ starts the worker threads.
            wfs_type, signal_type = self.normalize_signal_types(conf)
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

            self.wfs_type = wfs_type
            self.signal_type = signal_type
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
                # set_pupils allocated the PYWFS work buffers and reference slopes.

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
                self._configure_shwfs_centroider()

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
            with self._signal_lock:
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

        The reference is first reset to zero, then the next ``ref_slope_count``
        publications of the ``signal`` stream (all computed with the zero
        reference) are averaged. Each publication is a private snapshot, so
        the worker thread keeps running meanwhile. GPU-backed signal streams
        (torch tensors) are copied to the host.
        """
        component_logger = getattr(self, "logger", logger)
        try:
            count = int(self.ref_slope_count)
            if count < 1:
                raise ValueError("ref_slope_count must be at least 1")
            component_logger.info("Taking reference slopes using %s frames", count)
            # set_ref_slopes waits for any frame in flight, so every
            # publication after this count used the zero reference.
            self.set_ref_slopes(np.zeros_like(self.ref_slopes))
            stream = self._stream_object("signal")
            after = int(stream.count)
            accum = None
            for _ in range(count):
                publication = stream.read_after_publication(after)
                after = int(publication.count)
                frame = _host_array(publication.payload)
                if accum is None:
                    accum = np.zeros(frame.shape, dtype=np.float64)
                accum += frame
            accum /= count
            ref_slopes = self.compute_signal_2d(
                accum.astype(self.signal_dtype),
                out=np.zeros(self.ref_slopes.shape, dtype=self.signal_dtype),
            )
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
            ref_slopes = _host_array(ref_slopes).astype(self.signal_dtype)
            ref_slopes_1d = None
            if self.wfs_type == "pywfs":
                # Build the 1-D reference completely before publishing it, so
                # the worker never reads a partially filled array.
                slopemask = self.valid_sub_aps[:, : self.valid_sub_aps.shape[1] // 2]
                half = ref_slopes.shape[1] // 2
                ref_slopes_1d = np.concatenate(
                    [ref_slopes[:, :half][slopemask], ref_slopes[:, half:][slopemask]]
                ).astype(self.signal_dtype)
            with self._signal_lock:
                self.ref_slopes = ref_slopes
                if ref_slopes_1d is not None:
                    self.ref_slopes_1d = ref_slopes_1d
                self._invalidate_gpu_pywfs_cache(masks=False)
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

    def _invalidate_gpu_pywfs_cache(self, *, masks: bool = True, ref: bool = True) -> None:
        """Drop cached device copies of the PYWFS masks and/or reference slopes."""

        if masks:
            self._gpu_pywfs_masks = None
        if ref:
            self._gpu_pywfs_ref = None

    def _gpu_pywfs_tensors(self):
        """Return device-resident PYWFS masks, slopes buffer and reference slopes.

        Uploading the pupil masks and reference slopes every frame dominated the
        GPU slopes path (#64), so they are uploaded once and rebuilt only when
        their source arrays change. ``compute_pupils_mask`` and
        ``set_ref_slopes`` invalidate the cache explicitly; the cache is also
        keyed on the identity of the source arrays and the device, so a direct
        reassignment of ``p*mask``, ``slopes_arr_1d`` or ``ref_slopes_1d`` is
        picked up too. In-place edits of those arrays must go through the
        setters (or call ``_invalidate_gpu_pywfs_cache``).
        """

        import torch

        device = self.gpu_device
        mask_sources = (self.p1mask, self.p2mask, self.p3mask, self.p4mask, self.slopes_arr_1d)
        masks = getattr(self, "_gpu_pywfs_masks", None)
        if (
            masks is None
            or masks["device"] != device
            or any(a is not b for a, b in zip(masks["sources"], mask_sources))
        ):
            # Integer indices (row-major, same order as boolean masking) avoid a
            # per-frame mask compaction and its host synchronization.
            indices = tuple(
                torch.as_tensor(np.flatnonzero(mask), dtype=torch.long, device=device)
                for mask in mask_sources[:4]
            )
            slopes = torch.as_tensor(self.slopes_arr_1d, dtype=torch.float32, device=device)
            masks = {
                "device": device,
                "sources": mask_sources,
                "indices": indices,
                "slopes": slopes,
            }
            self._gpu_pywfs_masks = masks

        ref = getattr(self, "_gpu_pywfs_ref", None)
        if ref is None or ref["device"] != device or ref["source"] is not self.ref_slopes_1d:
            ref = {
                "device": device,
                "source": self.ref_slopes_1d,
                "tensor": torch.as_tensor(self.ref_slopes_1d, dtype=torch.float32, device=device),
            }
            self._gpu_pywfs_ref = ref

        return masks["indices"], masks["slopes"], ref["tensor"]

    def _compute_slopes_pywfs_gpu(self, image):
        """Run the torch PYWFS kernel on ``gpu_device`` and return a device tensor.

        ``image`` is a torch tensor when the ``wfs`` stream is GPU-backed and a
        NumPy array when the WFS producer created a CPU stream; the latter is
        copied to the device here.
        """

        import torch

        if isinstance(image, np.ndarray):
            image = torch.as_tensor(image, device=self.gpu_device)
        (p1, p2, p3, p4), slopes, ref_slopes = self._gpu_pywfs_tensors()
        return compute_slopes_pywfs_torch(
            image.reshape(-1),
            p1_mask=p1,
            p2_mask=p2,
            p3_mask=p3,
            p4_mask=p4,
            num_pixels_in_pupils=self.num_pixels_in_pupils,
            slopes=slopes,
            ref_slopes=ref_slopes,
        )

    def _configure_shwfs_centroider(self) -> None:
        """Read the SHWFS centroider settings and preallocate its buffers."""

        conf = getattr(self, "conf", None) or {}
        centroider = set_from_config(conf, "centroider", "cog").lower()
        if centroider not in SHWFS_CENTROIDERS:
            raise ValueError(
                f"slopes: unsupported centroider '{centroider}', expected one of "
                f"{', '.join(SHWFS_CENTROIDERS)}"
            )
        self.centroider = centroider
        int_n = int(self.region_size)
        self.wcog_fwhm = set_from_config(conf, "wcog_fwhm", float(int_n) / 2.0)
        if self.wcog_fwhm <= 0:
            raise ValueError(f"slopes: 'wcog_fwhm' must be > 0, got {self.wcog_fwhm}")
        self.wcog_spot_fwhm = set_from_config(conf, "wcog_spot_fwhm", 0.0)
        if self.wcog_spot_fwhm < 0:
            raise ValueError(f"slopes: 'wcog_spot_fwhm' must be >= 0, got {self.wcog_spot_fwhm}")
        self.correlation_search_radius = set_from_config(
            conf, "correlation_search_radius", max(1, int_n // 4)
        )
        self.reference_image_file = set_from_config(conf, "reference_image_file", "")

        self._correlation_radius()

        self._shwfs_coords = shwfs_subaperture_coords(int_n)
        self._shwfs_slopes = np.zeros(self.ref_slopes.shape, dtype=np.float32)
        self._shwfs_centre_positions = np.zeros(self.ref_slopes.shape, dtype=np.float32)
        self._wcog_weights_cache = None
        self._corr_window = np.empty((int_n, int_n), dtype=np.float32)
        self._shwfs_products = None
        self._warned_missing_reference = False
        self.reference_image = None
        if self.reference_image_file:
            self.load_reference_image()

    def _correlation_radius(self) -> int:
        """Return the validated correlation search radius."""

        int_n = int(self.region_size)
        radius = int(self.correlation_search_radius)
        if radius < 1 or int_n - 2 * radius < 2:
            raise ValueError(
                "slopes: 'correlation_search_radius' must be >= 1 and leave a template of "
                f"at least 2 pixels (sub-aperture size {int_n}), got {radius}"
            )
        return radius

    def _wcog_weights(self, centers):
        """Return the per-sub-aperture WCoG weights, rebuilt when centres or FWHM change."""

        fwhm = float(self.wcog_fwhm)
        if fwhm <= 0:
            raise ValueError(f"slopes: 'wcog_fwhm' must be > 0, got {fwhm}")
        cache = self._wcog_weights_cache
        if cache is not None and cache[0] is centers and cache[1] == fwhm:
            return cache[2], cache[3]
        n_reg, int_n = int(self.num_regions), int(self.region_size)
        weights_x = np.empty((n_reg, n_reg, int_n), dtype=np.float32)
        weights_y = np.empty_like(weights_x)
        sigma = fwhm / FWHM_PER_SIGMA
        build_shwfs_wcog_weights_numba(
            self._shwfs_coords,
            centers,
            np.float32(1.0 / (2.0 * sigma * sigma)),
            weights_x,
            weights_y,
        )
        self._wcog_weights_cache = (centers, fwhm, weights_x, weights_y)
        return weights_x, weights_y

    def _shwfs_threshold(self) -> np.float32:
        return np.float32(self.image_noise * self.shwfs_contrast)

    def _shwfs_reference_products(self, threshold):
        """Return the products derived from the reference image, or ``None``.

        The tuple ``(reference_image, threshold, radius, ref_positions,
        templates, template_flux, scores)`` holds the thresholded CoG of each
        reference spot, the correlation templates and a score buffer sized for
        ``radius``. It depends on the reference image, the threshold and the
        search radius, is rebuilt only when one of those changes, and is
        swapped in as a single object so a concurrent update cannot hand the
        kernel mismatched buffers.
        """

        reference_image = self.reference_image
        if reference_image is None:
            return None
        radius = self._correlation_radius()
        threshold = float(threshold)
        products = self._shwfs_products
        if (
            products is not None
            and products[0] is reference_image
            and products[1] == threshold
            and products[2] == radius
        ):
            return products

        int_n = int(self.region_size)
        n_reg = int(self.num_regions)
        spacing = np.float32(self.sub_ap_spacing)
        zeros = np.zeros(self.ref_slopes.shape, dtype=np.float32)
        ref_positions = compute_slopes_shwfs_optim_numba(
            reference_image,
            zeros.copy(),
            zeros,
            np.float32(threshold),
            spacing,
            np.meshgrid(self._shwfs_coords, self._shwfs_coords)[0],
            self.offset_x,
            self.offset_y,
            int_n,
        ).astype(np.float32)
        core = int_n - 2 * radius
        templates = np.zeros((n_reg, n_reg, core, core), dtype=np.float32)
        template_flux = np.zeros((n_reg, n_reg), dtype=np.float32)
        build_shwfs_correlation_templates_numba(
            reference_image,
            np.float32(threshold),
            spacing,
            self.offset_x,
            self.offset_y,
            int_n,
            radius,
            templates,
            template_flux,
        )
        scores = np.empty((2 * radius + 1, 2 * radius + 1), dtype=np.float64)
        products = (
            reference_image,
            threshold,
            radius,
            ref_positions,
            templates,
            template_flux,
            scores,
        )
        self._shwfs_products = products
        return products

    def set_reference_image(self, reference_image):
        """
        Set the SHWFS reference image.

        The ``"correlation"`` centroider cuts one template per sub-aperture
        from it, and the ``"wcog"`` centroider centres each Gaussian weight on
        the thresholded CoG of its reference spot. Take (or load) reference
        slopes afterwards so the residual offsets are removed.

        Parameters
        ----------
        reference_image : numpy.ndarray
            Dark-subtracted WFS frame with the shape of the ``wfs`` stream.
        """
        component_logger = getattr(self, "logger", logger)
        reference_image = np.asarray(reference_image, dtype=np.float32)
        expected = tuple(getattr(self, "image_shape", reference_image.shape))
        if reference_image.shape != expected:
            raise ValueError(
                f"Reference image shape {reference_image.shape} does not match WFS shape {expected}"
            )
        self.reference_image = np.ascontiguousarray(reference_image).copy()
        self._warned_missing_reference = False
        self._shwfs_reference_products(self._shwfs_threshold())
        component_logger.info("Set SHWFS reference image shape=%s", reference_image.shape)

    def take_reference_image(self, count=None):
        """
        Average WFS frames into the SHWFS reference image.

        Parameters
        ----------
        count : int, optional
            Number of frames to average. Defaults to ``ref_slope_count``.
        """
        count = int(self.ref_slope_count if count is None else count)
        if count < 1:
            raise ValueError("count must be at least 1")
        accum = np.zeros(self.image_shape, dtype=np.float64)
        for _ in range(count):
            accum += np.asarray(self.read_image(), dtype=np.float64)
        self.set_reference_image(accum / count)

    def save_reference_image(self, filename=""):
        """
        Save the SHWFS reference image to a file.

        Parameters
        ----------
        filename : str, optional
            Destination. Defaults to the configured ``reference_image_file``.
        """
        if filename == "":
            filename = self.reference_image_file
        if filename == "":
            raise ValueError("No reference image filename provided")
        if self.reference_image is None:
            raise ValueError("No reference image to save")
        np.save(filename, self.reference_image)
        getattr(self, "logger", logger).info("Saved SHWFS reference image to %s", filename)

    def load_reference_image(self, filename=""):
        """
        Load the SHWFS reference image from a file.

        Parameters
        ----------
        filename : str, optional
            Source file. Defaults to the configured ``reference_image_file``.
        """
        if filename == "":
            filename = self.reference_image_file
        if filename == "":
            raise ValueError("No reference image filename provided")
        self.set_reference_image(np.load(filename))

    def _compute_slopes_shwfs(self, image):
        """Run the configured SHWFS centroider and return the 2D slopes array."""

        threshold = self._shwfs_threshold()
        spacing = np.float32(self.sub_ap_spacing)
        int_n = int(self.region_size)
        centroider = getattr(self, "centroider", "cog")
        if centroider == "cog":
            return compute_slopes_shwfs_optim_numba(
                image=image,
                slopes=self._shwfs_slopes,
                unaberrated_slopes=self.ref_slopes,
                threshold=threshold,
                spacing=spacing,
                xvals=self.xvals,
                offset_x=self.offset_x,
                offset_y=self.offset_y,
                int_n=int_n,
            )

        products = self._shwfs_reference_products(threshold)
        if centroider == "wcog":
            centers = self._shwfs_centre_positions if products is None else products[3]
            weights_x, weights_y = self._wcog_weights(centers)
            return compute_slopes_shwfs_wcog_numba(
                image,
                self._shwfs_slopes,
                self.ref_slopes,
                threshold,
                spacing,
                self._shwfs_coords,
                self.offset_x,
                self.offset_y,
                int_n,
                centers,
                weights_x,
                weights_y,
                np.float32(wcog_gain_correction(self.wcog_fwhm, self.wcog_spot_fwhm)),
            )

        if products is None:
            if not self._warned_missing_reference:
                getattr(self, "logger", logger).warning(
                    "Correlation centroider has no reference image; publishing zero slopes "
                    "until one is set (take_reference_image / reference_image_file)"
                )
                self._warned_missing_reference = True
            self._shwfs_slopes.fill(0.0)
            return self._shwfs_slopes
        _, _, radius, ref_positions, templates, template_flux, scores = products
        return compute_slopes_shwfs_correlation_numba(
            image,
            self._shwfs_slopes,
            self.ref_slopes,
            threshold,
            spacing,
            self.offset_x,
            self.offset_y,
            int_n,
            templates,
            template_flux,
            ref_positions,
            radius,
            self._corr_window,
            scores,
        )

    def compute_signal(self):
        """
        Compute the signal from the WFS image.
        """
        image = self.read_stream("wfs", out=self._image_buffer)
        if self.signal_type != "slopes":
            raise ValueError(f"slopes: unsupported signal_type {self.signal_type!r}")
        # Hold the lock only while processing, not while waiting for a frame,
        # so set_pupils / set_ref_slopes never swap buffers mid-frame.
        with self._signal_lock:
            if self.wfs_type == "pywfs":
                if self.gpu_device is not None and gpu_torch_available():
                    slope_signal = self._compute_slopes_pywfs_gpu(image)
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
                slope_signal = self._gather_valid_slopes(self._compute_slopes_shwfs(image))
            if isinstance(slope_signal, np.ndarray):
                signal_host = slope_signal
            else:
                # GPU PYWFS result: write the device tensor to a GPU-backed
                # signal stream, NumPy otherwise; signal_2d is built on the host.
                signal_host = slope_signal.cpu().numpy()
                if getattr(self.signal, "gpu_device", None) is None:
                    slope_signal = signal_host
            self.write_stream("signal", slope_signal)
            self.write_stream("signal_2d", self.compute_signal_2d(signal_host))

        return

    def _gather_valid_slopes(self, slopes):
        """Return ``slopes[valid_sub_aps]`` in a reused buffer (no per-frame allocation).

        The flat indices and output buffer are cached on the identity of
        ``valid_sub_aps``; ``set_valid_sub_aps`` replaces the array, which
        rebuilds them. Edit the mask through ``set_valid_sub_aps``, not in
        place.
        """

        valid = self.valid_sub_aps
        cache = self.__dict__.get("_valid_gather_cache")
        if cache is None or cache[0] is not valid or cache[1] != slopes.shape:
            if valid.shape != slopes.shape:
                raise ValueError(
                    f"valid_sub_aps shape {valid.shape} does not match slopes shape {slopes.shape}"
                )
            indices = np.flatnonzero(valid)
            cache = (valid, slopes.shape, indices, np.empty(indices.size, dtype=np.float32))
            self._valid_gather_cache = cache
        # mode="clip" avoids the temporary that the default mode="raise"
        # buffers through when ``out`` is given; the indices are always valid.
        return np.take(slopes, cache[2], out=cache[3], mode="clip")

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

        Safe to call on a running instance: the signal worker is held off
        (see ``_signal_lock``) while the masks, the PYWFS work buffers, the
        reference slopes and the signal streams are rebuilt, so it never runs
        a frame with a mix of old and new geometry. Reference slopes whose
        shape no longer matches the new geometry are reset to zero.

        Parameters
        ----------
        pupil_locs : list of tuple
            List of pupil locations.
        pupil_radius : int
            Radius of the pupils.
        """
        component_logger = getattr(self, "logger", logger)
        try:
            with self._signal_lock:
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
                    self._allocate_pywfs_buffers()
            component_logger.info("Configured pupils locs=%s radius=%s", pupil_locs, pupil_radius)
        except Exception:
            component_logger.exception(
                "Failed to set pupils locs=%s radius=%s", pupil_locs, pupil_radius
            )
            raise

        return

    def _allocate_pywfs_buffers(self) -> None:
        """(Re)allocate the PYWFS work buffers and reference slopes for the pupil masks.

        Called by ``set_pupils`` with ``_signal_lock`` held. The numba kernel
        does not bounds-check, so the pupil geometry is validated first: the
        four pupils must hold the same number of pixels, matching the signal
        and valid sub-aperture sizes (overlapping or clipped pupils do not).
        """

        counts = [
            int(np.count_nonzero(mask))
            for mask in (self.p1mask, self.p2mask, self.p3mask, self.p4mask)
        ]
        n = counts[0]
        valid_half = int(np.count_nonzero(self.valid_sub_aps)) // 2
        if n == 0 or len(set(counts)) != 1 or int(self.signal_size) != 2 * n or valid_half != n:
            raise ValueError(
                "slopes: inconsistent PYWFS pupil geometry (pixels per pupil "
                f"{counts}, signal size {int(self.signal_size)}, valid sub-apertures "
                f"{2 * valid_half}); check that the pupils do not overlap"
            )
        self.num_pixels_in_pupils = n
        self.p1 = np.empty(n, dtype=self.signal_dtype)
        self.p2 = np.empty_like(self.p1)
        self.p3 = np.empty_like(self.p1)
        self.p4 = np.empty_like(self.p1)
        self.tmp1, self.tmp2 = np.empty_like(self.p1), np.empty_like(self.p1)
        self.slopes_arr_1d = np.zeros(2 * n, dtype=self.signal_dtype)

        ref_slopes = getattr(self, "ref_slopes", None)
        shape = tuple(self.valid_sub_aps.shape)
        if ref_slopes is None or tuple(np.shape(ref_slopes)) != shape:
            if ref_slopes is not None:
                getattr(self, "logger", logger).warning(
                    "Pupil geometry changed the slopes shape to %s; reference slopes reset to "
                    "zero (take or load new ones)",
                    shape,
                )
            ref_slopes = np.zeros(shape, dtype=self.signal_dtype)
        # Rebuilds ref_slopes_1d for the new masks (and drops the GPU cache).
        self.set_ref_slopes(ref_slopes)

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
            self._invalidate_gpu_pywfs_cache(ref=False)
            component_logger.info("Computed pupil masks for %s pupils", len(self.pupil_locs))
        except Exception:
            component_logger.exception("Failed to compute pupil mask")
            raise
        return

    def plot_pupils(self):
        """
        Plot the pupil mask and the masked WFS image to check the pupil setup.

        Returns the figure (not shown); call ``plt.show()`` to display it.
        """
        plt = pyplot()
        fig, (mask_ax, image_ax) = plt.subplots(1, 2, figsize=(12, 5))
        mask_im = mask_ax.imshow(self.pupil_mask, cmap="inferno", origin="lower", aspect="auto")
        fig.colorbar(mask_im, ax=mask_ax)
        mask_ax.set_title("Pupil Mask (Value is Pupil Number)")

        image_im = image_ax.imshow(
            self.pupil_mask * self.read_image(), cmap="inferno", origin="lower", aspect="auto"
        )
        colors = ["g", "b", "orange", "r"]
        for i, (px, py) in enumerate(self.pupil_locs):
            image_ax.axvline(x=px, color=colors[i % len(colors)], alpha=0.6)
            image_ax.axhline(y=py, color=colors[i % len(colors)], alpha=0.6)
        fig.colorbar(image_im, ax=image_ax)
        image_ax.set_title("Pupil Mask * Image")
        return fig

    def compute_signal_2d(self, signal, valid_sub_aps=None, out=None):
        """
        Compute the 2D signal from the valid sub-aperture mask.

        Parameters
        ----------
        signal : numpy.ndarray
            Signal to process.
        valid_sub_aps : numpy.ndarray, optional
            Valid sub-aperture mask. If not provided, uses the current valid sub-aperture mask.
        out : numpy.ndarray, optional
            Destination array with the mask's shape. Defaults to the shared
            ``cur_signal_2d`` buffer that ``compute_signal`` publishes from,
            which the worker thread overwrites every frame; pass a private
            array when calling from another thread.

        Returns
        -------
        numpy.ndarray
            2D signal.
        """
        if valid_sub_aps is None:
            valid_sub_aps = self.valid_sub_aps
        if not isinstance(valid_sub_aps, np.ndarray):
            return -1
        if out is None:
            out = self.cur_signal_2d

        if self.wfs_type == "pywfs":
            slopemask = valid_sub_aps[:, : valid_sub_aps.shape[1] // 2]
            out[:, : valid_sub_aps.shape[1] // 2][slopemask] = signal[: signal.size // 2]
            out[:, valid_sub_aps.shape[1] // 2 :][slopemask] = signal[signal.size // 2 :]
        else:
            out[valid_sub_aps] = signal
        return out


if __name__ == "__main__":
    launch_component(SlopesProcess, "slopes", start=True)
