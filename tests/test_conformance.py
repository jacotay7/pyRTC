"""pyrtc follows the AO stack conventions (aocore CONVENTIONS.md).

pyrtc's slope vectors use the *blocked* layout ``[sx_1 .. sx_N, sy_1 .. sy_N]``
(CONVENTIONS 7.1), and its image streams are ``(height, width)`` = ``[y, x]``
(CONVENTIONS 1.1, #162). The Shack-Hartmann images here are formed with aocore's
reference Fraunhofer propagator, so the sign checks test pyrtc's slope code
against the stack definition rather than against a local model.
"""

import uuid

import numpy as np
import pytest
from aocore import FocalPlanePropagator, conformance
from aocore import centroid as aocore_centroid

from pyrtc import slopes_process as sp
from pyrtc.modal_basis import actuator_positions_from_layout, generate_modes, parse_basis_config
from pyrtc.streams import clear_shms
from pyrtc.utils import centroid
from pyrtc.wavefront_sensor import WavefrontSensor
from testsupport import private_stream

# A 64 x 64 OPD grid over a 1 m square pupil, 8 x 8 lenslets of 8 pixels, and
# Nyquist-sampled spots (2 detector pixels per lambda / d).
N_PUPIL, N_LENSLETS, WAVELENGTH = 64, 8, 500e-9
PITCH = 1.0 / N_PUPIL
SUB_PIXELS = N_PUPIL // N_LENSLETS
SPOT_PIXEL_SCALE = WAVELENGTH / (2 * SUB_PIXELS * PITCH)
GRADIENT = 0.5 * SPOT_PIXEL_SCALE  # a ramp that moves every spot by half a pixel


def _shwfs_image(opd):
    """Shack-Hartmann image ``[y, x]`` of an OPD map in metres (CONVENTIONS 1.1, 3.1)."""

    m, n = SUB_PIXELS, N_LENSLETS
    blocks = np.asarray(opd, dtype=np.float64).reshape(n, m, n, m).transpose(0, 2, 1, 3)
    field = np.exp(2j * np.pi * blocks / WAVELENGTH)[:, :, None]
    propagator = FocalPlanePropagator((m, m), PITCH, WAVELENGTH, SPOT_PIXEL_SCALE, (m, m))
    spots = np.abs(propagator.forward(field)[:, :, 0]) ** 2
    return spots.transpose(0, 2, 1, 3).reshape(n * m, n * m).astype(np.float32)


@pytest.fixture
def shwfs_slopes():
    """``slopes(image)`` from a real SHWFS ``SlopesProcess`` on private streams.

    The reference slopes are those of a flat wavefront, as after
    ``take_ref_slopes`` on a calibrated system.
    """

    wfs = private_stream("wfs", (N_PUPIL, N_PUPIL), np.float32)
    suffix = uuid.uuid4().hex[:8]
    outputs = {"signal": f"sig_{suffix}", "signal_2d": f"sig2d_{suffix}"}
    proc = sp.SlopesProcess(
        {
            "type": "SHWFS",
            "signal_type": "slopes",
            "sub_ap_spacing": SUB_PIXELS,
            "sub_ap_offset_x": 0,
            "sub_ap_offset_y": 0,
            "functions": ["compute_signal"],
            "input_streams": {"wfs": wfs.name},
            "output_streams": outputs,
        }
    )

    def slopes(image):
        wfs.write(image)
        proc.compute_signal()
        return np.array(proc.read(block=False), dtype=np.float64)

    try:
        flat = slopes(_shwfs_image(np.zeros((N_PUPIL, N_PUPIL))))
        proc.set_ref_slopes(proc.compute_signal_2d(flat.astype(np.float32)))
        yield slopes
    finally:
        proc.close()
        clear_shms(list(outputs.values()))


def test_shwfs_slopes_are_blocked_and_positive_along_the_ramp(shwfs_slopes):
    """Rules 7.1 and 7.2 on pyrtc's own SHWFS slope path, in the array frame ``[y, x]``."""

    result = conformance.check_slope_sign(
        lambda opd: shwfs_slopes(_shwfs_image(opd)),
        pupil_shape=(N_PUPIL, N_PUPIL),
        pitch=PITCH,
        gradient=GRADIENT,
        n_subapertures=N_LENSLETS**2,
        layout="blocked",
    )
    # Slopes are in pixels: the ramp moves every spot by half a pixel.
    assert result["x_ramp_mean_sx"] == pytest.approx(0.5, rel=0.05)
    assert result["y_ramp_mean_sy"] == pytest.approx(0.5, rel=0.05)


# The camera frame is wider than the lenslet array, so a swapped axis cannot
# go unnoticed: (height, width) = (64, 80), lenslets in the first 64 columns.
CAMERA_SHAPE = (N_PUPIL, N_PUPIL + 16)


@pytest.fixture
def camera_slopes():
    """``slopes(image)`` for a camera frame published through a ``WavefrontSensor``.

    The frame is what a camera SDK returns, ``(height, width)`` = ``[y, x]``
    (#162). It goes through the sensor's ``wfs`` stream, as an adapter's
    ``expose`` publishes it, into a SHWFS ``SlopesProcess``; the reference
    slopes are those of a flat wavefront.
    """

    suffix = uuid.uuid4().hex[:8]
    outputs = {
        "wfs_raw": f"raw_{suffix}",
        "wfs": f"wfs_{suffix}",
        "signal": f"sig_{suffix}",
        "signal_2d": f"sig2d_{suffix}",
    }
    height, width = CAMERA_SHAPE
    wfs = WavefrontSensor(
        {
            "name": "wfs",
            "width": width,
            "height": height,
            "functions": [],
            "output_streams": {"wfs_raw": outputs["wfs_raw"], "wfs": outputs["wfs"]},
        }
    )
    proc = None
    try:
        assert tuple(wfs._stream_object("wfs").shape) == CAMERA_SHAPE
        proc = sp.SlopesProcess(
            {
                "type": "SHWFS",
                "signal_type": "slopes",
                "sub_ap_spacing": SUB_PIXELS,
                "sub_ap_offset_x": 0,
                "sub_ap_offset_y": 0,
                "functions": [],
                "input_streams": {"wfs": outputs["wfs"]},
                "output_streams": {"signal": outputs["signal"], "signal_2d": outputs["signal_2d"]},
            }
        )
        scale = 60000.0 / float(np.max(_shwfs_image(np.zeros((N_PUPIL, N_PUPIL)))))

        def slopes(image):
            frame = np.zeros(CAMERA_SHAPE, dtype=np.uint16)
            frame[:, :N_PUPIL] = np.rint(np.clip(image * scale, 0, 65535))
            wfs.data = frame
            wfs.expose()
            proc.compute_signal()
            return np.array(proc.read(block=False), dtype=np.float64)

        flat = slopes(_shwfs_image(np.zeros((N_PUPIL, N_PUPIL))))
        proc.set_ref_slopes(proc.compute_signal_2d(flat.astype(np.float32)))
        yield slopes
    finally:
        if proc is not None:
            proc.close()
        wfs.close()
        clear_shms(list(outputs.values()))


def test_shwfs_slopes_follow_camera_axes(camera_slopes):
    """Rules 1.1 and 7.1 for a camera frame published through a WFS stream (#162).

    The image streams are ``(height, width)``, adapters publish frames as the
    camera returns them, and the x slopes follow the camera's columns.
    """

    result = conformance.check_slope_sign(
        lambda opd: camera_slopes(_shwfs_image(opd)),
        pupil_shape=(N_PUPIL, N_PUPIL),
        pitch=PITCH,
        gradient=GRADIENT,
        n_subapertures=N_LENSLETS**2,
        layout="blocked",
    )
    assert result["x_ramp_mean_sx"] == pytest.approx(0.5, rel=0.05)
    assert result["y_ramp_mean_sy"] == pytest.approx(0.5, rel=0.05)


def test_shwfs_subaperture_coordinates_are_pixel_centres():
    """Rule 1.2 for the coordinates every SHWFS centroider measures spots in (#163)."""

    conformance.check_coordinates(lambda n, pitch: sp.shwfs_subaperture_coords(n) * pitch)


@pytest.mark.parametrize("centroider", ["cog", "wcog", "correlation"])
@pytest.mark.parametrize("sub_pixels", [7, 8])
def test_centred_spot_reads_zero_raw_slopes(centroider, sub_pixels):
    """#163: a spot on a sub-aperture's optical axis reads 0 with zero reference slopes.

    For an even sub-aperture the spot sits between the two middle pixels;
    pyrtc 1.x read -0.5 px there.
    """

    n_sub = 4
    shape = (n_sub * sub_pixels, n_sub * sub_pixels)
    rows, cols = np.indices(shape, dtype=np.float64)
    centre = (sub_pixels - 1) / 2.0
    spots = np.exp(
        -(((rows % sub_pixels) - centre) ** 2 + ((cols % sub_pixels) - centre) ** 2) / 1.5
    )
    image = np.rint(1000.0 * spots).astype(np.int32)
    wfs = private_stream("wfs", shape, np.int32)
    suffix = uuid.uuid4().hex[:8]
    outputs = {"signal": f"sig_{suffix}", "signal_2d": f"sig2d_{suffix}"}
    proc = sp.SlopesProcess(
        {
            "type": "SHWFS",
            "signal_type": "slopes",
            "sub_ap_spacing": sub_pixels,
            "sub_ap_offset_x": 0,
            "sub_ap_offset_y": 0,
            "centroider": centroider,
            "correlation_search_radius": 1,
            "functions": [],
            "input_streams": {"wfs": wfs.name},
            "output_streams": outputs,
        }
    )
    try:
        wfs.write(image)
        if centroider == "correlation":
            proc.set_reference_image(image)
        proc.compute_signal()
        np.testing.assert_allclose(proc.read(block=False), 0.0, atol=1e-5)
    finally:
        proc.close()
        clear_shms(list(outputs.values()))


def test_actuator_positions_from_layout_are_pixel_centres():
    """Rule 1.2 along x (columns) and y (rows) for actuator positions from a layout."""

    def along(axis):
        def coordinates(n, pitch):
            shape = (1, n) if axis == 0 else (n, 1)
            # The layout's outer actuators lie on the pupil edge.
            positions = actuator_positions_from_layout(np.ones(shape), pitch * (n - 1))
            return positions[:, axis]

        return coordinates

    conformance.check_coordinates(along(0))
    conformance.check_coordinates(along(1))
    positions = actuator_positions_from_layout(np.ones((3, 3)), 2.0)
    np.testing.assert_allclose(positions[1], [0.0, -1.0])  # row 0, column 1: -y


def test_centroid_is_aocore_centroid_in_pixel_indices():
    """``pyrtc.utils.centroid`` returns ``[x, y]`` from pixel (0, 0); aocore ``(y, x)`` from the centre."""

    image = np.random.default_rng(0).uniform(1.0, 2.0, (7, 10))
    image[2, 8] += 50.0
    cy, cx = aocore_centroid(image)
    np.testing.assert_allclose(
        centroid(image), [cx + (image.shape[1] - 1) / 2, cy + (image.shape[0] - 1) / 2], rtol=1e-5
    )


def test_zernike_modes_are_noll_with_tip_along_x():
    """Rules 5.1 and 5.2 for the Zernike modes pyrtc requests from aobasis."""

    basis = parse_basis_config({"type": "zernike", "ignore_piston": False})

    def zernike(j, y, x):
        positions = np.column_stack((x, y))
        modes = generate_modes(basis, positions, j, pupil_diameter=2.0)
        return modes[:, j - 1]

    conformance.check_zernike_basis(zernike)
