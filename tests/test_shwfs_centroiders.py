"""Accuracy and behaviour tests for the SHWFS centroiding algorithms (#59)."""

import logging
from pathlib import Path

import numpy as np
import pytest
from scipy.special import erf

from pyrtc import slopes_process as sp
from pyrtc.config_schema import read_system_config, validate_system_config
from pyrtc.utils import ConfigValidationError


SYNTHETIC_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "examples" / "synthetic_shwfs" / "config.yaml"
)

FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))


def _spot_image(num_regions, int_n, shifts, sigma=1.2, flux=1000.0):
    """Pixel-integrated Gaussian spots; ``shifts[i, j] = (dx, dy)`` in pixels."""

    coords = sp.shwfs_subaperture_coords(int_n).astype(np.float64)
    image = np.zeros((num_regions * int_n, num_regions * int_n))
    scale = np.sqrt(2.0) * sigma
    for i in range(num_regions):
        for j in range(num_regions):
            dx, dy = shifts[i, j]
            ex = erf((coords + 0.5 - dx) / scale) - erf((coords - 0.5 - dx) / scale)
            ey = erf((coords + 0.5 - dy) / scale) - erf((coords - 0.5 - dy) / scale)
            block = flux * 0.25 * np.outer(ey, ex)
            image[i * int_n : (i + 1) * int_n, j * int_n : (j + 1) * int_n] = block
    return image


def _scene_image(num_regions, int_n, shifts, seed=3, margin=8, cutoff=0.12):
    """Band-limited extended scene filling each sub-aperture, shifted exactly (Fourier)."""

    rng = np.random.default_rng(seed)
    size = int_n + 2 * margin
    freqs = np.fft.fftfreq(size)
    radius = np.hypot(*np.meshgrid(freqs, freqs))
    spectrum = np.fft.fft2(rng.normal(size=(size, size))) * np.exp(-((radius / cutoff) ** 2))
    base_min = np.real(np.fft.ifft2(spectrum)).min()
    image = np.zeros((num_regions * int_n, num_regions * int_n))
    for i in range(num_regions):
        for j in range(num_regions):
            dx, dy = shifts[i, j]
            phase = np.exp(-2j * np.pi * (freqs[None, :] * dx + freqs[:, None] * dy))
            field = np.real(np.fft.ifft2(spectrum * phase)) - base_min + 0.1
            block = field[margin : margin + int_n, margin : margin + int_n]
            image[i * int_n : (i + 1) * int_n, j * int_n : (j + 1) * int_n] = block
    return image


def _truth(shifts):
    return np.concatenate([shifts[..., 0], shifts[..., 1]], axis=0)


def _xvals(int_n):
    coords = sp.shwfs_subaperture_coords(int_n)
    return np.meshgrid(coords, coords)[0]


def _cog(image, int_n, num_regions, threshold=0.0, ref=None, slopes=None):
    zeros = np.zeros((2 * num_regions, num_regions), dtype=np.float32)
    ref = zeros if ref is None else ref
    return sp.compute_slopes_shwfs_optim_numba(
        image.astype(np.float32),
        zeros.copy() if slopes is None else slopes,
        ref,
        np.float32(threshold),
        np.float32(int_n),
        _xvals(int_n),
        0,
        0,
        int_n,
    )


def _wcog(
    image,
    int_n,
    num_regions,
    *,
    fwhm=4.0,
    gain=1.0,
    centers=None,
    threshold=0.0,
    ref=None,
    slopes=None,
):
    zeros = np.zeros((2 * num_regions, num_regions), dtype=np.float32)
    centers = zeros if centers is None else centers
    sigma = fwhm / FWHM
    weights_x = np.empty((num_regions, num_regions, int_n), dtype=np.float32)
    weights_y = np.empty_like(weights_x)
    sp.build_shwfs_wcog_weights_numba(
        sp.shwfs_subaperture_coords(int_n),
        centers,
        np.float32(1.0 / (2.0 * sigma * sigma)),
        weights_x,
        weights_y,
    )
    return sp.compute_slopes_shwfs_wcog_numba(
        image,
        zeros.copy() if slopes is None else slopes,
        zeros if ref is None else ref,
        np.float32(threshold),
        np.float32(int_n),
        sp.shwfs_subaperture_coords(int_n),
        0,
        0,
        int_n,
        centers,
        weights_x,
        weights_y,
        np.float32(gain),
    )


def _correlation(
    image, reference, int_n, num_regions, *, radius=3, threshold=0.0, ref=None, slopes=None
):
    zeros = np.zeros((2 * num_regions, num_regions), dtype=np.float32)
    core = int_n - 2 * radius
    templates = np.zeros((num_regions, num_regions, core, core), dtype=np.float32)
    flux = np.zeros((num_regions, num_regions), dtype=np.float32)
    sp.build_shwfs_correlation_templates_numba(
        reference, np.float32(threshold), np.float32(int_n), 0, 0, int_n, radius, templates, flux
    )
    ref_positions = _cog(reference, int_n, num_regions, threshold)
    return sp.compute_slopes_shwfs_correlation_numba(
        image,
        zeros.copy() if slopes is None else slopes,
        zeros if ref is None else ref,
        np.float32(threshold),
        np.float32(int_n),
        0,
        0,
        int_n,
        templates,
        flux,
        ref_positions,
        radius,
        np.empty((int_n, int_n), dtype=np.float32),
        np.empty((2 * radius + 1, 2 * radius + 1), dtype=np.float64),
    )


# --- kernel accuracy -------------------------------------------------------


N_REG, INT_N, SIGMA = 4, 12, 1.2
# Pixel-integrated Gaussian: effective variance sigma^2 + 1/12.
SPOT_FWHM = FWHM * np.sqrt(SIGMA**2 + 1.0 / 12.0)


@pytest.fixture(scope="module")
def known_shifts():
    return np.random.default_rng(0).uniform(-1.5, 1.5, (N_REG, N_REG, 2))


def test_cog_recovers_subpixel_shifts(known_shifts):
    image = _spot_image(N_REG, INT_N, known_shifts, SIGMA)
    np.testing.assert_allclose(_cog(image, INT_N, N_REG), _truth(known_shifts), atol=5e-3)


@pytest.mark.parametrize("weight_fwhm", [3.0, 4.0, 6.0])
def test_wcog_with_gain_correction_recovers_subpixel_shifts(known_shifts, weight_fwhm):
    image = _spot_image(N_REG, INT_N, known_shifts, SIGMA)
    gain = sp.wcog_gain_correction(weight_fwhm, SPOT_FWHM)
    out = _wcog(image, INT_N, N_REG, fwhm=weight_fwhm, gain=gain)
    np.testing.assert_allclose(out, _truth(known_shifts), atol=5e-3)


def test_wcog_uncorrected_gain_matches_gaussian_theory():
    shifts = np.full((N_REG, N_REG, 2), 0.2)
    image = _spot_image(N_REG, INT_N, shifts, SIGMA)
    weight_fwhm = 4.0
    out = _wcog(image, INT_N, N_REG, fwhm=weight_fwhm, gain=1.0)
    expected_gain = 1.0 / sp.wcog_gain_correction(weight_fwhm, SPOT_FWHM)
    np.testing.assert_allclose(out / 0.2, expected_gain, rtol=0.02)
    assert expected_gain < 0.7  # the bias is real and large for a narrow weight


def test_wcog_weight_centred_on_reference_position():
    # Spots at an off-centre reference; the weight follows the reference so a
    # further shift is measured without the off-centre truncation bias.
    ref_shift = np.full((N_REG, N_REG, 2), 1.0)
    extra = np.random.default_rng(1).uniform(-0.5, 0.5, (N_REG, N_REG, 2))
    reference = _spot_image(N_REG, INT_N, ref_shift, SIGMA)
    image = _spot_image(N_REG, INT_N, ref_shift + extra, SIGMA)
    centers = _cog(reference, INT_N, N_REG)
    gain = sp.wcog_gain_correction(4.0, SPOT_FWHM)
    out = _wcog(image, INT_N, N_REG, fwhm=4.0, gain=gain, centers=centers)
    np.testing.assert_allclose(out, _truth(ref_shift + extra), atol=5e-3)


def test_correlation_recovers_subpixel_shifts(known_shifts):
    reference = _spot_image(N_REG, INT_N, np.zeros_like(known_shifts), SIGMA)
    image = _spot_image(N_REG, INT_N, known_shifts, SIGMA)
    out = _correlation(image, reference, INT_N, N_REG, radius=3)
    np.testing.assert_allclose(out, _truth(known_shifts), atol=0.05)


def test_accuracy_vs_noise_wcog_and_correlation_beat_cog():
    rng = np.random.default_rng(2)
    reference = _spot_image(N_REG, INT_N, np.zeros((N_REG, N_REG, 2)), SIGMA)
    gain = sp.wcog_gain_correction(4.0, SPOT_FWHM)
    rms = {}
    for noise in (0.5, 5.0):
        errs = {"cog": [], "wcog": [], "correlation": []}
        for _ in range(10):
            shifts = rng.uniform(-1.0, 1.0, (N_REG, N_REG, 2))
            truth = _truth(shifts)
            image = _spot_image(N_REG, INT_N, shifts, SIGMA) + rng.normal(0, noise, (48, 48))
            errs["cog"].append(_cog(image, INT_N, N_REG) - truth)
            errs["wcog"].append(_wcog(image, INT_N, N_REG, fwhm=4.0, gain=gain) - truth)
            errs["correlation"].append(_correlation(image, reference, INT_N, N_REG) - truth)
        rms[noise] = {name: float(np.sqrt(np.mean(np.square(e)))) for name, e in errs.items()}

    for method in ("cog", "wcog", "correlation"):
        assert rms[0.5][method] < rms[5.0][method]
    # Measured: cog ~0.21 px, wcog ~0.05 px, correlation ~0.04 px at noise 5.
    assert rms[5.0]["wcog"] < 0.5 * rms[5.0]["cog"]
    assert rms[5.0]["correlation"] < 0.5 * rms[5.0]["cog"]
    assert rms[0.5]["wcog"] < 0.02
    assert rms[0.5]["correlation"] < 0.04


def test_extended_source_biases_cog_but_not_correlation():
    num_regions, int_n = 3, 16
    shifts = np.zeros((num_regions, num_regions, 2))
    shifts[..., 0], shifts[..., 1] = 1.3, -0.7
    reference = _scene_image(num_regions, int_n, np.zeros_like(shifts))
    image = _scene_image(num_regions, int_n, shifts)
    truth = _truth(shifts)

    # Reference slopes taken on the reference frame, as in operation.
    ref_slopes = _cog(reference, int_n, num_regions)
    cog = _cog(image, int_n, num_regions, ref=ref_slopes)
    corr = _correlation(image, reference, int_n, num_regions, radius=3, ref=ref_slopes)

    # The truncated scene barely moves its centre of gravity.
    assert np.max(np.abs(cog - truth)) > 0.5
    np.testing.assert_allclose(corr, truth, atol=0.06)


# --- kernel edge cases -----------------------------------------------------


def test_no_flux_and_sub_threshold_sub_apertures_give_zero_slopes():
    int_n, num_regions = 8, 2
    shifts = np.full((num_regions, num_regions, 2), 0.3)
    reference = _spot_image(num_regions, int_n, np.zeros_like(shifts))
    image = _spot_image(num_regions, int_n, shifts)
    image[:int_n, :int_n] = 0.0  # dark sub-aperture (0, 0)
    image[:int_n, int_n:] = 2.0  # sub-aperture (0, 1): flux only below threshold
    ref = np.ones((2 * num_regions, num_regions), dtype=np.float32)
    dirty = np.full_like(ref, 99.0)

    cog = _cog(image, int_n, num_regions, threshold=3.0, ref=ref, slopes=dirty.copy())
    wcog = _wcog(image, int_n, num_regions, threshold=3.0, ref=ref, slopes=dirty.copy())
    corr = _correlation(
        image, reference, int_n, num_regions, radius=2, threshold=3.0, ref=ref, slopes=dirty.copy()
    )
    for out in (cog, wcog, corr):
        assert out[0, 0] == 0.0 and out[num_regions, 0] == 0.0
        assert out[0, 1] == 0.0 and out[num_regions, 1] == 0.0
        assert np.all(np.isfinite(out))
        assert out[1, 0] != 0.0

    # Reference sub-aperture without flux: correlation has no template -> 0.
    dark_reference = reference.copy()
    dark_reference[int_n:, :int_n] = 0.0
    corr = _correlation(image, dark_reference, int_n, num_regions, radius=2, slopes=dirty.copy())
    assert corr[1, 0] == 0.0 and corr[num_regions + 1, 0] == 0.0


def test_out_of_bounds_sub_apertures_are_zeroed():
    int_n = 8
    image = _spot_image(2, int_n, np.full((2, 2, 2), 0.2))[:, : int_n + 4]
    dirty = np.full((4, 2), 7.0, dtype=np.float32)
    out = _cog(image, int_n, 2, slopes=dirty.copy())
    assert np.all(out[:, 1] == 0.0) and np.all(out[:, 0] != 0.0)
    out = _wcog(image, int_n, 2, slopes=dirty.copy())
    assert np.all(out[:, 1] == 0.0)
    out = _correlation(image, image, int_n, 2, radius=2, slopes=dirty.copy())
    assert np.all(out[:, 1] == 0.0)


def test_correlation_clamps_shifts_beyond_search_window():
    int_n, radius = 12, 2
    reference = _spot_image(1, int_n, np.zeros((1, 1, 2)))
    image = _spot_image(1, int_n, np.array([[[3.4, 0.0]]]))
    out = _correlation(image, reference, int_n, 1, radius=radius)
    assert out[0, 0] == pytest.approx(radius, abs=1e-3)
    assert abs(out[1, 0]) < 0.05


def test_wcog_gain_correction_values():
    assert sp.wcog_gain_correction(4.0, 0.0) == 1.0
    assert sp.wcog_gain_correction(4.0, 4.0) == pytest.approx(2.0)
    with pytest.raises(ValueError):
        sp.wcog_gain_correction(0.0, 1.0)


# --- SlopesProcess wiring --------------------------------------------------


def _shwfs_process(conf_extra=None, *, int_n=12, num_regions=4):
    proc = sp.SlopesProcess.__new__(sp.SlopesProcess)
    proc.conf = {"type": "SHWFS", "signal_type": "slopes", **(conf_extra or {})}
    proc.signal_type = "slopes"
    proc.wfs_type = "shwfs"
    proc.signal_dtype = np.float32
    proc.image_shape = (int_n * num_regions, int_n * num_regions)
    proc.image_noise = 0.0
    proc.shwfs_contrast = 0.0
    proc.sub_ap_spacing = float(int_n)
    proc.region_size = int_n
    proc.num_regions = num_regions
    proc.offset_x = 0
    proc.offset_y = 0
    proc.xvals = _xvals(int_n)
    proc.valid_sub_aps = np.ones((2 * num_regions, num_regions), dtype=bool)
    proc.cur_signal_2d = np.zeros(proc.valid_sub_aps.shape)
    proc.ref_slopes = np.zeros(proc.valid_sub_aps.shape, dtype=np.float32)
    proc._image_buffer = None
    proc.written = {}
    proc.write_stream = lambda name, value: proc.written.__setitem__(name, value)
    proc._configure_shwfs_centroider()
    return proc


def _signal(proc, image):
    proc.read_stream = lambda name, out=None, **kwargs: image
    proc.compute_signal()
    return proc.written["signal_2d"].copy()


def test_default_centroider_is_cog_and_unchanged(known_shifts):
    proc = _shwfs_process()
    assert proc.centroider == "cog"
    image = _spot_image(N_REG, INT_N, known_shifts).astype(np.int32)
    np.testing.assert_allclose(_signal(proc, image), _cog(image, INT_N, N_REG), atol=1e-6)


def test_cog_kernel_matches_float_copy_for_integer_images(known_shifts):
    """The CoG kernel converts pixels itself; results equal the old float copy."""

    image = _spot_image(N_REG, INT_N, known_shifts) * 3.0
    for dtype in (np.uint16, np.int32, np.float32, np.float64):
        raw = image.astype(dtype)
        zeros = np.zeros((2 * N_REG, N_REG), dtype=np.float32)
        direct = sp.compute_slopes_shwfs_optim_numba(
            raw, zeros.copy(), zeros, np.float32(2.0), np.float32(INT_N), _xvals(INT_N), 0, 0, INT_N
        )
        np.testing.assert_array_equal(direct, _cog(raw, INT_N, N_REG, threshold=2.0))


def test_cog_compute_signal_reuses_buffers(known_shifts):
    proc = _shwfs_process()
    proc.valid_sub_aps[0, 0] = False
    proc.set_valid_sub_aps(proc.valid_sub_aps.copy())
    image = _spot_image(N_REG, INT_N, known_shifts).astype(np.int32)
    image[:INT_N, INT_N : 2 * INT_N] = 0  # dark sub-aperture (0, 1)
    expected = _cog(image, INT_N, N_REG)

    first = _signal(proc, image)
    signal = proc.written["signal"]
    buffer = proc._shwfs_slopes
    np.testing.assert_allclose(first, expected * proc.valid_sub_aps, atol=1e-6)
    assert first[0, 1] == 0.0 and first[N_REG, 1] == 0.0
    np.testing.assert_array_equal(signal, expected[proc.valid_sub_aps])

    # A brighter frame then the same frame again: no stale values survive.
    _signal(proc, image * 2)
    again = _signal(proc, image)
    np.testing.assert_array_equal(again, first)
    assert proc._shwfs_slopes is buffer
    assert proc.written["signal"] is signal
    np.testing.assert_array_equal(signal, expected[proc.valid_sub_aps])

    # A new valid-sub-aperture mask rebuilds the gather buffer.
    proc.set_valid_sub_aps(np.ones_like(proc.valid_sub_aps))
    _signal(proc, image)
    assert proc.written["signal"].shape == (expected.size,)
    np.testing.assert_array_equal(proc.written["signal"], expected.ravel())


def test_wcog_centroider_through_compute_signal(known_shifts):
    proc = _shwfs_process({"centroider": "WCoG", "wcog_fwhm": 4.0, "wcog_spot_fwhm": SPOT_FWHM})
    assert proc.centroider == "wcog"
    image = _spot_image(N_REG, INT_N, known_shifts)
    np.testing.assert_allclose(_signal(proc, image), _truth(known_shifts), atol=5e-3)

    # Changing the FWHM at runtime rebuilds the cached weights.
    centres = proc._shwfs_centre_positions
    weights = proc._wcog_weights(centres)[0]
    assert proc._wcog_weights(centres)[0] is weights
    proc.wcog_fwhm = 5.0
    assert proc._wcog_weights(centres)[0] is not weights

    # A reference image moves the weight centres onto the reference spots.
    ref_shift = np.full_like(known_shifts, 0.8)
    proc.set_reference_image(_spot_image(N_REG, INT_N, ref_shift))
    ref_positions = proc._shwfs_products[3]
    np.testing.assert_allclose(ref_positions, _truth(ref_shift), atol=5e-3)
    image = _spot_image(N_REG, INT_N, ref_shift + 0.3)
    np.testing.assert_allclose(_signal(proc, image), _truth(ref_shift + 0.3), atol=5e-3)


def test_correlation_centroider_reference_lifecycle(tmp_path, known_shifts, caplog):
    proc = _shwfs_process({"centroider": "correlation", "correlation_search_radius": 3})
    image = _spot_image(N_REG, INT_N, known_shifts)

    with caplog.at_level(logging.WARNING):
        assert np.all(_signal(proc, image) == 0.0)
    assert "no reference image" in caplog.text

    reference = _spot_image(N_REG, INT_N, np.zeros_like(known_shifts))
    frames = iter([reference, reference])
    proc.read_image = lambda block=True: next(frames)
    proc.take_reference_image(count=2)
    np.testing.assert_allclose(_signal(proc, image), _truth(known_shifts), atol=0.05)

    path = tmp_path / "shwfs_reference.npy"
    proc.save_reference_image(str(path))
    other = _shwfs_process(
        {
            "centroider": "correlation",
            "correlation_search_radius": 3,
            "reference_image_file": str(path),
        }
    )
    np.testing.assert_allclose(_signal(other, image), _truth(known_shifts), atol=0.05)

    # Threshold and search-radius changes rebuild the derived templates.
    products = proc._shwfs_products
    _signal(proc, image)
    assert proc._shwfs_products is products  # cached while nothing changes
    proc.shwfs_contrast, proc.image_noise = 1.0, 1.0
    proc.correlation_search_radius = 2
    np.testing.assert_allclose(_signal(proc, image), _truth(known_shifts), atol=0.06)
    _, threshold, radius, _, templates, _, scores = proc._shwfs_products
    assert (threshold, radius) == (1.0, 2)
    assert templates.shape[-1] == INT_N - 4
    assert scores.shape == (5, 5)

    # An invalid runtime radius is rejected before any kernel runs.
    proc.correlation_search_radius = 6
    with pytest.raises(ValueError):
        _signal(proc, image)


def test_reference_image_errors():
    proc = _shwfs_process({"centroider": "correlation"})
    with pytest.raises(ValueError, match="shape"):
        proc.set_reference_image(np.zeros((3, 3)))
    with pytest.raises(ValueError):
        proc.save_reference_image()
    with pytest.raises(ValueError):
        proc.load_reference_image()


@pytest.mark.parametrize(
    "conf",
    [
        {"centroider": "median"},
        {"centroider": "wcog", "wcog_fwhm": 0.0},
        {"centroider": "wcog", "wcog_spot_fwhm": -1.0},
        {"centroider": "correlation", "correlation_search_radius": 0},
        {"centroider": "correlation", "correlation_search_radius": 6},
    ],
)
def test_invalid_centroider_settings_raise(conf):
    with pytest.raises(ValueError):
        _shwfs_process(conf)


@pytest.mark.parametrize(
    "extra, ok",
    [
        ({"centroider": "correlation", "correlation_search_radius": 2}, True),
        ({"centroider": "wcog", "wcog_fwhm": 3.5, "wcog_spot_fwhm": 2.0}, True),
        ({"centroider": "median"}, False),
        ({"centroider": 3}, False),
        ({"wcog_fwhm": -1.0}, False),
        ({"wcog_spot_fwhm": -1.0}, False),
        ({"correlation_search_radius": 0}, False),
        ({"reference_image_file": 5}, False),
    ],
)
def test_config_schema_validates_centroider_fields(extra, ok):
    conf = read_system_config(SYNTHETIC_CONFIG_PATH, validate=False)
    conf["slopes"].update(extra)
    if ok:
        validate_system_config(conf, config_path=SYNTHETIC_CONFIG_PATH)
    else:
        with pytest.raises(ConfigValidationError):
            validate_system_config(conf, config_path=SYNTHETIC_CONFIG_PATH)
