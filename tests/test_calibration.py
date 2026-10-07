"""Versioned calibration files and the conversion of pyrtc 1.x files (#162, #163).

The 1.x pipeline is emulated with the 2.0 code: a 1.x adapter that
transposed frames ("xy") published ``frame.T``, and 1.x SHWFS raw slopes of
an even sub-aperture read 0.5 px less than 2.0's (``k - n // 2`` against
``k - (n - 1) / 2``). A converted 1.x calibration must then behave exactly as
one measured with 2.0.
"""

import json
import uuid

import numpy as np
import pytest

from pyrtc import calibration as cal
from pyrtc.loop import Loop
from pyrtc.science_camera import ScienceCamera
from pyrtc.slopes_process import SlopesProcess
from pyrtc.streams import clear_shms
from pyrtc.wavefront_sensor import WavefrontSensor
from testsupport import bare_component, private_stream

SUB = 8
N_SUB = 4
SIDE = SUB * N_SUB


def _spot_image(seed, sub=SUB, n_sub=N_SUB, max_shift=1.5):
    """A Shack-Hartmann image ``[y, x]`` with a randomly displaced spot in every sub-aperture."""

    rng = np.random.default_rng(seed)
    side = sub * n_sub
    rows, cols = np.indices((side, side), dtype=np.float64)
    shifts = rng.uniform(-max_shift, max_shift, (2, n_sub, n_sub))
    centre = (sub - 1) / 2.0
    dx = (cols % sub) - centre - np.repeat(np.repeat(shifts[0], sub, 0), sub, 1)
    dy = (rows % sub) - centre - np.repeat(np.repeat(shifts[1], sub, 0), sub, 1)
    return np.rint(20.0 + 4000.0 * np.exp(-(dx**2 + dy**2) / 2.0)).astype(np.int32)


@pytest.fixture
def shwfs_factory():
    """Build SHWFS ``SlopesProcess`` objects on private streams; closes them afterwards."""

    built = []

    def build(shape=(SIDE, SIDE), sub=SUB, **conf):
        wfs = private_stream("wfs", shape, np.int32)
        wfs.write(np.zeros(shape, dtype=np.int32))
        suffix = uuid.uuid4().hex[:8]
        outputs = {"signal": f"sig_{suffix}", "signal_2d": f"sig2d_{suffix}"}
        proc = SlopesProcess(
            {
                "type": "SHWFS",
                "signal_type": "slopes",
                "sub_ap_spacing": sub,
                "sub_ap_offset_x": 0,
                "sub_ap_offset_y": 0,
                "image_noise": 1.0,
                "contrast": 30.0,
                "functions": [],
                "input_streams": {"wfs": wfs.name},
                "output_streams": outputs,
                **conf,
            }
        )
        built.append((proc, outputs))

        def slopes(image, two_d=True):
            wfs.write(np.ascontiguousarray(image, dtype=np.int32))
            proc.compute_signal()
            signal = np.array(proc.read(block=False), dtype=np.float32)
            if two_d:
                return proc.compute_signal_2d(signal, out=np.zeros(proc.signal_2d_shape))
            return signal

        return proc, slopes

    yield build
    for proc, outputs in built:
        proc.close()
        clear_shms(list(outputs.values()))


# -- the file format ------------------------------------------------------------


def test_save_and_load_round_trip(tmp_path):
    path = tmp_path / "im.npy"  # the configured name is kept, whatever its suffix
    data = np.arange(6, dtype=np.float32).reshape(3, 2)
    written = cal.save_calibration(path, data, "interaction_matrix", note="test")
    assert written == str(path) and path.exists()
    assert not list(tmp_path.glob(".pyrtc-cal-*"))  # the temporary file is gone

    loaded = cal.load_calibration(path, "interaction_matrix")
    assert not loaded.legacy
    np.testing.assert_array_equal(loaded.data, data)
    assert loaded.data.dtype == np.float32
    assert loaded.metadata["format"] == cal.CALIBRATION_FORMAT
    assert loaded.metadata["kind"] == "interaction_matrix"
    assert loaded.metadata["image_axes"] == "yx"
    assert loaded.metadata["note"] == "test"

    # Plain numpy still reads it: an .npz archive with the array under "data".
    with np.load(path) as archive:
        np.testing.assert_array_equal(archive["data"], data)
        assert json.loads(str(archive["pyrtc_calibration"][()]))["format"] == 2


def test_plain_npy_files_are_legacy(tmp_path):
    path = tmp_path / "old.npy"
    np.save(path, np.ones(3))
    loaded = cal.load_calibration(path, "ref_slopes")
    assert loaded.legacy and loaded.metadata is None
    np.testing.assert_array_equal(loaded.data, np.ones(3))

    archive = tmp_path / "old.npz"
    np.savez(archive, only=np.zeros(2))
    assert cal.load_calibration(archive).legacy
    np.savez(archive, a=np.zeros(2), b=np.zeros(2))
    with pytest.raises(cal.CalibrationError, match="2 arrays"):
        cal.load_calibration(archive)


def test_newer_formats_and_wrong_kinds_are_refused(tmp_path):
    path = tmp_path / "cal.npz"
    cal.save_calibration(path, np.zeros(2), "ref_slopes")
    with pytest.raises(cal.CalibrationError, match="not 'interaction_matrix'"):
        cal.load_calibration(path, "interaction_matrix")

    cal.save_calibration(path, np.zeros(2), "ref_slopes", format=cal.CALIBRATION_FORMAT + 1)
    with pytest.raises(cal.CalibrationError, match="newer pyrtc"):
        cal.load_calibration(path)
    with pytest.raises(ValueError, match="unknown calibration kind"):
        cal.save_calibration(path, np.zeros(2), "flat")


def test_legacy_calibration_values_are_validated():
    assert cal.normalize_legacy_calibration(None) is None
    assert cal.normalize_legacy_calibration("") is None
    assert cal.normalize_legacy_calibration(" XY ") == "xy"
    with pytest.raises(ValueError, match="legacy_calibration must be one of"):
        cal.normalize_legacy_calibration("transposed")


def test_refusal_message_explains_the_options(tmp_path):
    message = cal.legacy_calibration_message(tmp_path / "ref.npy", "ref_slopes")
    for text in ("pyrtc 1.x", "'slopes' config section", "'yx'", "'xy'", "'as_is'"):
        assert text in message
    assert "take_ref_slopes()" in message and "Migrating to pyrtc 2.0" in message


# -- conversions ------------------------------------------------------------------


def test_slope_maps_swap_and_transpose_their_halves():
    sx = np.arange(9.0).reshape(3, 3)
    sy = 100 + sx
    legacy = np.concatenate([sx, sy])
    converted = cal.slope_map_from_legacy(legacy, "xy")
    np.testing.assert_array_equal(converted, np.concatenate([sy.T, sx.T]))
    for frame in ("yx", "as_is"):
        np.testing.assert_array_equal(cal.slope_map_from_legacy(legacy, frame), legacy)
    with pytest.raises(cal.CalibrationError, match="SHWFS slope map"):
        cal.slope_map_from_legacy(np.zeros((3, 3)), "xy")

    left, right = np.arange(4.0).reshape(2, 2), 10 + np.arange(4.0).reshape(2, 2)
    pywfs = np.concatenate([left, right], axis=1)
    np.testing.assert_array_equal(
        cal.slope_map_from_legacy(pywfs, "xy", "pywfs"), np.concatenate([left.T, right.T], 1)
    )
    np.testing.assert_array_equal(
        cal.slope_map_from_legacy(pywfs, "xy", "pywfs", swap_pywfs_halves=True),
        np.concatenate([right.T, left.T], 1),
    )


def test_images_are_transposed_only_from_the_xy_frame():
    image = np.arange(6).reshape(2, 3)
    np.testing.assert_array_equal(cal.image_from_legacy(image, "xy"), image.T)
    assert cal.image_from_legacy(image, "xy").flags.c_contiguous
    np.testing.assert_array_equal(cal.image_from_legacy(image, "yx"), image)


@pytest.mark.parametrize("frame", ["xy", "yx"])
@pytest.mark.parametrize("sub", [7, 8])
def test_converted_shwfs_reference_cancels_the_reference_frame(shwfs_factory, frame, sub):
    """A 1.x reference of an image reads zero residual on the same image in 2.0."""

    image = _spot_image(1, sub=sub)
    proc, slopes = shwfs_factory(shape=image.shape, sub=sub)
    camera_frame_1x = image.T if frame == "xy" else image
    raw_1x = slopes(camera_frame_1x) - (0.5 if sub % 2 == 0 else 0.0)

    ref = cal.shwfs_ref_slopes_from_legacy(raw_1x, frame, sub)
    np.testing.assert_allclose(ref, slopes(image), atol=1e-4)
    proc.set_ref_slopes(ref)
    np.testing.assert_allclose(slopes(image), 0.0, atol=1e-4)


def test_wcog_reference_without_reference_image_is_refused():
    with pytest.raises(cal.LegacyCalibrationError, match="WCoG"):
        cal.shwfs_ref_slopes_from_legacy(np.zeros((8, 4)), "yx", 8, centroider="wcog")
    # Exact when the weights follow a reference image, or for odd sub-apertures.
    cal.shwfs_ref_slopes_from_legacy(
        np.zeros((8, 4)), "yx", 8, centroider="wcog", has_reference_image=True
    )
    cal.shwfs_ref_slopes_from_legacy(np.zeros((8, 4)), "yx", 7, centroider="wcog")
    np.testing.assert_array_equal(
        cal.shwfs_ref_slopes_from_legacy(np.ones((8, 4)), "as_is", 8, centroider="wcog"), 1.0
    )


def test_signal_order_and_interaction_matrix_follow_the_new_frame(tmp_path, shwfs_factory):
    """A 1.x "xy" signal, reordered, is the 2.0 signal; the IM's rows move with it."""

    rng = np.random.default_rng(3)
    valid_1x = rng.random((2 * N_SUB, N_SUB)) > 0.25
    valid_1x[N_SUB:] = valid_1x[:N_SUB]  # x and y slopes of the same sub-apertures
    order, valid = cal.signal_permutation_from_legacy(valid_1x, "xy")
    np.testing.assert_array_equal(valid, cal.slope_map_from_legacy(valid_1x, "xy"))

    cal.save_calibration(tmp_path / "valid_1x.npz", valid_1x, "valid_sub_aps")
    cal.save_calibration(tmp_path / "valid.npz", valid, "valid_sub_aps")
    _, slopes_1x = shwfs_factory(valid_sub_aps_file=str(tmp_path / "valid_1x.npz"))
    _, slopes = shwfs_factory(valid_sub_aps_file=str(tmp_path / "valid.npz"))
    images = [_spot_image(seed) for seed in range(5)]
    im_1x = np.stack([slopes_1x(image.T, two_d=False) for image in images], axis=1)
    im = np.stack([slopes(image, two_d=False) for image in images], axis=1)
    np.testing.assert_allclose(im_1x[order], im, atol=1e-5)
    np.testing.assert_allclose(cal.interaction_matrix_from_legacy(im_1x, "xy", valid_1x), im)

    # From the yx frame (and as_is) nothing moves.
    np.testing.assert_array_equal(cal.interaction_matrix_from_legacy(im_1x, "yx"), im_1x)
    with pytest.raises(cal.LegacyCalibrationError, match="pyrtc-migrate-calibration"):
        cal.interaction_matrix_from_legacy(im_1x, "xy")
    with pytest.raises(cal.CalibrationError, match="rows"):
        cal.interaction_matrix_from_legacy(im_1x[:-1], "xy", valid_1x)


@pytest.mark.parametrize("configured", [True, False])
def test_pywfs_reference_slopes_from_the_xy_frame(configured):
    """1.x PYWFS reference slopes through a transposing adapter, converted, match 2.0's.

    A configured ``pupils`` list keeps its strings: 1.x read "a,b" as row a,
    column b of its transposed stream, the same camera pixel as 2.0's "x,y".
    The default layout swaps its second and third pupils with the frame.
    """

    size = 48
    rng = np.random.default_rng(5)
    image = rng.uniform(1.0, 2.0, (size, size)).astype(np.float32)
    strings = ["12,10", "12,36", "36,12", "34,36"]

    def reference(frame, pupils):
        wfs = private_stream("wfs", (size, size), np.float32)
        wfs.write(frame)
        suffix = uuid.uuid4().hex[:8]
        outputs = {"signal": f"sig_{suffix}", "signal_2d": f"sig2d_{suffix}"}
        conf = {
            "type": "PYWFS",
            "signal_type": "slopes",
            "functions": [],
            "input_streams": {"wfs": wfs.name},
            "output_streams": outputs,
        }
        if pupils is not None:
            conf.update({"pupils": pupils, "pupils_radius": 8})
        proc = SlopesProcess(conf)
        try:
            proc.compute_signal()
            return proc.compute_signal_2d(
                np.array(proc.read(block=False)), out=np.zeros(proc.valid_sub_aps.shape)
            )
        finally:
            proc.close()
            clear_shms(list(outputs.values()))

    # 1.x parsed "a,b" as (x=b, y=a) of its own stream, which 2.0 spells "b,a".
    legacy_pupils = [",".join(s.split(",")[::-1]) for s in strings] if configured else None
    ref_1x = reference(np.ascontiguousarray(image.T), legacy_pupils)
    expected = reference(image, strings if configured else None)
    converted = cal.slope_map_from_legacy(ref_1x, "xy", "pywfs", swap_pywfs_halves=not configured)
    np.testing.assert_allclose(converted, expected, atol=1e-6)


# -- components -------------------------------------------------------------------


def _wfs(tmp_path, monkeypatch, **conf):
    import pyrtc.wavefront_sensor as wavefront_sensor

    monkeypatch.setattr(wavefront_sensor, "create_stream", private_stream)
    base = {"name": "wfs", "width": 6, "height": 4, "functions": []}
    return WavefrontSensor({**base, **conf})


def test_wfs_dark_files_are_versioned_and_legacy_ones_converted(tmp_path, monkeypatch):
    dark = np.arange(24, dtype=np.int32).reshape(4, 6)  # (height, width)
    legacy_xy = tmp_path / "dark_xy.npy"
    np.save(legacy_xy, dark.T)  # 1.x through a transposing adapter: (width, height)
    legacy_yx = tmp_path / "dark_yx.npy"
    np.save(legacy_yx, dark)

    with pytest.raises(cal.LegacyCalibrationError, match="'wfs' config section"):
        _wfs(tmp_path, monkeypatch, dark_file=str(legacy_xy))
    wfs = _wfs(tmp_path, monkeypatch, dark_file=str(legacy_xy), legacy_calibration="xy")
    try:
        np.testing.assert_array_equal(wfs.dark, dark)
        saved = tmp_path / "dark_v2.npy"
        wfs.save_dark(str(saved))
        assert not cal.load_calibration(saved).legacy
        wfs.legacy_calibration = None  # a 2.0 file needs no legacy setting
        wfs.load_dark(str(saved))
        np.testing.assert_array_equal(wfs.dark, dark)
        wfs.legacy_calibration = "yx"
        wfs.load_dark(str(legacy_yx))
        np.testing.assert_array_equal(wfs.dark, dark)
        wfs.legacy_calibration = "as_is"
        with pytest.raises(cal.CalibrationError, match=r"\(height, width\) = \(4, 6\)"):
            wfs.load_dark(str(legacy_xy))  # a (width, height) array, not converted
    finally:
        wfs.close()


def test_science_camera_files_are_versioned_and_legacy_ones_converted(tmp_path, monkeypatch):
    import pyrtc.science_camera as science_camera

    monkeypatch.setattr(science_camera, "create_stream", private_stream)
    model = np.random.default_rng(0).random((4, 6))
    legacy = tmp_path / "model.npy"
    np.save(legacy, model.T)
    conf = {"name": "psf", "width": 6, "height": 4, "dark_count": 1, "integration": 1}
    conf["functions"] = []
    with pytest.raises(cal.LegacyCalibrationError, match="'psf' config section"):
        ScienceCamera({**conf, "model_file": str(legacy)})
    camera = ScienceCamera({**conf, "model_file": str(legacy), "legacy_calibration": "xy"})
    try:
        assert camera.image_shape == (4, 6)
        np.testing.assert_allclose(camera.model, model)
        camera.save_model_psf(str(tmp_path / "model_v2.npz"))
        camera.save_dark(str(tmp_path / "dark_v2.npz"))
        assert cal.load_calibration(tmp_path / "model_v2.npz", "psf_model").metadata
        assert cal.load_calibration(tmp_path / "dark_v2.npz", "psf_dark").metadata
    finally:
        camera.close()


def test_slopes_process_converts_or_refuses_legacy_files(tmp_path, shwfs_factory):
    image = _spot_image(7)
    proc, slopes = shwfs_factory()
    raw_1x = slopes(image.T) - 0.5  # 1.x, transposing adapter, even sub-apertures
    valid_1x = np.ones((2 * N_SUB, N_SUB), dtype=bool)
    valid_1x[0, 1] = valid_1x[N_SUB, 1] = False
    np.save(tmp_path / "ref.npy", raw_1x)
    np.save(tmp_path / "valid.npy", valid_1x)
    np.save(tmp_path / "refimg.npy", image.T)
    files = {
        "ref_slopes_file": str(tmp_path / "ref.npy"),
        "valid_sub_aps_file": str(tmp_path / "valid.npy"),
    }

    with pytest.raises(cal.LegacyCalibrationError, match="valid sub-aperture mask"):
        shwfs_factory(**files)
    converted, slopes = shwfs_factory(**files, legacy_calibration="xy")
    np.testing.assert_array_equal(
        converted.valid_sub_aps, cal.slope_map_from_legacy(valid_1x, "xy")
    )
    np.testing.assert_allclose(slopes(image), 0.0, atol=1e-4)

    converted.load_reference_image(str(tmp_path / "refimg.npy"))
    np.testing.assert_array_equal(converted.reference_image, image)

    # Saved again, they are 2.0 files and load without the legacy setting.
    converted.save_ref_slopes(str(tmp_path / "ref_v2.npy"))
    converted.save_valid_sub_aps(str(tmp_path / "valid_v2.npy"))
    record = cal.load_calibration(tmp_path / "ref_v2.npy").metadata
    assert record["kind"] == "ref_slopes" and record["sub_aperture_size"] == SUB
    _, slopes_v2 = shwfs_factory(
        ref_slopes_file=str(tmp_path / "ref_v2.npy"),
        valid_sub_aps_file=str(tmp_path / "valid_v2.npy"),
    )
    np.testing.assert_allclose(slopes_v2(image), 0.0, atol=1e-4)

    # A file for another geometry is refused with its shape.
    np.save(tmp_path / "small.npy", np.zeros((4, 2)))
    with pytest.raises(cal.CalibrationError, match="needs"):
        converted.legacy_calibration = "as_is"
        converted.load_ref_slopes(str(tmp_path / "small.npy"))


def test_loop_interaction_matrix_legacy_handling(tmp_path):
    loop = bare_component(Loop)
    loop.im = np.zeros((6, 2), dtype=np.float32)
    loop.compute_cm = lambda: None
    legacy = tmp_path / "im.npy"
    im = np.arange(12, dtype=np.float32).reshape(6, 2)
    np.save(legacy, im)
    loop.im_file = str(legacy)

    loop.legacy_calibration = None
    with pytest.raises(cal.LegacyCalibrationError, match="'loop' config section"):
        loop.load_im()
    loop.legacy_calibration = "xy"
    with pytest.raises(cal.LegacyCalibrationError, match="pyrtc-migrate-calibration"):
        loop.load_im()
    for frame in ("yx", "as_is"):
        loop.legacy_calibration = frame
        loop.im = np.zeros((6, 2), dtype=np.float32)
        loop.load_im()
        np.testing.assert_array_equal(loop.im, im)

    loop.im_file = str(tmp_path / "im_v2.npy")
    loop.save_im()
    loop.legacy_calibration = None
    loop.load_im()
    np.testing.assert_array_equal(loop.im, im)
    loop.im = np.zeros((5, 2), dtype=np.float32)
    with pytest.raises(cal.CalibrationError, match="needs"):
        loop.load_im()


def test_component_configs_accept_legacy_calibration():
    from pyrtc.component_descriptors import validate_config_with_descriptor

    for section in ("wfs", "psf", "slopes", "loop"):
        conf = {"legacy_calibration": "XY"}
        if section in ("wfs", "psf"):
            conf.update(width=4, height=4)
        if section == "psf":
            conf.update(name="psf", dark_count=1, integration=1)
        if section == "slopes":
            conf.update(type="SHWFS", signal_type="slopes")
        validate_config_with_descriptor(section, conf)
        with pytest.raises(ValueError, match="legacy_calibration"):
            validate_config_with_descriptor(section, {**conf, "legacy_calibration": "transposed"})


# -- offline migration --------------------------------------------------------------


def test_migrate_calibration_cli(tmp_path, shwfs_factory):
    from pyrtc.scripts.migrate_calibration import main

    rng = np.random.default_rng(11)
    valid_1x = rng.random((2 * N_SUB, N_SUB)) > 0.3
    valid_1x[N_SUB:] = valid_1x[:N_SUB]
    size = int(np.count_nonzero(valid_1x))
    im_1x = rng.normal(size=(size, 3)).astype(np.float32)
    np.save(tmp_path / "im.npy", im_1x)
    np.save(tmp_path / "valid.npy", valid_1x)

    out = tmp_path / "im_v2.npz"
    argv = ["interaction_matrix", str(tmp_path / "im.npy"), str(out), "--legacy-frame", "xy"]
    assert main([*argv, "--valid-sub-aps", str(tmp_path / "valid.npy")]) == 0
    migrated = cal.load_calibration(out, "interaction_matrix")
    expected = cal.interaction_matrix_from_legacy(im_1x, "xy", valid_1x)
    np.testing.assert_array_equal(migrated.data, expected)
    assert migrated.metadata["legacy_calibration"] == "xy"

    # Without the mask the error explains what is missing; 2.0 inputs are refused.
    assert main([*argv[:2], str(tmp_path / "x.npz"), "--legacy-frame", "xy"]) == 1
    assert main([*argv[:1], str(out), str(tmp_path / "y.npz"), "--legacy-frame", "yx"]) == 1

    np.save(tmp_path / "ref.npy", np.zeros((2 * N_SUB, N_SUB), dtype=np.float32))
    assert (
        main(
            [
                "ref_slopes",
                str(tmp_path / "ref.npy"),
                str(tmp_path / "ref_v2.npz"),
                "--legacy-frame",
                "yx",
                "--sub-aperture-size",
                "8",
            ]
        )
        == 0
    )
    np.testing.assert_array_equal(cal.load_calibration(tmp_path / "ref_v2.npz").data, 0.5)
