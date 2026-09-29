"""HCIPy simulator bridge (#55); skipped without HCIPy."""

import numpy as np
import pytest

pytest.importorskip("hcipy")

from pyrtc.hardware.hcipy_interface import (  # noqa: E402
    DEFAULT_PARAMS,
    HCIPySystemContext,
)

SMALL = {
    "num_lenslets": 6,
    "pixels_per_lenslet": 8,
    "num_actuators_across": 7,
    "science_num_airy": 8,
}


def test_context_geometry_follows_the_parameters():
    context = HCIPySystemContext(SMALL)
    assert context.wfs_shape == (48, 48)
    assert context.psf_shape == (2 * 4 * 8, 2 * 4 * 8)
    layout = context.actuator_layout
    assert layout.shape == (7, 7)
    assert context.dm.num_actuators == int(layout.sum()) == context.actuator_positions.shape[0]
    # Fried geometry: the actuators kept are those inside the pupil.
    radius = np.hypot(*context.actuator_positions.T)
    assert radius.max() <= 0.5 * DEFAULT_PARAMS["telescope_diameter"] * 1.1


def test_unknown_parameters_and_wfs_types_are_rejected():
    with pytest.raises(ValueError, match="unknown HCIPy parameters"):
        HCIPySystemContext({"lenslet_count": 10})
    with pytest.raises(ValueError, match="wfs_type"):
        HCIPySystemContext({**SMALL, "wfs_type": "curvature"})


def test_flat_system_gives_a_diffraction_limited_psf_and_centred_spots():
    context = HCIPySystemContext(SMALL)
    psf = context.psf_image()
    assert psf.max() == pytest.approx(1.0, rel=1e-6)  # normalized to the unaberrated peak
    image = context.wfs_image()
    assert image.sum() == pytest.approx(DEFAULT_PARAMS["wfs_photons"], rel=1e-6)
    # Each lit sub-aperture's spot sits at its centre.
    sub = image[16:24, 16:24]
    yy, xx = np.indices(sub.shape)
    assert (yy * sub).sum() / sub.sum() == pytest.approx(3.5, abs=0.05)
    assert (xx * sub).sum() / sub.sum() == pytest.approx(3.5, abs=0.05)


def test_dm_commands_move_the_spots_and_degrade_the_psf():
    context = HCIPySystemContext(SMALL)
    flat = context.wfs_image()
    xx = np.indices((8, 8))[1]

    def x_shift(slope):
        context.set_actuators(context.actuator_positions[:, 0] * slope)
        tilted = context.wfs_image()[16:24, 16:24]
        base = flat[16:24, 16:24]
        return (xx * tilted).sum() / tilted.sum() - (xx * base).sum() / base.sum()

    # A DM tilt moves the spots along x, proportionally and with its sign.
    small, large, negative = x_shift(1e-7), x_shift(2e-7), x_shift(-1e-7)
    assert abs(small) > 0.5
    assert large == pytest.approx(2 * small, rel=0.1)
    assert negative == pytest.approx(-small, rel=0.05)
    context.set_actuators(np.zeros(context.dm.num_actuators))
    context.set_actuators(np.random.default_rng(0).normal(0, 1e-7, context.dm.num_actuators))
    assert context.psf_image().max() < 0.9


def test_atmosphere_evolves_only_when_enabled():
    context = HCIPySystemContext(SMALL)
    start = context.layer.t
    context.wfs_image()
    assert context.layer.t == start
    context.add_atmosphere()
    first = context.wfs_image()
    second = context.wfs_image()
    assert context.layer.t == pytest.approx(start + 2 * context.frame_time)
    assert not np.allclose(first, second)
    assert context.psf_image().max() < 1.0
    context.remove_atmosphere()
    assert context.psf_image().max() == pytest.approx(1.0, rel=1e-6)


def test_pyramid_wfs_builds_four_pupil_images():
    context = HCIPySystemContext(
        {**SMALL, "wfs_type": "pywfs", "pyramid_pupil_pixels": 16, "pyramid_modulation_steps": 4}
    )
    image = context.wfs_image()
    assert image.shape == context.wfs_shape == (32, 32)
    quadrants = [image[:16, :16], image[:16, 16:], image[16:, :16], image[16:, 16:]]
    sums = np.array([q.sum() for q in quadrants])
    assert np.all(sums > 0.15 * sums.sum())  # light in all four pupils


def test_components_adopt_the_simulated_geometry(monkeypatch):
    from testsupport import private_stream

    import pyrtc.science_camera as science_camera
    import pyrtc.wavefront_corrector as wavefront_corrector
    import pyrtc.wavefront_sensor as wavefront_sensor
    from pyrtc.hardware.hcipy_interface import (
        HCIPyScienceCamera,
        HCIPyWFCorrector,
        HCIPyWFSensor,
    )

    for module in (wavefront_sensor, wavefront_corrector, science_camera):
        monkeypatch.setattr(module, "create_stream", private_stream)
    context = HCIPySystemContext(SMALL)
    wfs = HCIPyWFSensor({"name": "wfs", "width": 10, "height": 10, "functions": []}, context)
    wfc = HCIPyWFCorrector(
        {
            "name": "wfc",
            "num_actuators": 1,
            "num_modes": 5,
            "basis": {"type": "kl"},
            "functions": [],
        },
        context,
    )
    psf = HCIPyScienceCamera(
        {
            "name": "psf",
            "width": 8,
            "height": 8,
            "dark_count": 1,
            "integration": 2,
            "functions": [],
        },
        context,
    )
    try:
        assert (wfs.width, wfs.height) == (48, 48)
        assert wfc.num_actuators == context.dm.num_actuators
        assert wfc.M2C.shape == (context.dm.num_actuators, 5)
        np.testing.assert_allclose(wfc.basis_actuator_positions(), context.actuator_positions)
        assert tuple(psf.image_shape) == context.psf_shape
        wfc.write(np.array([1e-8, 0, 0, 0, 0], dtype=np.float32))
        wfc.send_to_hardware()
        np.testing.assert_allclose(context.dm.actuators, wfc.current_shape, rtol=1e-6)
        wfs.expose()
        assert wfs.data.dtype == np.uint16 and wfs.data.shape == (48, 48)
        psf.expose()
        assert psf.data.max() > 0
    finally:
        for component in (wfs, wfc, psf):
            component.close()
