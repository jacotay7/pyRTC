import numpy as np
import importlib

import pytest

sci_mod = importlib.import_module("pyrtc.science_camera")


def test_science_camera_core(monkeypatch, tmp_path):
    from testsupport import private_stream

    monkeypatch.setattr(sci_mod, "create_stream", private_stream)

    conf = {
        "name": "psf",
        "width": 8,
        "height": 8,
        "dark_count": 2,
        "integration": 2,
        "functions": [],
    }
    cam = sci_mod.ScienceCamera(conf)

    cam.set_roi([2, 3, 4, 5])
    cam.set_exposure(10)
    cam.set_binning(2)
    cam.set_gain(1)
    cam.set_gamma(2.2)
    cam.set_bit_depth(16)
    cam.set_integration_length(2)

    cam.data = np.ones((8, 8), dtype=np.uint16) * 5
    cam.dark = np.ones((8, 8), dtype=np.int32) * 2
    cam.expose()
    assert np.all(cam.read(block=False) == 3)

    # Integrate from mocked reads
    frames = [np.ones((8, 8), dtype=np.int32) * 3, np.ones((8, 8), dtype=np.int32) * 5]
    cam.read = lambda block=True: frames.pop(0)
    cam.integrate()
    assert np.allclose(cam.read_long(), 4)

    dark_file = tmp_path / "dark.npy"
    model_file = tmp_path / "model.npy"
    cam.set_dark(np.ones((8, 8), dtype=np.int32) * 7)
    cam.save_dark(str(dark_file))
    cam.load_dark(str(dark_file))
    assert np.all(cam.dark == 7)

    cam.set_model_psf(np.ones((8, 8), dtype=np.float64) * 10)
    cam.save_model_psf(str(model_file))
    cam.load_model_psf(str(model_file))
    assert np.all(cam.model == 10)

    cam.psf_long.write(np.ones((8, 8), dtype=np.float64) * 10)
    sr = cam.compute_strehl()
    assert np.isclose(sr, 1.0)


def test_science_camera_default_files_plot_and_error_paths(monkeypatch, tmp_path):
    from testsupport import private_stream

    monkeypatch.setattr(sci_mod, "create_stream", private_stream)

    conf = {
        "name": "psf",
        "width": 4,
        "height": 4,
        "dark_count": 2,
        "integration": 2,
        "functions": [],
    }
    cam = sci_mod.ScienceCamera(conf)

    assert np.array_equal(cam.dark, np.zeros((4, 4), dtype=np.int32))
    assert np.array_equal(cam.model, np.zeros((4, 4), dtype=np.float64))

    with pytest.raises(ValueError, match="No dark frame filename provided"):
        cam.save_dark()
    with pytest.raises(ValueError, match="No model PSF filename provided"):
        cam.save_model_psf()

    frame = np.arange(16, dtype=np.int32).reshape(4, 4)
    cam.psf_short.write(frame)
    monkeypatch.setattr(sci_mod.plt, "imshow", lambda *args, **kwargs: None)
    monkeypatch.setattr(sci_mod.plt, "colorbar", lambda *args, **kwargs: None)
    monkeypatch.setattr(sci_mod.plt, "show", lambda: None)
    cam.plot()

    cam.psf_long.write(np.ones((4, 4), dtype=np.float64) * 5)
    cam.take_model_psf()
    assert np.array_equal(cam.model, np.ones((4, 4), dtype=np.float64) * 5)

    # The first frame (possibly subtracted with the old dark) is discarded.
    frames = [np.full((4, 4), value, dtype=np.int32) for value in (100, 2, 4)]
    cam.read = lambda block=True: frames.pop(0)
    cam.take_dark()
    assert np.all(cam.dark == 3)


def test_science_camera_setter_and_load_error_paths(monkeypatch):
    from testsupport import private_stream

    monkeypatch.setattr(sci_mod, "create_stream", private_stream)
    cam = sci_mod.ScienceCamera(
        {
            "name": "psf",
            "width": 4,
            "height": 4,
            "dark_count": 1,
            "integration": 1,
            "functions": [],
        }
    )

    class BadLogger:
        def info(self, *args, **kwargs):
            raise RuntimeError("log failed")

        def exception(self, *args, **kwargs):
            return None

    cam.logger = BadLogger()

    with pytest.raises(RuntimeError, match="log failed"):
        cam.set_roi([1, 2, 3, 4])
    with pytest.raises(RuntimeError, match="log failed"):
        cam.set_exposure(1)
    with pytest.raises(RuntimeError, match="log failed"):
        cam.set_binning(1)
    with pytest.raises(RuntimeError, match="log failed"):
        cam.set_gain(1)
    with pytest.raises(RuntimeError, match="log failed"):
        cam.set_gamma(1.0)
    with pytest.raises(RuntimeError, match="log failed"):
        cam.set_bit_depth(16)
    with pytest.raises(RuntimeError, match="log failed"):
        cam.set_integration_length(1)
    with pytest.raises(RuntimeError, match="log failed"):
        cam.set_dark(np.ones((4, 4), dtype=np.int32))
    with pytest.raises(RuntimeError, match="log failed"):
        cam.set_model_psf(np.ones((4, 4), dtype=np.float64))

    cam = sci_mod.ScienceCamera(
        {
            "name": "psf",
            "width": 4,
            "height": 4,
            "dark_count": 0,
            "integration": 1,
            "functions": [],
        }
    )
    with pytest.raises(ValueError, match="dark_count must be at least 1"):
        cam.take_dark()

    monkeypatch.setattr(
        sci_mod.np, "load", lambda filename: (_ for _ in ()).throw(RuntimeError("load failed"))
    )
    with pytest.raises(RuntimeError, match="load failed"):
        cam.load_dark("missing-dark.npy")
    with pytest.raises(RuntimeError, match="load failed"):
        cam.load_model_psf("missing-model.npy")


def _gaussian_psf(sigma, size=33):
    y, x = np.indices((size, size)) - size // 2
    return np.exp(-(x**2 + y**2) / (2 * sigma**2))


@pytest.mark.parametrize("flux_scale", [0.01, 1.0, 250.0])
def test_estimate_strehl_is_independent_of_flux(flux_scale):
    model = _gaussian_psf(1.5)
    aberrated = _gaussian_psf(3.0)
    reference = sci_mod.estimate_strehl(aberrated, model)

    assert 0 < reference < 1
    assert np.isclose(sci_mod.estimate_strehl(flux_scale * aberrated, model), reference)
    assert np.isclose(sci_mod.estimate_strehl(model * flux_scale, model), 1.0)


def test_estimate_strehl_without_flux_is_zero():
    model = _gaussian_psf(1.5)
    assert sci_mod.estimate_strehl(np.zeros_like(model), model) == 0.0
    assert sci_mod.estimate_strehl(model, np.zeros_like(model)) == 0.0
