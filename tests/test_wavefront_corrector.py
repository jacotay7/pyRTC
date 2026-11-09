import numpy as np
import pytest
import importlib

wfc_mod = importlib.import_module("pyrtc.wavefront_corrector")


def test_modal_to_zonal_with_flat():
    corr = np.array([1.0, 2.0], dtype=np.float32)
    m2c = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32)
    flat = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    out = wfc_mod.ModaltoZonalWithFlat(corr, m2c, flat)
    assert out.shape == (3,)


def test_wavefront_corrector_core(monkeypatch, tmp_path):
    from testsupport import private_stream

    monkeypatch.setattr(wfc_mod, "create_stream", private_stream)

    conf = {
        "name": "wfc",
        "num_actuators": 9,
        "num_modes": 4,
        "m2c_file": "",
        "frame_delay": 1,
        "save_file": str(tmp_path / "shape.npy"),
        "functions": [],
    }
    wfc = wfc_mod.WavefrontCorrector(conf)

    layout = np.ones((3, 3), dtype=bool)
    wfc.set_layout(layout)
    assert wfc.correction_vector_2d is not None

    m2c = np.random.RandomState(0).randn(9, 4).astype(np.float32)
    wfc.set_m2c(m2c)
    assert wfc.M2C.shape == (9, 4)

    corr = np.ones(4, dtype=np.float32)
    wfc.write(corr)
    assert np.array_equal(wfc.read(), corr)

    wfc.send_to_hardware()
    assert wfc.current_shape.shape == (9,)

    wfc.push(1, 0.25)
    pushed = wfc.read()
    assert np.isclose(pushed[1], 0.25)

    wfc.flatten()
    assert np.all(wfc.read() == 0)

    wfc.deactivate_actuators([0, 2])
    assert not wfc.actuator_status[0]
    wfc.reactivate_actuators([0, 2])
    assert wfc.actuator_status[0]

    wfc.save_shape()
    assert (tmp_path / "shape.npy").exists()


def test_wavefront_corrector_applies_command_cap(monkeypatch):
    from testsupport import private_stream

    monkeypatch.setattr(wfc_mod, "create_stream", private_stream)

    conf = {
        "name": "wfc",
        "num_actuators": 3,
        "num_modes": 3,
        "m2c_file": "",
        "command_cap": 0.5,
        "functions": [],
    }
    wfc = wfc_mod.WavefrontCorrector(conf)

    wfc.set_m2c(np.eye(3, dtype=np.float32))
    wfc.write(np.array([2.0, -0.75, 0.25], dtype=np.float32))

    wfc.send_to_hardware()

    assert np.allclose(wfc.current_shape, np.array([0.5, -0.5, 0.25], dtype=np.float32))


def test_wavefront_corrector_clears_wfc2d_outside_layout(monkeypatch):
    from testsupport import private_stream

    monkeypatch.setattr(wfc_mod, "create_stream", private_stream)

    conf = {
        "name": "wfc",
        "num_actuators": 5,
        "num_modes": 5,
        "m2c_file": "",
        "functions": [],
    }
    wfc = wfc_mod.WavefrontCorrector(conf)

    layout = np.array(
        [
            [False, True, False],
            [True, True, True],
            [False, True, False],
        ],
        dtype=bool,
    )
    wfc.set_layout(layout)
    wfc.set_m2c(np.eye(5, dtype=np.float32))
    wfc.write(np.arange(5, dtype=np.float32))

    # Simulate stale garbage in the non-actuator area from a previous frame.
    wfc.correction_vector_2d_template[:] = 1234.0

    wfc.send_to_hardware()

    wfc2d = wfc.correction_vector_2d.read()
    assert np.all(wfc2d[~layout] == 0.0)
    assert np.array_equal(wfc2d[layout], np.arange(5, dtype=np.float32))


def test_wavefront_corrector_reports_saturation(monkeypatch, caplog):
    from testsupport import private_stream

    monkeypatch.setattr(wfc_mod, "create_stream", private_stream)
    wfc = wfc_mod.WavefrontCorrector(
        {
            "name": "wfc",
            "num_actuators": 4,
            "num_modes": 4,
            "m2c_file": "",
            "command_cap": 0.5,
            "saturation_warn_fraction": 0.5,
            "functions": [],
        }
    )
    wfc.set_m2c(np.eye(4, dtype=np.float32))
    try:
        wfc.write(np.array([0.1, 0.2, -0.3, 0.4], dtype=np.float32))
        wfc.send_to_hardware()
        status = wfc.safety_status()
        assert status["saturated_actuators"] == 0 and status["alerts"] == []

        wfc.write(np.array([2.0, -0.75, 0.25, 0.1], dtype=np.float32))
        wfc.send_to_hardware()
        status = wfc.safety_status()
        assert status["saturated_actuators"] == 2
        assert status["saturated_fraction"] == pytest.approx(0.5)
        assert status["saturated_frames"] == 1
        assert status["alerts"] == ["2/4 actuators at command_cap=0.5"]
        assert "2 of 4 actuators (50%) at command_cap=0.5" in caplog.text
    finally:
        wfc.close()


def test_wavefront_corrector_without_cap_reports_no_saturation(monkeypatch):
    from testsupport import private_stream

    monkeypatch.setattr(wfc_mod, "create_stream", private_stream)
    wfc = wfc_mod.WavefrontCorrector(
        {"name": "wfc", "num_actuators": 2, "num_modes": 2, "m2c_file": "", "functions": []}
    )
    try:
        wfc.set_m2c(np.eye(2, dtype=np.float32))
        wfc.write(np.array([5.0, -5.0], dtype=np.float32))
        wfc.send_to_hardware()
        assert wfc.safety_status()["saturated_actuators"] == 0
        assert wfc.safety_status()["alerts"] == []
    finally:
        wfc.close()


def _limited_wfc(monkeypatch, **conf):
    from testsupport import private_stream

    monkeypatch.setattr(wfc_mod, "create_stream", private_stream)
    wfc = wfc_mod.WavefrontCorrector(
        {"name": "wfc", "num_actuators": 4, "num_modes": 4, "m2c_file": "", "functions": [], **conf}
    )
    wfc.set_m2c(np.eye(4, dtype=np.float32))
    return wfc


def _apply(wfc, command):
    wfc.write(np.array(command, dtype=np.float32))
    wfc.send_to_hardware()
    return wfc.current_shape


def test_one_sided_command_limit(monkeypatch):
    """``command_min: 0`` alone: an actuator that only pushes, with no upper limit."""
    wfc = _limited_wfc(monkeypatch, command_min=0)
    try:
        shape = _apply(wfc, [-1.0, 0.0, 0.5, 7.0])
        assert np.allclose(shape, [0.0, 0.0, 0.5, 7.0])
        assert shape.dtype == np.float32
        assert wfc.safety_status()["saturated_actuators"] == 2  # at or below the limit
        assert wfc.command_limits()[1] is None
    finally:
        wfc.close()


def test_per_actuator_command_limits(monkeypatch, tmp_path):
    upper = tmp_path / "upper.npy"
    np.save(upper, np.array([1.0, 2.0, 3.0, 4.0]))
    wfc = _limited_wfc(monkeypatch, command_min=[0.0, -1.0, 0.0, -1.0], command_max=str(upper))
    try:
        assert np.allclose(_apply(wfc, [5.0, 5.0, -5.0, -5.0]), [1.0, 2.0, 0.0, -1.0])
        status = wfc.safety_status()
        assert status["saturated_actuators"] == 4
        assert status["alerts"] == ["4/4 actuators at a command limit"]
    finally:
        wfc.close()


def test_command_limits_apply_to_the_flat_and_combine_with_cap(monkeypatch):
    wfc = _limited_wfc(monkeypatch, command_min=0.0, command_max=10.0, command_cap=2.0)
    try:
        wfc.flat = np.full(4, 1.5, dtype=np.float32)
        # flat + command: 0.5 (inside), 3.5 (cap is tighter than command_max), -0.5 (min)
        assert np.allclose(_apply(wfc, [-1.0, 2.0, -2.0, 0.0]), [0.5, 2.0, 0.0, 1.5])
        wfc.command_cap = 5.0  # changed at run time, as through set_property
        assert np.allclose(_apply(wfc, [-1.0, 2.0, -2.0, 0.0]), [0.5, 3.5, 0.0, 1.5])
    finally:
        wfc.close()


def test_set_command_limits_at_run_time_and_validation(monkeypatch):
    wfc = _limited_wfc(monkeypatch)
    try:
        assert wfc.command_limits() == (None, None)
        assert np.allclose(_apply(wfc, [-9.0, 9.0, 0.0, 0.0]), [-9.0, 9.0, 0.0, 0.0])
        wfc.set_command_limits(-1.0, [1.0, 2.0, 3.0, 4.0])
        assert np.allclose(_apply(wfc, [-9.0, 9.0, 0.0, 0.0]), [-1.0, 2.0, 0.0, 0.0])
        wfc.set_command_limits()
        assert wfc.command_limits() == (None, None)
        with pytest.raises(ValueError, match="num_actuators=4"):
            wfc.set_command_limits([0.0, 0.0])
        with pytest.raises(ValueError, match="must not exceed"):
            wfc.set_command_limits(1.0, 0.0)
        with pytest.raises(ValueError, match="NaN"):
            wfc.set_command_limits(float("nan"))
    finally:
        wfc.close()
