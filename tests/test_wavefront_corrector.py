import numpy as np
import importlib

wfc_mod = importlib.import_module("pyRTC.WavefrontCorrector")


def test_modal_to_zonal_with_flat():
    corr = np.array([1.0, 2.0], dtype=np.float32)
    m2c = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32)
    flat = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    out = wfc_mod.ModaltoZonalWithFlat(corr, m2c, flat)
    assert out.shape == (3,)


def test_wavefront_corrector_core(monkeypatch, tmp_path):
    from testsupport import DummySHM

    monkeypatch.setattr(wfc_mod, "ImageSHM", DummySHM)

    conf = {
        "name": "wfc",
        "numActuators": 9,
        "numModes": 4,
        "m2cFile": "",
        "frameDelay": 1,
        "saveFile": str(tmp_path / "shape.npy"),
        "functions": [],
    }
    wfc = wfc_mod.WavefrontCorrector(conf)

    layout = np.ones((3, 3), dtype=bool)
    wfc.setLayout(layout)
    assert wfc.correctionVector2D is not None

    m2c = np.random.RandomState(0).randn(9, 4).astype(np.float32)
    wfc.setM2C(m2c)
    assert wfc.M2C.shape == (9, 4)

    corr = np.ones(4, dtype=np.float32)
    wfc.write(corr)
    assert np.array_equal(wfc.read(), corr)

    wfc.sendToHardware()
    assert wfc.currentShape.shape == (9,)

    wfc.push(1, 0.25)
    pushed = wfc.read()
    assert np.isclose(pushed[1], 0.25)

    wfc.flatten()
    assert np.all(wfc.read() == 0)

    wfc.deactivateActuators([0, 2])
    assert not wfc.actuatorStatus[0]
    wfc.reactivateActuators([0, 2])
    assert wfc.actuatorStatus[0]

    wfc.saveShape()
    assert (tmp_path / "shape.npy").exists()


def test_wavefront_corrector_applies_command_cap(monkeypatch):
    from testsupport import DummySHM

    monkeypatch.setattr(wfc_mod, "ImageSHM", DummySHM)

    conf = {
        "name": "wfc",
        "numActuators": 3,
        "numModes": 3,
        "m2cFile": "",
        "commandCap": 0.5,
        "functions": [],
    }
    wfc = wfc_mod.WavefrontCorrector(conf)

    wfc.setM2C(np.eye(3, dtype=np.float32))
    wfc.write(np.array([2.0, -0.75, 0.25], dtype=np.float32))

    wfc.sendToHardware()

    assert np.allclose(wfc.currentShape, np.array([0.5, -0.5, 0.25], dtype=np.float32))


def test_wavefront_corrector_clears_wfc2d_outside_layout(monkeypatch):
    from testsupport import DummySHM

    monkeypatch.setattr(wfc_mod, "ImageSHM", DummySHM)

    conf = {
        "name": "wfc",
        "numActuators": 5,
        "numModes": 5,
        "m2cFile": "",
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
    wfc.setLayout(layout)
    wfc.setM2C(np.eye(5, dtype=np.float32))
    wfc.write(np.arange(5, dtype=np.float32))

    # Simulate stale garbage in the non-actuator area from a previous frame.
    wfc.correctionVector2D_template[:] = 1234.0

    wfc.sendToHardware()

    wfc2d = wfc.correctionVector2D.read_noblock()
    assert np.all(wfc2d[~layout] == 0.0)
    assert np.array_equal(wfc2d[layout], np.arange(5, dtype=np.float32))
