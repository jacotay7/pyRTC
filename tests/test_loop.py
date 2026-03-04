import numpy as np
import importlib

loop_mod = importlib.import_module("pyRTC.Loop")


def test_loop_helper_functions(monkeypatch):
    slopes = np.array([1.0, 2.0], dtype=np.float32)
    cm = np.eye(2, dtype=np.float32)
    old = np.array([0.5, 0.5], dtype=np.float32)
    correction = np.zeros(2, dtype=np.float32)

    out = loop_mod.leakyIntegratorNumba(slopes, cm, old, correction, np.float32(0.1), 1)
    assert out.shape == (2,)

    assert np.array_equal(loop_mod.compCorrection(cm, slopes), slopes)
    upd = loop_mod.updateCorrection(np.array([1.0, 1.0], dtype=np.float32), cm, slopes)
    assert np.array_equal(upd, np.array([0.0, -1.0], dtype=np.float32))

    monkeypatch.setattr(loop_mod, "gpu_torch_available", lambda: False)
    try:
        loop_mod.leakIntegratorGPU(slopes, cm, old, 0.1, 1)
        assert False
    except ImportError:
        assert True


def test_loop_methods_without_full_init(tmp_path):
    loop = loop_mod.Loop.__new__(loop_mod.Loop)
    loop.numModes = 4
    loop.numDroppedModes = 1
    loop.numActiveModes = 3
    loop.IM = np.random.RandomState(0).randn(6, 4).astype(np.float32)
    loop.CM = np.zeros((4, 6), dtype=np.float32)
    loop.gain = 0.2
    loop.computeCM()
    assert loop.CM.shape == (4, 6)

    loop.setGain(0.5)
    assert np.allclose(loop.gCM, 0.5 * loop.CM)

    loop.setPeturbAmp(0.3)
    assert np.isclose(loop.perturbAmp, 0.3)

    loop.IMFile = str(tmp_path / "im.npy")
    loop.saveIM()
    loop.IM = np.zeros_like(loop.IM)
    loop.loadIM()
    assert np.any(loop.IM != 0)

    loop.fIM = np.copy(loop.IM)
    correction = np.ones(4, dtype=np.float32)
    slopes = np.ones(6, dtype=np.float32)
    upd = loop.updateCorrectionPOL(correction, slopes)
    assert upd.shape == (4,)

    # pid integrator path
    loop.CM = np.eye(4, 6, dtype=np.float32)
    loop.leakyGain = 0.0
    loop.controlLimits = [-1.0, 1.0]
    loop.integralLimits = [-5.0, 5.0]
    loop.absoluteLimits = [-2.0, 2.0]
    loop.pGain = 0.1
    loop.iGain = 0.01
    loop.dGain = 0.01
    loop.derivativeFilter = 0.5
    loop.previousWfError = np.zeros(4, dtype=np.float32)
    loop.previousDerivative = np.zeros(4, dtype=np.float32)
    loop.controlOutput = np.zeros(4, dtype=np.float32)
    loop.integral = np.zeros(4, dtype=np.float32)
    loop.sendToWfc = lambda correction, slopes=None: setattr(loop, "_sent", correction)
    loop.numActiveModes = 3
    loop.pidIntegrator(slopes=np.ones(6, dtype=np.float32), correction=np.zeros(4, dtype=np.float32))
    assert hasattr(loop, "_sent")

    # sendToWfc branch with CL DOCRIME
    class _W:
        def __init__(self):
            self.last = None

        def write(self, x):
            self.last = np.asarray(x)

    loop.wfcShm = _W()
    loop.flat = np.zeros(4, dtype=np.float32)
    loop.clDocrime = True
    loop.pokeAmp = 0.1
    loop.docrimeBuffer = np.zeros((2, 4, 1), dtype=np.float32)
    loop.docrimeCross = np.zeros((6, 4), dtype=np.float32)
    loop.docrimeAuto = np.zeros((4, 4), dtype=np.float32)
    loop.numItersDC = 0
    loop.numActiveModes = 3
    loop.sendToWfc = loop_mod.Loop.sendToWfc.__get__(loop, loop_mod.Loop)
    loop.sendToWfc(np.zeros(4, dtype=np.float32), slopes=np.ones(6, dtype=np.float32))
    assert loop.numItersDC == 1

    loop.IMFile = str(tmp_path / "im.npy")
    loop.docrimeAuto = np.eye(4, dtype=np.float32)
    loop.docrimeCross = np.ones((6, 4), dtype=np.float32)
    loop.solveDocrime()
