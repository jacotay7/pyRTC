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
    loop.CMMethod = "svd"
    loop.conditioning = None
    loop.tikhonovReg = 0.0
    loop.lastSingularValues = np.array([], dtype=np.float64)
    loop.lastRetainedSingularMask = np.array([], dtype=bool)
    loop.lastSuggestedConditioning = None
    loop.lastSingularValueFit = None
    loop.IM = np.random.RandomState(0).randn(6, 4).astype(np.float32)
    loop.CM = np.zeros((4, 6), dtype=np.float32)
    loop.gain = 0.2
    loop.computeCM()
    assert loop.CM.shape == (4, 6)

    loop.computeCM(conditioning=10.0)
    assert loop.conditioning == 10.0
    assert loop.lastSingularValues.size == min(loop.IM[:, : loop.numActiveModes].shape)

    loop.computeCM(method="tikhonov", conditioning=10.0, tikhonovReg=0.05)
    assert loop.CMMethod == "tikhonov"
    assert np.isclose(loop.tikhonovReg, 0.05)

    suggestion = loop.suggestConditioningNumber()
    assert suggestion is None or suggestion >= 1.0
    if suggestion is not None:
        assert loop.lastSingularValueFit is not None
        assert "fit_curve" in loop.lastSingularValueFit

    plotted = loop.plotSingularValues()
    assert plotted is None or plotted >= 1.0

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


def test_standard_integrator_uses_nonblocking_wfc_read():
    loop = loop_mod.Loop.__new__(loop_mod.Loop)
    loop.RELEASE_GIL = True
    loop.gCM = np.eye(4, dtype=np.float32) * 0.25
    loop.nullCorrection = np.zeros(4, dtype=np.float32)
    loop.numActiveModes = 3

    class _Signal:
        def read(self, SAFE=False, RELEASE_GIL=True):
            return np.ones(4, dtype=np.float32)

    class _Wfc:
        def read(self, SAFE=False):
            raise AssertionError("standardIntegrator should not block on wfc.read()")

        def read_noblock(self, SAFE=False):
            return np.zeros(4, dtype=np.float32)

    sent = {}
    loop.signalShm = _Signal()
    loop.wfcShm = _Wfc()
    loop.sendToWfc = lambda correction, slopes=None: sent.setdefault("correction", correction.copy())

    loop.standardIntegrator()

    assert "correction" in sent
    assert np.max(np.abs(sent["correction"])) > 0


def test_loop_compute_cm_zero_matrix_without_failure():
    loop = loop_mod.Loop.__new__(loop_mod.Loop)
    loop.numModes = 3
    loop.numDroppedModes = 0
    loop.numActiveModes = 3
    loop.CMMethod = "svd"
    loop.conditioning = None
    loop.tikhonovReg = 0.0
    loop.lastSingularValueFit = None
    loop.IM = np.zeros((5, 3), dtype=np.float32)
    loop.CM = np.zeros((3, 5), dtype=np.float32)
    loop.gain = 0.1

    loop.computeCM()

    assert np.allclose(loop.CM, 0.0)
    assert loop.lastSingularValues.size == 3


def test_conditioning_suggestion_tracks_knee():
    singular_values = np.array([1.0, 0.5, 0.25, 0.125, 1e-3, 5e-4], dtype=np.float64)

    suggestion, fit = loop_mod.Loop._suggest_conditioning_from_singular_values(singular_values)

    assert suggestion is not None
    assert fit is not None
    assert fit["suggested_index"] == 4
    assert np.isclose(suggestion, 1.0 / singular_values[4])
