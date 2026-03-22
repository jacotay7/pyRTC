import numpy as np
import importlib

slopes_mod = importlib.import_module("pyRTC.SlopesProcess")


def test_slope_algorithms_numpy_numba():
    img = np.arange(16, dtype=np.float32).reshape(4, 4)
    p1 = np.zeros_like(img, dtype=bool)
    p2 = np.zeros_like(img, dtype=bool)
    p3 = np.zeros_like(img, dtype=bool)
    p4 = np.zeros_like(img, dtype=bool)
    p1[:2, :2] = True
    p2[:2, 2:] = True
    p3[2:, :2] = True
    p4[2:, 2:] = True

    n = int(np.sum(p1))
    slopes = np.zeros(2 * n, dtype=np.float32)
    ref = np.zeros_like(slopes)

    out = slopes_mod.computeSlopesPYWFSOptimNumpy(
        image=img.ravel(),
        p1Mask=p1.ravel(),
        p2Mask=p2.ravel(),
        p3Mask=p3.ravel(),
        p4Mask=p4.ravel(),
        p1=np.zeros(n, dtype=np.float32),
        p2=np.zeros(n, dtype=np.float32),
        p3=np.zeros(n, dtype=np.float32),
        p4=np.zeros(n, dtype=np.float32),
        tmp1=np.zeros(n, dtype=np.float32),
        tmp2=np.zeros(n, dtype=np.float32),
        numPixelsInPupils=n,
        slopes=slopes,
        refSlopes=ref,
    )
    assert out.shape == (2 * n,)


def test_pywfs_slope_algorithms_return_zero_on_dark_frame():
    img = np.zeros((4, 4), dtype=np.float32)
    p1 = np.zeros_like(img, dtype=bool)
    p2 = np.zeros_like(img, dtype=bool)
    p3 = np.zeros_like(img, dtype=bool)
    p4 = np.zeros_like(img, dtype=bool)
    p1[:2, :2] = True
    p2[:2, 2:] = True
    p3[2:, :2] = True
    p4[2:, 2:] = True

    n = int(np.sum(p1))
    ref = np.ones(2 * n, dtype=np.float32)

    numpy_out = slopes_mod.computeSlopesPYWFSOptimNumpy(
        image=img.ravel(),
        p1Mask=p1.ravel(),
        p2Mask=p2.ravel(),
        p3Mask=p3.ravel(),
        p4Mask=p4.ravel(),
        p1=np.zeros(n, dtype=np.float32),
        p2=np.zeros(n, dtype=np.float32),
        p3=np.zeros(n, dtype=np.float32),
        p4=np.zeros(n, dtype=np.float32),
        tmp1=np.zeros(n, dtype=np.float32),
        tmp2=np.zeros(n, dtype=np.float32),
        numPixelsInPupils=n,
        slopes=np.zeros(2 * n, dtype=np.float32),
        refSlopes=ref,
    )

    numba_out = slopes_mod.computeSlopesPYWFSOptimNumba(
        image=img.ravel(),
        p1Mask=p1.ravel(),
        p2Mask=p2.ravel(),
        p3Mask=p3.ravel(),
        p4Mask=p4.ravel(),
        p1=np.zeros(n, dtype=np.float32),
        p2=np.zeros(n, dtype=np.float32),
        p3=np.zeros(n, dtype=np.float32),
        p4=np.zeros(n, dtype=np.float32),
        tmp1=np.zeros(n, dtype=np.float32),
        tmp2=np.zeros(n, dtype=np.float32),
        numPixelsInPupils=n,
        slopes=np.zeros(2 * n, dtype=np.float32),
        refSlopes=ref,
    )

    assert np.all(np.isfinite(numpy_out))
    assert np.all(np.isfinite(numba_out))
    assert np.all(numpy_out == 0.0)
    assert np.all(numba_out == 0.0)



def test_torch_path_disabled(monkeypatch):
    monkeypatch.setattr(slopes_mod, "gpu_torch_available", lambda: False)
    try:
        slopes_mod.computeSlopesPYWFSTorch(None, None, None, None, None, 0, None, None)
        assert False
    except ImportError:
        assert True



def test_slopes_process_methods(tmp_path):
    sp = slopes_mod.SlopesProcess.__new__(slopes_mod.SlopesProcess)
    sp.signalDType = np.float32
    sp.wfsType = "pywfs"
    sp.validSubAps = np.ones((4, 8), dtype=bool)
    sp.curSignal2D = np.zeros((4, 8), dtype=np.float32)

    class _Sig:
        def read_noblock(self):
            return np.zeros(np.count_nonzero(sp.validSubAps), dtype=np.float32)

    sp.signal = _Sig()

    sp.setValidSubAps(np.ones((4, 8)))
    assert sp.validSubAps.dtype == bool

    sp.validSubApsFile = str(tmp_path / "valid.npy")
    sp.saveValidSubAps()
    sp.setValidSubAps(np.zeros((4, 8), dtype=bool))
    sp.loadValidSubAps()
    assert np.all(sp.validSubAps)

    sp.refSlopes = np.zeros((4, 8), dtype=np.float32)
    sp.refSlopesFile = str(tmp_path / "ref.npy")
    sp.setRefSlopes(np.ones((4, 8), dtype=np.float32))
    sp.saveRefSlopes()
    sp.setRefSlopes(np.zeros((4, 8), dtype=np.float32))
    sp.loadRefSlopes()
    assert np.all(sp.refSlopes == 1)

    sig = np.arange(np.count_nonzero(sp.validSubAps), dtype=np.float32)
    out2d = sp.computeSignal2D(sig)
    assert out2d.shape == (4, 8)



def test_compute_signal2d_shwfs():
    sp = slopes_mod.SlopesProcess.__new__(slopes_mod.SlopesProcess)
    sp.wfsType = "shwfs"
    sp.validSubAps = np.array([[True, False], [False, True]])
    sp.curSignal2D = np.zeros((2, 2), dtype=np.float32)
    out = sp.computeSignal2D(np.array([1.0, 2.0], dtype=np.float32))
    assert out[0, 0] == 1.0
    assert out[1, 1] == 2.0


def test_set_pupils_registers_pywfs_output_streams(monkeypatch):
    sp = slopes_mod.SlopesProcess.__new__(slopes_mod.SlopesProcess)
    sp.signalType = "slopes"
    sp.signalDType = np.float32
    sp.gpuDevice = None
    sp.validSubApsFile = ""
    sp._stream_inputs = {}
    sp._stream_outputs = {}
    sp._stream_defaults = {}
    sp._last_stream_metadata = {}
    sp.system_streams = {}
    sp.section_name = None

    monkeypatch.setattr(
        sp,
        "computePupilsMask",
        lambda: setattr(sp, "pupilMask", np.ones((4, 4), dtype=bool)),
    )
    monkeypatch.setattr(
        sp,
        "setValidSubAps",
        lambda valid_sub_aps: (
            setattr(sp, "validSubAps", valid_sub_aps.astype(bool)),
            setattr(sp, "curSignal2D", np.zeros(valid_sub_aps.shape, dtype=np.float32)),
        ),
    )
    monkeypatch.setattr(slopes_mod, "clear_shms", lambda names: None)
    monkeypatch.setattr(slopes_mod, "initExistingShm", lambda name, gpuDevice=None: (_ for _ in ()).throw(FileNotFoundError(name)))

    class _FakeShm:
        def __init__(self, name, shape, dtype, gpuDevice=None, consumer=False):
            self.name = name
            self.shape = shape
            self.dtype = dtype

    monkeypatch.setattr(slopes_mod, "ImageSHM", _FakeShm)

    sp.setPupils([(1, 1), (1, 2), (2, 1), (2, 2)], 1)

    assert "signal" in sp._stream_outputs
    assert "signal2D" in sp._stream_outputs
