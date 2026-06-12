import numpy as np
import importlib

slopes_mod = importlib.import_module("pyrtc.slopes_process")


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

    out = slopes_mod.compute_slopes_pywfs_optim_numpy(
        image=img.ravel(),
        p1_mask=p1.ravel(),
        p2_mask=p2.ravel(),
        p3_mask=p3.ravel(),
        p4_mask=p4.ravel(),
        p1=np.zeros(n, dtype=np.float32),
        p2=np.zeros(n, dtype=np.float32),
        p3=np.zeros(n, dtype=np.float32),
        p4=np.zeros(n, dtype=np.float32),
        tmp1=np.zeros(n, dtype=np.float32),
        tmp2=np.zeros(n, dtype=np.float32),
        num_pixels_in_pupils=n,
        slopes=slopes,
        ref_slopes=ref,
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

    numpy_out = slopes_mod.compute_slopes_pywfs_optim_numpy(
        image=img.ravel(),
        p1_mask=p1.ravel(),
        p2_mask=p2.ravel(),
        p3_mask=p3.ravel(),
        p4_mask=p4.ravel(),
        p1=np.zeros(n, dtype=np.float32),
        p2=np.zeros(n, dtype=np.float32),
        p3=np.zeros(n, dtype=np.float32),
        p4=np.zeros(n, dtype=np.float32),
        tmp1=np.zeros(n, dtype=np.float32),
        tmp2=np.zeros(n, dtype=np.float32),
        num_pixels_in_pupils=n,
        slopes=np.zeros(2 * n, dtype=np.float32),
        ref_slopes=ref,
    )

    numba_out = slopes_mod.compute_slopes_pywfs_optim_numba(
        image=img.ravel(),
        p1_mask=p1.ravel(),
        p2_mask=p2.ravel(),
        p3_mask=p3.ravel(),
        p4_mask=p4.ravel(),
        p1=np.zeros(n, dtype=np.float32),
        p2=np.zeros(n, dtype=np.float32),
        p3=np.zeros(n, dtype=np.float32),
        p4=np.zeros(n, dtype=np.float32),
        tmp1=np.zeros(n, dtype=np.float32),
        tmp2=np.zeros(n, dtype=np.float32),
        num_pixels_in_pupils=n,
        slopes=np.zeros(2 * n, dtype=np.float32),
        ref_slopes=ref,
    )

    assert np.all(np.isfinite(numpy_out))
    assert np.all(np.isfinite(numba_out))
    assert np.all(numpy_out == 0.0)
    assert np.all(numba_out == 0.0)



def test_torch_path_disabled(monkeypatch):
    monkeypatch.setattr(slopes_mod, "gpu_torch_available", lambda: False)
    try:
        slopes_mod.compute_slopes_pywfs_torch(None, None, None, None, None, 0, None, None)
        assert False
    except ImportError:
        assert True



def test_slopes_process_methods(tmp_path):
    sp = slopes_mod.SlopesProcess.__new__(slopes_mod.SlopesProcess)
    sp.signal_dtype = np.float32
    sp.wfs_type = "pywfs"
    sp.valid_sub_aps = np.ones((4, 8), dtype=bool)
    sp.cur_signal_2d = np.zeros((4, 8), dtype=np.float32)

    class _Sig:
        def read(self):
            return np.zeros(np.count_nonzero(sp.valid_sub_aps), dtype=np.float32)

    sp.signal = _Sig()

    sp.set_valid_sub_aps(np.ones((4, 8)))
    assert sp.valid_sub_aps.dtype == bool

    sp.valid_sub_aps_file = str(tmp_path / "valid.npy")
    sp.save_valid_sub_aps()
    sp.set_valid_sub_aps(np.zeros((4, 8), dtype=bool))
    sp.load_valid_sub_aps()
    assert np.all(sp.valid_sub_aps)

    sp.ref_slopes = np.zeros((4, 8), dtype=np.float32)
    sp.ref_slopes_file = str(tmp_path / "ref.npy")
    sp.set_ref_slopes(np.ones((4, 8), dtype=np.float32))
    sp.save_ref_slopes()
    sp.set_ref_slopes(np.zeros((4, 8), dtype=np.float32))
    sp.load_ref_slopes()
    assert np.all(sp.ref_slopes == 1)

    sig = np.arange(np.count_nonzero(sp.valid_sub_aps), dtype=np.float32)
    out2d = sp.compute_signal_2d(sig)
    assert out2d.shape == (4, 8)



def test_compute_signal2d_shwfs():
    sp = slopes_mod.SlopesProcess.__new__(slopes_mod.SlopesProcess)
    sp.wfs_type = "shwfs"
    sp.valid_sub_aps = np.array([[True, False], [False, True]])
    sp.cur_signal_2d = np.zeros((2, 2), dtype=np.float32)
    out = sp.compute_signal_2d(np.array([1.0, 2.0], dtype=np.float32))
    assert out[0, 0] == 1.0
    assert out[1, 1] == 2.0


def test_set_pupils_registers_pywfs_output_streams(monkeypatch):
    sp = slopes_mod.SlopesProcess.__new__(slopes_mod.SlopesProcess)
    sp.signal_type = "slopes"
    sp.signal_dtype = np.float32
    sp.gpu_device = None
    sp.valid_sub_aps_file = ""
    sp._stream_inputs = {}
    sp._stream_outputs = {}
    sp._stream_defaults = {}
    sp._last_stream_metadata = {}
    sp.system_streams = {}
    sp.section_name = None

    monkeypatch.setattr(
        sp,
        "compute_pupils_mask",
        lambda: setattr(sp, "pupil_mask", np.ones((4, 4), dtype=bool)),
    )
    monkeypatch.setattr(
        sp,
        "set_valid_sub_aps",
        lambda valid_sub_aps: (
            setattr(sp, "valid_sub_aps", valid_sub_aps.astype(bool)),
            setattr(sp, "cur_signal_2d", np.zeros(valid_sub_aps.shape, dtype=np.float32)),
        ),
    )
    monkeypatch.setattr(slopes_mod, "clear_shms", lambda names: None)
    monkeypatch.setattr(slopes_mod, "open_stream", lambda name, gpu_device=None: (_ for _ in ()).throw(FileNotFoundError(name)))

    class _FakeShm:
        def __init__(self, name, shape, dtype, gpu_device=None, consumer=False):
            self.name = name
            self.shape = shape
            self.dtype = dtype

    monkeypatch.setattr(slopes_mod, "create_stream", _FakeShm)

    sp.set_pupils([(1, 1), (1, 2), (2, 1), (2, 2)], 1)

    assert "signal" in sp._stream_outputs
    assert "signal_2d" in sp._stream_outputs
