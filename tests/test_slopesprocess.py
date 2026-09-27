import numpy as np
import pytest
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
    monkeypatch.setattr(
        slopes_mod,
        "open_stream",
        lambda name, gpu_device=None: (_ for _ in ()).throw(FileNotFoundError(name)),
    )

    class _FakeShm:
        def __init__(self, name, shape, dtype, gpu_device=None, consumer=False):
            self.name = name
            self.shape = shape
            self.dtype = dtype

    monkeypatch.setattr(slopes_mod, "create_stream", _FakeShm)

    sp.set_pupils([(1, 1), (1, 2), (2, 1), (2, 2)], 1)

    assert "signal" in sp._stream_outputs
    assert "signal_2d" in sp._stream_outputs


def _pywfs_process(gpu_device, *, size=64, radius=10):
    """Minimal PYWFS ``SlopesProcess`` (no streams) for exercising compute_signal."""

    sp = slopes_mod.SlopesProcess.__new__(slopes_mod.SlopesProcess)
    sp.image_shape = (size, size)
    sp.signal_dtype = np.float32
    sp.signal_type = "slopes"
    sp.wfs_type = "pywfs"
    sp.central_obscuration_ratio = 0.0
    lo, hi = size // 4, 3 * size // 4
    sp.pupil_locs = [(lo, lo), (hi, lo), (lo, hi), (hi, hi)]
    sp.pupil_radius = radius
    sp.compute_pupils_mask()
    n = int(np.count_nonzero(sp.p1mask))
    sp.num_pixels_in_pupils = n
    slopemask = sp.pupil_mask[lo - radius : lo + radius, lo - radius : lo + radius] > 0
    sp.valid_sub_aps = np.concatenate([slopemask, slopemask], axis=1)
    sp.cur_signal_2d = np.zeros(sp.valid_sub_aps.shape)
    sp.ref_slopes = np.zeros(sp.valid_sub_aps.shape, dtype=np.float32)
    sp.ref_slopes_1d = np.zeros(2 * n, dtype=np.float32)
    sp.slopes_arr_1d = np.zeros(2 * n, dtype=np.float32)
    for attr in ("p1", "p2", "p3", "p4", "tmp1", "tmp2"):
        setattr(sp, attr, np.empty(n, dtype=np.float32))
    sp._image_buffer = None
    sp.gpu_device = gpu_device

    class _Signal:
        gpu_device = None

        def read(self):
            return np.zeros(2 * n, dtype=np.float32)

    sp.signal = _Signal()
    sp.written = {}
    sp.write_stream = lambda name, value: sp.written.__setitem__(name, value)
    return sp


def _run_compute_signal(sp, image):
    sp.read_stream = lambda name, out=None, **kwargs: image
    sp.compute_signal()
    return sp.written["signal"], sp.written["signal_2d"].copy()


def _numba_reference(sp, image):
    return slopes_mod.compute_slopes_pywfs_optim_numba(
        image=np.asarray(image).ravel(),
        p1_mask=sp.p1mask.ravel(),
        p2_mask=sp.p2mask.ravel(),
        p3_mask=sp.p3mask.ravel(),
        p4_mask=sp.p4mask.ravel(),
        p1=np.empty_like(sp.p1),
        p2=np.empty_like(sp.p1),
        p3=np.empty_like(sp.p1),
        p4=np.empty_like(sp.p1),
        tmp1=np.empty_like(sp.p1),
        tmp2=np.empty_like(sp.p1),
        num_pixels_in_pupils=sp.num_pixels_in_pupils,
        slopes=np.zeros_like(sp.slopes_arr_1d),
        ref_slopes=sp.ref_slopes_1d,
    )


def _cuda_available():
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


@pytest.mark.gpu
@pytest.mark.skipif(not _cuda_available(), reason="CUDA is not available")
def test_gpu_pywfs_matches_cpu_numba_path():
    import torch

    rng = np.random.default_rng(64)
    image = rng.integers(0, 4000, (240, 240)).astype(np.uint16)
    sp = _pywfs_process("cuda:0", size=240, radius=30)
    ref_2d = rng.normal(scale=0.01, size=sp.valid_sub_aps.shape).astype(np.float32)
    sp.set_ref_slopes(ref_2d)
    expected = _numba_reference(sp, image)

    # WFS stream GPU-backed: read_stream returns a CUDA tensor (float or raw uint16).
    for gpu_image in (
        torch.as_tensor(image, device="cuda:0"),
        torch.as_tensor(image.astype(np.float32), device="cuda:0"),
    ):
        signal, signal_2d = _run_compute_signal(sp, gpu_image)
        assert isinstance(signal, np.ndarray)
        np.testing.assert_allclose(signal, expected, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(signal_2d, sp.compute_signal_2d(expected), atol=1e-6)

    # WFS producer made a CPU stream: read_stream returns NumPy even with gpu_device.
    signal, _ = _run_compute_signal(sp, image)
    np.testing.assert_allclose(signal, expected, rtol=1e-5, atol=1e-6)

    # A GPU-backed signal stream receives the device tensor directly.
    sp.signal.gpu_device = "cuda:0"
    signal, signal_2d = _run_compute_signal(sp, image)
    assert isinstance(signal, torch.Tensor) and signal.is_cuda
    np.testing.assert_allclose(signal.cpu().numpy(), expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(signal_2d, sp.compute_signal_2d(expected), atol=1e-6)

    # Dark frame: both paths return zeros rather than dividing by ~0.
    dark = np.zeros_like(image)
    sp.signal.gpu_device = None
    signal, _ = _run_compute_signal(sp, torch.as_tensor(dark, device="cuda:0"))
    assert np.all(signal == 0.0)
    assert np.all(_numba_reference(sp, dark) == 0.0)


def test_gpu_pywfs_device_cache_rebuilds_on_change():
    """Masks/ref slopes are uploaded once and re-uploaded only when they change.

    Uses torch's CPU device, so the cache logic is covered without CUDA.
    """
    pytest.importorskip("torch")

    rng = np.random.default_rng(1)
    image = rng.random((64, 64)).astype(np.float32) + 1.0
    sp = _pywfs_process("cpu")

    masks, slopes, ref = sp._gpu_pywfs_tensors()
    masks2, slopes2, ref2 = sp._gpu_pywfs_tensors()
    assert all(a is b for a, b in zip(masks, masks2))
    assert slopes is slopes2 and ref is ref2
    for idx, mask in zip(masks, (sp.p1mask, sp.p2mask, sp.p3mask, sp.p4mask)):
        np.testing.assert_array_equal(idx.numpy(), np.flatnonzero(mask))

    signal, _ = _run_compute_signal(sp, image)
    np.testing.assert_allclose(signal, _numba_reference(sp, image), rtol=1e-5, atol=1e-6)

    # New reference slopes: only the reference tensor is rebuilt, and used.
    sp.set_ref_slopes(np.full(sp.valid_sub_aps.shape, 0.25, dtype=np.float32))
    masks3, _, ref3 = sp._gpu_pywfs_tensors()
    assert ref3 is not ref
    assert all(a is b for a, b in zip(masks, masks3))
    np.testing.assert_allclose(ref3.numpy(), sp.ref_slopes_1d)
    signal, _ = _run_compute_signal(sp, image)
    np.testing.assert_allclose(signal, _numba_reference(sp, image), rtol=1e-5, atol=1e-6)

    # Loading/resetting reference slopes goes through set_ref_slopes too.
    sp.ref_slopes_file = ""
    sp.load_ref_slopes()
    _, _, ref4 = sp._gpu_pywfs_tensors()
    assert ref4 is not ref3
    assert np.all(ref4.numpy() == 0.0)

    # Direct reassignment of the 1-D reference is picked up by identity.
    sp.ref_slopes_1d = np.full_like(sp.ref_slopes_1d, -0.5)
    _, _, ref5 = sp._gpu_pywfs_tensors()
    np.testing.assert_allclose(ref5.numpy(), -0.5)

    # New pupil geometry: the mask indices are rebuilt.
    sp.pupil_locs = [(x + 1, y) for x, y in sp.pupil_locs]
    sp.compute_pupils_mask()
    masks6, _, ref6 = sp._gpu_pywfs_tensors()
    assert all(a is not b for a, b in zip(masks, masks6))
    for idx, mask in zip(masks6, (sp.p1mask, sp.p2mask, sp.p3mask, sp.p4mask)):
        np.testing.assert_array_equal(idx.numpy(), np.flatnonzero(mask))
    signal, _ = _run_compute_signal(sp, image)
    np.testing.assert_allclose(signal, _numba_reference(sp, image), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "conf, message",
    [
        ({"type": "SHWFS", "signal_type": "phase"}, "signal_type"),
        ({"type": "SHWFS"}, "signal_type"),
        ({"type": "curvature", "signal_type": "slopes"}, "type"),
    ],
)
def test_slopes_process_rejects_unsupported_types_before_starting(conf, message):
    # Raised before Component.__init__, so no worker thread or stream is created.
    with pytest.raises(ValueError, match=f"unsupported {message}"):
        slopes_mod.SlopesProcess({**conf, "functions": ["compute_signal"]})


def test_slopes_process_normalizes_type_case():
    assert slopes_mod.SlopesProcess.normalize_signal_types(
        {"type": "ShWfS", "signal_type": "SLOPES"}
    ) == (
        "shwfs",
        "slopes",
    )


def test_compute_signal_raises_for_unsupported_signal_type():
    sp = slopes_mod.SlopesProcess.__new__(slopes_mod.SlopesProcess)
    sp.signal_type = "phase"
    sp.wfs_type = "shwfs"
    sp._image_buffer = None
    sp.read_stream = lambda *args, **kwargs: np.zeros((4, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="unsupported signal_type"):
        sp.compute_signal()
