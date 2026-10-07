import contextlib
import importlib
import logging
import threading
import time
import uuid

import numpy as np
import pytest

from pyrtc.streams import clear_shms
from testsupport import private_stream

from testsupport import bare_component

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
    sp = bare_component(slopes_mod.SlopesProcess)
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
    sp = bare_component(slopes_mod.SlopesProcess)
    sp.wfs_type = "shwfs"
    sp.valid_sub_aps = np.array([[True, False], [False, True]])
    sp.cur_signal_2d = np.zeros((2, 2), dtype=np.float32)
    out = sp.compute_signal_2d(np.array([1.0, 2.0], dtype=np.float32))
    assert out[0, 0] == 1.0
    assert out[1, 1] == 2.0


def test_set_pupils_registers_pywfs_output_streams(monkeypatch):
    sp = bare_component(slopes_mod.SlopesProcess)
    sp.signal_type = "slopes"
    sp.wfs_type = "pywfs"
    sp.signal_dtype = np.float32
    sp.gpu_device = None
    sp.valid_sub_aps_file = ""
    sp.image_shape = (8, 8)
    sp.central_obscuration_ratio = 0.0

    # The default names are global; keep this test off them.
    monkeypatch.setattr(slopes_mod, "clear_shms", lambda names: None)
    monkeypatch.setattr(
        slopes_mod,
        "open_stream",
        lambda name, gpu_device=None: (_ for _ in ()).throw(FileNotFoundError(name)),
    )
    monkeypatch.setattr(slopes_mod, "create_stream", private_stream)

    sp.set_pupils([(2, 2), (6, 2), (2, 6), (6, 6)], 2)

    assert "signal" in sp._stream_outputs
    assert "signal_2d" in sp._stream_outputs
    n = int(np.count_nonzero(sp.p1mask))
    assert sp.num_pixels_in_pupils == n
    assert sp._stream_outputs["signal"].shape == (2 * n,)
    assert sp.slopes_arr_1d.shape == sp.ref_slopes_1d.shape == (2 * n,)
    assert sp.ref_slopes.shape == sp.valid_sub_aps.shape


def test_set_pupils_rejects_overlapping_pupils(monkeypatch):
    sp = bare_component(slopes_mod.SlopesProcess)
    sp.signal_type = "slopes"
    sp.wfs_type = "pywfs"
    sp.signal_dtype = np.float32
    sp.gpu_device = None
    sp.valid_sub_aps_file = ""
    sp.image_shape = (16, 16)
    sp.central_obscuration_ratio = 0.0
    monkeypatch.setattr(slopes_mod, "clear_shms", lambda names: None)
    monkeypatch.setattr(slopes_mod, "create_stream", private_stream)

    # Overlapping pupils would make the numba kernel write past its buffers.
    with pytest.raises(ValueError, match="pupil geometry"):
        sp.set_pupils([(5, 5), (8, 5), (5, 11), (11, 11)], 3)


def _pywfs_process(gpu_device, *, size=64, radius=10):
    """Minimal PYWFS ``SlopesProcess`` (no streams) for exercising compute_signal."""

    sp = bare_component(slopes_mod.SlopesProcess)
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


@pytest.mark.gpu
@pytest.mark.skipif(not _cuda_available(), reason="CUDA is not available")
def test_gpu_pywfs_cuda_graph_matches_eager_torch_path():
    """The CUDA-graph replay gives the eager torch slopes bit for bit."""
    import torch

    rng = np.random.default_rng(7)
    sp = _pywfs_process("cuda:0", size=128, radius=20)
    indices = [
        torch.as_tensor(np.flatnonzero(mask), device="cuda:0")
        for mask in (sp.p1mask, sp.p2mask, sp.p3mask, sp.p4mask)
    ]

    def eager(image, ref_1d):
        return slopes_mod.compute_slopes_pywfs_torch(
            torch.as_tensor(image, device="cuda:0").reshape(-1),
            *indices,
            sp.num_pixels_in_pupils,
            torch.zeros(2 * sp.num_pixels_in_pupils, device="cuda:0"),
            torch.as_tensor(ref_1d, device="cuda:0"),
        ).cpu()

    image = rng.integers(0, 4000, (128, 128)).astype(np.int32)
    signal, _ = _run_compute_signal(sp, image)
    graph = sp._pywfs_graph[2]
    assert graph is not None, "the CUDA graph should be captured"
    assert torch.equal(torch.as_tensor(signal), eager(image, sp.ref_slopes_1d))

    # GPU-resident frames reuse the graph (same shape and dtype).
    image = rng.integers(0, 4000, (128, 128)).astype(np.int32)
    signal, _ = _run_compute_signal(sp, torch.as_tensor(image, device="cuda:0"))
    assert sp._pywfs_graph[2] is graph
    assert torch.equal(torch.as_tensor(signal), eager(image, sp.ref_slopes_1d))

    # New reference slopes reach the graph without a recapture.
    sp.set_ref_slopes(rng.normal(scale=0.01, size=sp.valid_sub_aps.shape).astype(np.float32))
    signal, _ = _run_compute_signal(sp, image)
    assert sp._pywfs_graph[2] is graph
    assert torch.equal(torch.as_tensor(signal), eager(image, sp.ref_slopes_1d))

    # A dark frame still gives zeros, and a float frame gets its own graph.
    signal, _ = _run_compute_signal(sp, np.zeros_like(image))
    assert np.all(signal == 0.0)
    float_image = rng.random((128, 128)).astype(np.float32) * 100
    signal, _ = _run_compute_signal(sp, float_image)
    assert sp._pywfs_graph[2] is not graph
    assert torch.equal(torch.as_tensor(signal), eager(float_image, sp.ref_slopes_1d))


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
    sp = bare_component(slopes_mod.SlopesProcess)
    sp.signal_type = "phase"
    sp.wfs_type = "shwfs"
    sp._image_buffer = None
    sp.read_stream = lambda *args, **kwargs: np.zeros((4, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="unsupported signal_type"):
        sp.compute_signal()


# --- live instances (real streams, running worker thread) -------------------

LIVE_SIZE = 64
LIVE_LOCS = [(16, 16), (48, 16), (16, 48), (48, 48)]


class _ErrorRecords(logging.Handler):
    def __init__(self):
        super().__init__(logging.ERROR)
        self.records = []

    def emit(self, record):
        self.records.append(record)


@contextlib.contextmanager
def _live_pywfs(gpu_device=None, radius=10):
    """A running PYWFS ``SlopesProcess`` on private streams.

    Yields ``(process, wfs_stream, errors)``; ``errors`` collects ERROR log
    records of the component (worker crashes are logged, not raised).
    """

    wfs = private_stream("wfs", (LIVE_SIZE, LIVE_SIZE), np.float32)
    wfs.write(np.ones((LIVE_SIZE, LIVE_SIZE), dtype=np.float32))
    suffix = uuid.uuid4().hex[:8]
    outputs = {"signal": f"sig_{suffix}", "signal_2d": f"sig2d_{suffix}"}
    conf = {
        "type": "PYWFS",
        "signal_type": "slopes",
        "functions": ["compute_signal"],
        "input_streams": {"wfs": wfs.name},
        "output_streams": outputs,
        "pupils": [f"{x},{y}" for x, y in LIVE_LOCS],  # "x,y" = column,row
        "pupils_radius": radius,
    }
    if gpu_device is not None:
        conf["gpu_device"] = gpu_device
    errors = _ErrorRecords()
    proc = None
    try:
        proc = slopes_mod.SlopesProcess(conf)
        proc.logger.addHandler(errors)
        proc.start()
        yield proc, wfs, errors
    finally:
        if proc is not None:
            proc.logger.removeHandler(errors)
            proc.running = False
            proc.alive = False
            # The worker may be parked in a blocking WFS read: feed it frames
            # until it exits, and only then close the streams it uses.
            deadline = time.monotonic() + 10
            for thread in proc.work_threads:
                while thread.is_alive() and time.monotonic() < deadline:
                    wfs.write(np.ones((LIVE_SIZE, LIVE_SIZE), dtype=np.float32))
                    thread.join(timeout=0.05)
            assert not any(thread.is_alive() for thread in proc.work_threads)
            for stream in (proc.signal, proc.signal_2d, proc.wfs_shm):
                stream.close()
        clear_shms(list(outputs.values()))


def _host(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


_FRAME_IDS = iter(range(10**6, 10**7))


def _publish_and_read(proc, wfs, image, timeout=10.0):
    """Write one WFS frame and return the signal the worker publishes for it.

    The frame id identifies the publication, since the worker may still be
    finishing an earlier frame when this one is written.
    """

    stream = proc.signal
    after = stream.count
    frame_id = next(_FRAME_IDS)
    wfs.write(image, frame_id=frame_id)
    deadline = time.monotonic() + timeout
    while True:
        publication = stream.read_after_publication(
            after, timeout=max(0.0, deadline - time.monotonic())
        )
        if publication.frame_id == frame_id:
            return _host(publication.payload)
        after = publication.count


@contextlib.contextmanager
def _frame_producer(wfs, images):
    """Write ``images`` round-robin to ``wfs`` from a background thread."""

    stop = threading.Event()

    def run():
        i = 0
        while not stop.is_set():
            wfs.write(images[i % len(images)])
            i += 1
            time.sleep(2e-4)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=5)


def _wait_for_new_signals(proc, count, timeout=10.0):
    """Wait until ``proc`` has published ``count`` more signal frames."""

    target = proc.signal.count + count
    deadline = time.monotonic() + timeout
    while proc.signal.count < target:
        assert time.monotonic() < deadline, "the slopes worker stopped publishing"
        time.sleep(1e-3)


def _expected_pywfs(proc, image):
    n = int(np.count_nonzero(proc.p1mask))
    buffers = {k: np.empty(n, dtype=np.float32) for k in ("p1", "p2", "p3", "p4", "tmp1", "tmp2")}
    return slopes_mod.compute_slopes_pywfs_optim_numba(
        image=np.asarray(image, dtype=np.float32).ravel(),
        p1_mask=proc.p1mask.ravel(),
        p2_mask=proc.p2mask.ravel(),
        p3_mask=proc.p3mask.ravel(),
        p4_mask=proc.p4mask.ravel(),
        num_pixels_in_pupils=n,
        slopes=np.zeros(2 * n, dtype=np.float32),
        ref_slopes=proc.ref_slopes_1d,
        **buffers,
    )


def _live_images(seed=0, count=4):
    rng = np.random.default_rng(seed)
    return [
        rng.uniform(100.0, 200.0, (LIVE_SIZE, LIVE_SIZE)).astype(np.float32) for _ in range(count)
    ]


def _check_set_pupils_live(gpu_device):
    images = _live_images()
    with _live_pywfs(gpu_device, radius=10) as (proc, wfs, errors):
        old_n = proc.num_pixels_in_pupils
        signal = _publish_and_read(proc, wfs, images[0])
        np.testing.assert_allclose(signal, _expected_pywfs(proc, images[0]), rtol=1e-5, atol=1e-6)

        # Resize (and later shrink) the pupils while frames keep arriving.
        for radius in (14, 7):
            with _frame_producer(wfs, images):
                time.sleep(0.05)
                proc.set_pupils(list(LIVE_LOCS), radius)
                time.sleep(0.05)

            n = int(np.count_nonzero(proc.p1mask))
            assert n != old_n
            assert proc.num_pixels_in_pupils == n
            for name in ("p1", "p2", "p3", "p4", "tmp1", "tmp2"):
                assert getattr(proc, name).shape == (n,)
            assert proc.slopes_arr_1d.shape == proc.ref_slopes_1d.shape == (2 * n,)
            assert proc.ref_slopes.shape == proc.valid_sub_aps.shape == (2 * radius, 4 * radius)
            assert proc.signal.shape == (2 * n,)
            assert proc.signal_2d.shape == (2 * radius, 4 * radius)

            signal = _publish_and_read(proc, wfs, images[1])
            assert signal.shape == (2 * n,)
            np.testing.assert_allclose(
                signal, _expected_pywfs(proc, images[1]), rtol=1e-5, atol=1e-6
            )
            old_n = n

        assert not errors.records, [r.getMessage() for r in errors.records]


def _check_take_ref_slopes_live(gpu_device):
    image = _live_images(seed=1, count=1)[0]
    with _live_pywfs(gpu_device) as (proc, wfs, errors):
        if gpu_device is not None:
            import torch

            assert isinstance(proc.signal.read(), torch.Tensor)
        proc.set_ref_slopes(np.full(proc.ref_slopes.shape, 0.5, dtype=np.float32))
        proc.ref_slope_count = 5
        with _frame_producer(wfs, [image]):
            # take_ref_slopes averages the next signal frames; wait until the
            # worker is past the setup image, or its last signal from that
            # image can land in the average (seen on a loaded CI runner).
            _wait_for_new_signals(proc, 3)
            proc.take_ref_slopes()

        # The reference is the zero-reference signal of the (constant) frame,
        # not a mix with the old 0.5 reference or the shared 2D buffer.
        expected_1d = _expected_pywfs(proc, image) + proc.ref_slopes_1d
        np.testing.assert_allclose(proc.ref_slopes_1d, expected_1d, rtol=1e-5, atol=1e-6)
        expected_2d = proc.compute_signal_2d(
            expected_1d, out=np.zeros(proc.ref_slopes.shape, dtype=np.float32)
        )
        np.testing.assert_allclose(proc.ref_slopes, expected_2d, rtol=1e-5, atol=1e-6)
        assert np.all(proc.ref_slopes[~proc.valid_sub_aps] == 0.0)

        signal = _publish_and_read(proc, wfs, image)
        np.testing.assert_allclose(signal, 0.0, atol=1e-5)
        assert not errors.records, [r.getMessage() for r in errors.records]


def test_set_pupils_on_live_instance_reallocates_buffers():
    _check_set_pupils_live(None)


def test_take_ref_slopes_on_live_instance():
    _check_take_ref_slopes_live(None)


def test_take_ref_slopes_reads_stream_snapshots_not_shared_buffer():
    """take_ref_slopes must not average through the worker's ``cur_signal_2d``."""

    with _live_pywfs() as (proc, wfs, _):
        proc.running = False  # no worker writes: publish by hand
        time.sleep(0.01)
        frames = [np.full(proc.signal.shape, v, dtype=np.float32) for v in (1.0, 3.0)]
        proc.cur_signal_2d.fill(99.0)
        proc.ref_slope_count = 2

        def publish():
            time.sleep(0.05)
            for frame in frames:
                after = proc.signal.count
                proc.signal.write(frame)
                while proc.signal.count <= after:
                    time.sleep(1e-3)
                time.sleep(0.05)

        thread = threading.Thread(target=publish)
        thread.start()
        proc.take_ref_slopes()
        thread.join()
        np.testing.assert_allclose(proc.ref_slopes_1d, 2.0)
        assert np.all(proc.ref_slopes[proc.valid_sub_aps] == 2.0)
        assert np.all(proc.ref_slopes[~proc.valid_sub_aps] == 0.0)


@pytest.mark.gpu
@pytest.mark.skipif(not _cuda_available(), reason="CUDA is not available")
def test_set_pupils_on_live_gpu_instance_rebuilds_device_cache():
    _check_set_pupils_live("cuda:0")


@pytest.mark.gpu
@pytest.mark.skipif(not _cuda_available(), reason="CUDA is not available")
def test_take_ref_slopes_with_gpu_signal_stream():
    _check_take_ref_slopes_live("cuda:0")


@pytest.mark.parametrize("configured", [True, False])
def test_pywfs_pupils_are_x_y_and_x_slopes_compare_columns(configured):
    """#162: ``pupils`` entries are "x,y" (column,row), and sx compares pupils across columns.

    The default layout follows the documented order too. The image is
    non-square, so a swapped axis cannot pass.
    """

    shape = (40, 64)  # (height, width)
    locs = [(16, 10), (16, 30), (48, 10), (48, 30)]  # (x, y): low x first, then high x
    wfs = private_stream("wfs", shape, np.float32)
    suffix = uuid.uuid4().hex[:8]
    outputs = {"signal": f"sig_{suffix}", "signal_2d": f"sig2d_{suffix}"}
    conf = {
        "type": "PYWFS",
        "signal_type": "slopes",
        "functions": [],
        "input_streams": {"wfs": wfs.name},
        "output_streams": outputs,
    }
    if configured:
        conf.update({"pupils": [f"{x},{y}" for x, y in locs], "pupils_radius": 6})
    wfs.write(np.ones(shape, dtype=np.float32))
    proc = slopes_mod.SlopesProcess(conf)
    try:
        if not configured:
            assert slopes_mod.default_pupil_layout(shape) == (locs, 10)
        assert list(proc.pupil_locs) == locs
        for index, (x, y) in enumerate(locs, start=1):
            assert proc.pupil_mask[y, x] == index  # row y, column x

        def mean_slopes(image):
            wfs.write(image.astype(np.float32))
            proc.compute_signal()
            signal = np.asarray(proc.read(block=False))
            half = signal.size // 2
            return float(np.mean(signal[:half])), float(np.mean(signal[half:]))

        brighter_left = np.where(np.arange(shape[1])[None, :] < 32, 3.0, 1.0) * np.ones(shape)
        assert mean_slopes(brighter_left) == pytest.approx((0.5, 0.0), abs=1e-6)
        brighter_top = np.where(np.arange(shape[0])[:, None] < 20, 3.0, 1.0) * np.ones(shape)
        assert mean_slopes(brighter_top) == pytest.approx((0.0, 0.5), abs=1e-6)
    finally:
        proc.close()
        clear_shms(list(outputs.values()))


def test_pupil_location_parsing():
    assert slopes_mod.parse_pupil_location("30, 10") == (30, 10)
    assert slopes_mod.parse_pupil_location([4, 5]) == (4, 5)
    with pytest.raises(ValueError, match="not 'x,y'"):
        slopes_mod.parse_pupil_location("1,2,3")


def test_shwfs_grid_uses_the_shorter_image_side():
    """A non-square SHWFS image gets as many sub-apertures as its shorter side holds."""

    wfs = private_stream("wfs", (24, 40), np.int32)  # (height, width)
    suffix = uuid.uuid4().hex[:8]
    outputs = {"signal": f"sig_{suffix}", "signal_2d": f"sig2d_{suffix}"}
    proc = slopes_mod.SlopesProcess(
        {
            "type": "SHWFS",
            "signal_type": "slopes",
            "sub_ap_spacing": 8,
            "sub_ap_offset_x": 0,
            "sub_ap_offset_y": 0,
            "functions": [],
            "input_streams": {"wfs": wfs.name},
            "output_streams": outputs,
        }
    )
    try:
        assert proc.num_regions == 3
        assert proc.signal_2d_shape == (6, 3)
    finally:
        proc.close()
        clear_shms(list(outputs.values()))
