"""TorchImageReconstructor: a PyTorch model in the slopes section."""

import threading
import time
import uuid

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import pyshmem  # noqa: E402

import pyrtc.image_reconstructor as recon_mod  # noqa: E402
from pyrtc.component_descriptors import (  # noqa: E402
    get_component_descriptor,
    known_config_keys,
    unknown_config_key_warnings,
)
from pyrtc.image_reconstructor import (  # noqa: E402
    TorchImageReconstructor,
    TorchModelRunner,
    load_torch_model,
)
from testsupport import private_stream  # noqa: E402

CUDA = torch.cuda.is_available()
gpu = [pytest.mark.gpu, pytest.mark.skipif(not CUDA, reason="CUDA is not available")]

IMAGE_SHAPE = (8, 8)
SIGNAL_SIZE = 6

FACTORY_SOURCE = """
import torch


class TinyNet(torch.nn.Module):
    def __init__(self, num_outputs=6, side=8):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 3, 3, padding=1)
        self.fc = torch.nn.Linear(3 * side * side, num_outputs)

    def forward(self, x):
        return self.fc(torch.relu(self.conv(x)).flatten(1))


def build(num_outputs=6, side=8, seed=0):
    torch.manual_seed(seed)
    return TinyNet(num_outputs, side)
"""


@pytest.fixture
def factory_file(tmp_path):
    path = tmp_path / f"recon_models_{uuid.uuid4().hex[:6]}.py"
    path.write_text(FACTORY_SOURCE, encoding="utf-8")
    return path


def tiny_net(num_outputs=SIGNAL_SIZE, side=IMAGE_SHAPE[0], seed=0):
    torch.manual_seed(seed)
    return torch.nn.Sequential(
        torch.nn.Conv2d(1, 3, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Flatten(),
        torch.nn.Linear(3 * side * side, num_outputs),
    )


def reference_output(model, image, *, normalization="none", sqrt=False, scale=None):
    """The model applied by hand, in float64 preprocessing then float32."""
    x = np.asarray(image, dtype=np.float64)
    if normalization == "sum":
        x = x / x.sum()
    elif normalization == "mean":
        x = x / x.mean()
    if sqrt:
        x = np.sqrt(np.clip(x, 0, None))
    with torch.no_grad():
        y = model(torch.as_tensor(x, dtype=torch.float32).reshape(1, 1, *x.shape))
    y = y.cpu().numpy().ravel().astype(np.float32)
    if scale is not None:
        y = y * scale
    return y


def random_image(seed=1, shape=IMAGE_SHAPE):
    return np.random.default_rng(seed).integers(1, 1000, size=shape).astype(np.int32)


@pytest.fixture
def make_reconstructor(monkeypatch):
    built = []

    def factory(model=None, wfs=None, **conf_overrides):
        monkeypatch.setattr(recon_mod, "create_stream", private_stream)
        wfs = wfs if wfs is not None else private_stream("wfs", IMAGE_SHAPE, np.int32)
        conf = {"signal_size": SIGNAL_SIZE, "functions": [], "input_streams": {"wfs": wfs.name}}
        conf.update(conf_overrides)
        reconstructor = TorchImageReconstructor(conf, model=model)
        built.append(reconstructor)
        return reconstructor, wfs

    yield factory
    for reconstructor in built:
        reconstructor.close()


# -- config and descriptor ----------------------------------------------------


def test_descriptor_declares_every_config_key():
    descriptor = get_component_descriptor("image_reconstructor")
    assert descriptor.component_class is TorchImageReconstructor
    assert TorchImageReconstructor.describe() is descriptor
    assert descriptor.worker_functions == ("compute_signal",)
    assert [s.name for s in descriptor.input_streams] == ["wfs"]
    assert [s.name for s in descriptor.output_streams] == ["signal", "signal_2d"]
    assert descriptor.required_field_names == ("signal_size",)

    conf = {
        "signal_size": 6,
        "model_file": "m.ts",
        "model_factory": "",
        "model_factory_file": "",
        "model_kwargs": {},
        "state_dict_file": "",
        "device": "cpu",
        "dtype": "float32",
        "input_shape": [1, 1, 8, 8],
        "flux_normalization": "sum",
        "sqrt_stretch": True,
        "output_scale_file": "",
        "signal_2d_shape": [2, 3],
        "cuda_graph": True,
        "warmup_iters": 2,
        "cpu_threads": 1,
        "timing_window": 10,
    }
    assert unknown_config_key_warnings("slopes", conf, TorchImageReconstructor) == []
    # Every key _parse_settings reads is declared.
    settings = TorchImageReconstructor._parse_settings(conf, model_given=False)
    assert set(settings) <= known_config_keys(TorchImageReconstructor)
    assert unknown_config_key_warnings("slopes", {"signal_sise": 6}, TorchImageReconstructor)


@pytest.mark.parametrize(
    "conf, match",
    [
        ({"model_file": "m.ts"}, "signal_size"),
        ({"signal_size": 0, "model_file": "m.ts"}, "signal_size"),
        ({"signal_size": 6}, "exactly one"),
        ({"signal_size": 6, "model_file": "a", "model_factory": "b:c"}, "exactly one"),
        ({"signal_size": 6, "model_file": "a", "dtype": "bfloat16"}, "dtype"),
        ({"signal_size": 6, "model_file": "a", "flux_normalization": "max"}, "flux"),
        ({"signal_size": 6, "model_file": "a", "signal_2d_shape": [2, 2]}, "signal_2d_shape"),
        ({"signal_size": 6, "model_file": "a", "input_shape": [1, -1]}, "input_shape"),
        ({"signal_size": 6, "model_file": "a", "model_kwargs": [1]}, "model_kwargs"),
    ],
)
def test_bad_config_is_rejected_before_any_stream_is_touched(conf, match):
    with pytest.raises(ValueError, match=match):
        TorchImageReconstructor(dict(conf, input_streams={"wfs": "does_not_exist"}))


def test_system_config_accepts_the_reconstructor_in_the_slopes_section(tmp_path, factory_file):
    import yaml

    from pyrtc.config_schema import collect_config_warnings, validate_system_config
    from pyrtc.streams import expected_output_shm_specs_for_config
    from testsupport import private_synthetic_config

    config_path, names = private_synthetic_config(tmp_path, include_psf=False)
    config = yaml.safe_load(config_path.read_text())
    config["slopes"] = {
        "class_name": "TorchImageReconstructor",
        "signal_size": 97,
        "signal_2d_shape": [97, 1],
        "model_factory": "build",
        "model_factory_file": str(factory_file),
        "input_streams": {"wfs": names["wfs"]},
        "output_streams": {"signal": names["signal"], "signal_2d": names["signal_2d"]},
        "functions": ["compute_signal"],
    }
    for key in ("component_classes", "component_files"):
        config["manager"][key].pop("slopes", None)

    normalized = validate_system_config(config)
    assert collect_config_warnings(normalized) == []
    specs = expected_output_shm_specs_for_config(normalized)
    assert specs[names["signal"]]["shape"] == (97,)
    assert specs[names["signal_2d"]]["shape"] == (97, 1)

    config["slopes"]["functions"] = ["no_such_worker"]
    with pytest.raises(Exception, match="no_such_worker"):
        validate_system_config(config)


# -- model loading -------------------------------------------------------------


@pytest.mark.filterwarnings("ignore:`torch.jit:DeprecationWarning")
def test_load_model_from_files_factory_and_state_dict(tmp_path, factory_file):
    reference = tiny_net(seed=3)
    image = random_image()
    expected = reference_output(reference, image)

    script_path = tmp_path / "model.ts"
    torch.jit.save(torch.jit.script(reference), str(script_path))
    scripted = load_torch_model(model_file=str(script_path))
    np.testing.assert_allclose(reference_output(scripted, image), expected, rtol=1e-6)

    export_path = tmp_path / "model.pt2"
    example = (torch.zeros(1, 1, *IMAGE_SHAPE),)
    torch.export.save(torch.export.export(reference.eval(), example), str(export_path))
    exported = load_torch_model(model_file=str(export_path))
    np.testing.assert_allclose(reference_output(exported, image), expected, rtol=1e-6)
    # The runner accepts exported modules (which refuse .eval()).
    runner = TorchModelRunner(exported, image_shape=IMAGE_SHAPE, signal_size=SIGNAL_SIZE)
    np.testing.assert_allclose(runner.run(image), expected, rtol=1e-5)

    # A factory built with another seed, then the reference weights loaded.
    from_file = load_torch_model(model_factory="build", model_factory_file=str(factory_file))
    state_path = tmp_path / "weights.pt"
    torch.save(from_file.state_dict(), state_path)
    loaded = load_torch_model(
        model_factory="build",
        model_factory_file=str(factory_file),
        model_kwargs={"seed": 99},
        state_dict_file=str(state_path),
    )
    image_t = torch.as_tensor(image, dtype=torch.float32).reshape(1, 1, *IMAGE_SHAPE)
    with torch.no_grad():
        torch.testing.assert_close(loaded(image_t), from_file(image_t))


def test_model_factory_resolves_module_function(tmp_path, monkeypatch, factory_file):
    monkeypatch.syspath_prepend(str(factory_file.parent))
    model = load_torch_model(model_factory=f"{factory_file.stem}:build", model_kwargs={"seed": 1})
    assert isinstance(model, torch.nn.Module)
    with pytest.raises(ValueError, match="module:function"):
        load_torch_model(model_factory="build")
    with pytest.raises(TypeError, match="nn.Module"):
        load_torch_model(model_factory="builtins:dict")


# -- runner ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "normalization, sqrt", [("none", False), ("sum", False), ("mean", True), ("sum", True)]
)
def test_runner_matches_the_model_called_directly(normalization, sqrt):
    model = tiny_net()
    scale = np.linspace(0.5, 2.0, SIGNAL_SIZE).astype(np.float32)
    image = random_image()
    expected = reference_output(model, image, normalization=normalization, sqrt=sqrt, scale=scale)
    runner = TorchModelRunner(
        model,
        image_shape=IMAGE_SHAPE,
        image_dtype=np.int32,
        signal_size=SIGNAL_SIZE,
        flux_normalization=normalization,
        sqrt_stretch=sqrt,
        output_scale=scale,
    )
    np.testing.assert_allclose(runner.run(image), expected, rtol=1e-5, atol=1e-6)
    # A torch tensor input and a flat input give the same answer.
    np.testing.assert_allclose(runner.run(torch.from_numpy(image)), expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(runner.run(image.ravel()), expected, rtol=1e-5, atol=1e-6)


def test_dark_frames_give_zeros_when_normalizing():
    runner = TorchModelRunner(
        tiny_net(), image_shape=IMAGE_SHAPE, signal_size=SIGNAL_SIZE, flux_normalization="sum"
    )
    np.testing.assert_array_equal(runner.run(np.zeros(IMAGE_SHAPE)), np.zeros(SIGNAL_SIZE))


def test_input_shape_override_feeds_a_flat_mlp():
    torch.manual_seed(0)
    mlp = torch.nn.Sequential(torch.nn.Linear(64, 16), torch.nn.Tanh(), torch.nn.Linear(16, 4))
    image = random_image().astype(np.float32)
    runner = TorchModelRunner(mlp, image_shape=IMAGE_SHAPE, signal_size=4, input_shape=(1, 64))
    with torch.no_grad():
        expected = mlp(torch.from_numpy(image).reshape(1, 64)).numpy().ravel()
    np.testing.assert_allclose(runner.run(image), expected, rtol=1e-5)


@pytest.mark.parametrize(
    "kwargs, error, match",
    [
        ({"signal_size": SIGNAL_SIZE + 1}, ValueError, "6 elements .* signal_size is 7"),
        ({"input_shape": (1, 1, 4, 4)}, ValueError, "input_shape"),
        ({"input_shape": (1, 2, 4, 8)}, ValueError, "failed on an input of shape"),
        ({"output_scale": np.ones(3)}, ValueError, "output scale"),
        ({"dtype": "float16"}, ValueError, "CUDA"),
        ({"device": "tpu"}, ValueError, "device"),
        ({"flux_normalization": "max"}, ValueError, "flux_normalization"),
    ],
)
def test_runner_rejects_inconsistent_setups(kwargs, error, match):
    options = {"image_shape": IMAGE_SHAPE, "signal_size": SIGNAL_SIZE}
    options.update(kwargs)
    with pytest.raises(error, match=match):
        TorchModelRunner(tiny_net(), **options)


def test_runner_rejects_models_that_do_not_return_one_tensor():
    class TwoHeads(torch.nn.Module):
        def forward(self, x):
            return x.flatten()[:3], x.flatten()[3:6]

    with pytest.raises(TypeError, match="one tensor"):
        TorchModelRunner(TwoHeads(), image_shape=IMAGE_SHAPE, signal_size=SIGNAL_SIZE)
    with pytest.raises(TypeError, match="nn.Module"):
        TorchModelRunner(lambda x: x, image_shape=IMAGE_SHAPE, signal_size=SIGNAL_SIZE)


# -- component on real streams ---------------------------------------------------


def _wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise TimeoutError("condition not met")
        time.sleep(1e-3)


def test_component_publishes_model_output_with_the_wfs_frame_id(make_reconstructor):
    model = tiny_net()
    reconstructor, wfs = make_reconstructor(
        model=model,
        functions=["compute_signal"],
        flux_normalization="sum",
        signal_2d_shape=[2, 3],
    )
    signal = pyshmem.open(reconstructor.signal.name, readonly=True)
    signal_2d = pyshmem.open(reconstructor.signal_2d.name, readonly=True)
    try:
        reconstructor.start()
        count = signal.read_publication().count
        for frame_id in (11, 12, 13):
            image = random_image(seed=frame_id)
            wfs.write(image, frame_id=frame_id)
            # The worker's first read returns the frame already in the stream
            # (no frame id yet), so skip publications from before this frame.
            while True:
                publication = signal.read_after_publication(count, timeout=5.0)
                count = publication.count
                if publication.frame_id in (None, 0, frame_id - 1):
                    continue
                break
            assert publication.frame_id == frame_id
            expected = reference_output(model, image, normalization="sum")
            np.testing.assert_allclose(publication.payload, expected, rtol=1e-5, atol=1e-6)
            _wait_for(lambda: signal_2d.read_publication().frame_id == frame_id)
            np.testing.assert_allclose(signal_2d.read(), expected.reshape(2, 3), rtol=1e-5)
    finally:
        signal.close()
        signal_2d.close()
    assert reconstructor.frames_processed >= 3
    stats = reconstructor.timing_stats()
    assert stats["count"] >= 3
    assert 0 < stats["median"] <= stats["p99"] <= stats["max"]
    assert reconstructor.last_compute_time > 0
    reconstructor.reset_timing()
    assert reconstructor.timing_stats() == {"count": 0}


def test_component_reads_frames_into_the_runner_buffer(make_reconstructor):
    reconstructor, wfs = make_reconstructor(model=tiny_net())
    wfs.write(random_image(), frame_id=5)
    reconstructor.compute_signal()
    # The CPU path reads the frame straight into the runner's input buffer.
    np.testing.assert_array_equal(reconstructor.runner.host_input, random_image())
    assert reconstructor.read(block=False).shape == (SIGNAL_SIZE,)
    np.testing.assert_array_equal(reconstructor.read_image(block=False), random_image())
    assert reconstructor.signal_2d is None


def test_component_rejects_a_model_with_the_wrong_output_size(make_reconstructor):
    before = {thread for thread in threading.enumerate() if thread.name.endswith("(work)")}
    with pytest.raises(ValueError, match="signal_size is 5"):
        make_reconstructor(model=tiny_net(), signal_size=5, functions=["compute_signal"])
    # Worker threads start on start(), so the failed build leaves none (#155).
    after = {thread for thread in threading.enumerate() if thread.name.endswith("(work)")}
    assert after == before


@pytest.mark.filterwarnings("ignore:`torch.jit:DeprecationWarning")
def test_component_loads_the_configured_model_and_scale(make_reconstructor, tmp_path):
    reference = tiny_net(seed=4)
    script_path = tmp_path / "model.ts"
    torch.jit.save(torch.jit.script(reference), str(script_path))
    scale = np.arange(1, SIGNAL_SIZE + 1, dtype=np.float32)
    scale_path = tmp_path / "scale.npy"
    np.save(scale_path, scale)
    reconstructor, wfs = make_reconstructor(
        model_file=str(script_path), output_scale_file=str(scale_path), sqrt_stretch=True
    )
    image = random_image()
    wfs.write(image, frame_id=1)
    reconstructor.compute_signal()
    expected = reference_output(reference, image, sqrt=True, scale=scale)
    np.testing.assert_allclose(reconstructor.read(block=False), expected, rtol=1e-5, atol=1e-6)

    # set_model swaps the network on a live instance.
    other = tiny_net(seed=8)
    reconstructor.set_model(other)
    wfs.write(image, frame_id=2)
    reconstructor.compute_signal()
    expected = reference_output(other, image, sqrt=True, scale=scale)
    np.testing.assert_allclose(reconstructor.read(block=False), expected, rtol=1e-5, atol=1e-6)


# -- GPU -------------------------------------------------------------------------


@pytest.mark.parametrize("dtype, tolerance", [("float32", 1e-5), ("float16", 2e-2)])
@pytest.mark.gpu
@pytest.mark.skipif(not CUDA, reason="CUDA is not available")
def test_cuda_graph_matches_eager_and_cpu(dtype, tolerance):
    model = tiny_net()
    image = random_image()
    cpu = TorchModelRunner(model, image_shape=IMAGE_SHAPE, signal_size=SIGNAL_SIZE).run(image)
    cpu = cpu.copy()
    outputs = {}
    for graph in (False, True):
        runner = TorchModelRunner(
            tiny_net(),
            image_shape=IMAGE_SHAPE,
            image_dtype=np.int32,
            signal_size=SIGNAL_SIZE,
            device="cuda",
            dtype=dtype,
            flux_normalization="sum",
            cuda_graph=graph,
        )
        assert runner.graph_active is graph
        # Several frames, so a graph replaying stale inputs would show.
        outputs[graph] = [runner.run(random_image(seed)).copy() for seed in range(4)]
    for eager, graphed in zip(outputs[False], outputs[True]):
        np.testing.assert_allclose(graphed, eager, rtol=1e-6, atol=1e-6)
    normalized = reference_output(model, image, normalization="sum")
    np.testing.assert_allclose(outputs[True][1], normalized, rtol=tolerance, atol=tolerance)
    assert cpu.shape == (SIGNAL_SIZE,)


@pytest.mark.gpu
@pytest.mark.skipif(not CUDA, reason="CUDA is not available")
def test_cuda_graph_falls_back_to_eager_when_capture_fails(caplog):
    class HostSync(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(64, SIGNAL_SIZE)

        def forward(self, x):
            # .item() synchronises with the host, which capture forbids.
            return self.fc(x.flatten(1)) * (1.0 if x.sum().item() >= 0 else -1.0)

    runner = TorchModelRunner(HostSync(), image_shape=IMAGE_SHAPE, signal_size=6, device="cuda")
    assert not runner.graph_active
    assert "running the model eagerly" in caplog.text
    assert runner.run(random_image()).shape == (SIGNAL_SIZE,)


@pytest.mark.gpu
@pytest.mark.skipif(not CUDA, reason="CUDA is not available")
def test_gpu_streams_end_to_end_with_frame_ids():
    from pyrtc.streams import clear_shms, create_stream

    names = [f"rcg_{uuid.uuid4().hex[:8]}" for _ in range(2)]
    wfs = create_stream(names[0], IMAGE_SHAPE, np.float32, gpu_device="cuda:0")
    model = tiny_net()
    reconstructor = None
    try:
        reconstructor = TorchImageReconstructor(
            {
                "signal_size": SIGNAL_SIZE,
                "device": "cuda:0",
                "gpu_device": "cuda:0",
                "input_streams": {"wfs": names[0]},
                "output_streams": {"signal": names[1]},
            },
            model=model,
        )
        assert reconstructor.runner.graph_active
        assert reconstructor.signal.gpu_device is not None
        image = random_image().astype(np.float32)
        wfs.write(torch.as_tensor(image, device="cuda:0"), frame_id=42)
        reconstructor.compute_signal()
        reader = pyshmem.open(names[1], gpu_device=False, readonly=True)
        try:
            publication = reader.read_publication()
        finally:
            reader.close()
        assert publication.frame_id == 42
        expected = reference_output(tiny_net(), image)
        np.testing.assert_allclose(publication.payload, expected, rtol=1e-4, atol=1e-5)
    finally:
        if reconstructor is not None:
            reconstructor.close()
        wfs.close()
        clear_shms(names)
