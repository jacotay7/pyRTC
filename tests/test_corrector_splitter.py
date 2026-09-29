"""Woofer/tweeter splitting and offloading (#58)."""

import numpy as np
import pytest

import pyrtc.corrector_splitter as splitter_mod
from pyrtc.corrector_splitter import CorrectorSplitter
from testsupport import private_stream


@pytest.fixture
def make_splitter(monkeypatch):
    created = []

    def factory(correctors=None, offload=None):
        monkeypatch.setattr(splitter_mod, "create_stream", private_stream)
        outputs = {}
        conf = {
            "name": "wfc",
            "functions": [],
            "correctors": correctors
            or [
                {"name": "woofer", "stream": "woofer_wfc", "modes": 2},
                {"name": "tweeter", "stream": "tweeter_wfc", "modes": 3},
            ],
        }
        if offload is not None:
            conf["offload"] = offload
        for entry in conf["correctors"]:
            outputs[entry["stream"]] = private_stream(entry["stream"], (entry["modes"],), "float32")
        monkeypatch.setattr(splitter_mod, "open_stream", lambda name: _reader(outputs[name]))
        splitter = CorrectorSplitter(conf)
        created.append((splitter, outputs))
        return splitter, outputs

    yield factory
    for splitter, outputs in created:
        splitter.close()
        for stream in outputs.values():
            stream.close()


def _reader(stream):
    import pyshmem

    return pyshmem.open(stream.name)


def test_split_forwards_each_correctors_slice(make_splitter):
    splitter, outputs = make_splitter()
    assert splitter.num_modes == 5
    splitter.write_stream("wfc", np.arange(5, dtype=np.float32))
    splitter.split()
    np.testing.assert_array_equal(outputs["woofer_wfc"].read(), [0, 1])
    np.testing.assert_array_equal(outputs["tweeter_wfc"].read(), [2, 3, 4])


def test_offload_preserves_the_wavefront_and_moves_content_to_the_target(make_splitter):
    splitter, _ = make_splitter(offload={"source": "tweeter", "target": "woofer", "gain": 0.2})
    # Woofer mode 0 equals tweeter mode 0; woofer mode 1 equals tweeter modes 1+2.
    coupling = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    assert splitter.offload_gain == 0.0  # configured gain waits for the coupling
    splitter.set_coupling(coupling)
    assert splitter.offload_gain == 0.2
    command = np.array([0.0, 0.0, 1.0, 0.5, 0.5])  # all content on the tweeter
    loop_equivalent = command[2:] + coupling @ command[:2]
    for _ in range(100):
        parts = splitter.split_command(command)
        # The total, in tweeter-mode units, never changes.
        np.testing.assert_allclose(parts["tweeter"] + coupling @ parts["woofer"], loop_equivalent)
    # The woofer-representable content has moved to the woofer.
    np.testing.assert_allclose(parts["woofer"], [1.0, 0.5], atol=1e-6)
    np.testing.assert_allclose(parts["tweeter"], [0.0, 0.0, 0.0], atol=1e-6)


def test_offload_leaves_what_the_target_cannot_represent(make_splitter):
    splitter, _ = make_splitter(offload={"source": "tweeter", "target": "woofer"})
    splitter.set_coupling(np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]))
    splitter.set_offload_gain(0.5)
    command = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
    for _ in range(60):
        parts = splitter.split_command(command)
    np.testing.assert_allclose(parts["woofer"], [1.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(parts["tweeter"], [0.0, 0.0, 1.0], atol=1e-6)


def test_coupling_from_the_interaction_matrix(make_splitter):
    splitter, _ = make_splitter(offload={"source": "tweeter", "target": "woofer"})
    rng = np.random.default_rng(0)
    im_tweeter = rng.standard_normal((40, 3))
    true_coupling = np.array([[1.0, 0.2], [0.0, 0.8], [0.3, 0.0]])
    im_woofer = im_tweeter @ true_coupling  # each woofer mode looks like a tweeter combination
    coupling = splitter.set_coupling_from_im(np.hstack((im_woofer, im_tweeter)))
    np.testing.assert_allclose(coupling, true_coupling, atol=1e-10)


def test_offload_needs_a_coupling_and_is_off_by_default(make_splitter):
    splitter, _ = make_splitter(offload={"source": "tweeter", "target": "woofer"})
    assert splitter.offload_gain == 0.0
    with pytest.raises(RuntimeError, match="coupling"):
        splitter.set_offload_gain(0.1)
    parts = splitter.split_command(np.arange(5.0))
    np.testing.assert_array_equal(parts["tweeter"], [2, 3, 4])  # untouched


def test_tip_tilt_offload_to_a_two_mode_stage(make_splitter):
    splitter, _ = make_splitter(
        correctors=[
            {"name": "tt", "stream": "tt_wfc", "modes": 2},
            {"name": "dm", "stream": "dm_wfc", "modes": 4},
        ],
        offload={
            "source": "dm",
            "target": "tt",
            "gain": 0.3,
            "coupling": [[2, 0], [0, 2], [0, 0], [0, 0]],
        },
    )
    command = np.array([0.0, 0.0, 0.4, -0.2, 0.1, 0.1])
    for _ in range(80):
        parts = splitter.split_command(command)
    # DM tip/tilt (modes 0, 1) moved to the stage, scaled by the coupling.
    np.testing.assert_allclose(parts["tt"], [0.2, -0.1], atol=1e-6)
    np.testing.assert_allclose(parts["dm"], [0.0, 0.0, 0.1, 0.1], atol=1e-6)
    splitter.reset_offload()
    np.testing.assert_allclose(splitter.split_command(command)["tt"], [0.0, 0.0])


@pytest.mark.parametrize(
    ("correctors", "match"),
    [
        ([], "non-empty"),
        ([{"stream": "a"}], "'stream' and 'modes'"),
        ([{"stream": "a", "modes": 0}], ">= 1"),
        ([{"stream": "a", "modes": 1}, {"stream": "a", "modes": 1}], "duplicate"),
    ],
)
def test_corrector_list_is_validated(correctors, match):
    with pytest.raises(ValueError, match=match):
        CorrectorSplitter._parse_correctors(correctors)


def test_missing_corrector_streams_are_skipped(monkeypatch, caplog):
    monkeypatch.setattr(splitter_mod, "create_stream", private_stream)

    def missing(name):
        raise FileNotFoundError(name)

    monkeypatch.setattr(splitter_mod, "open_stream", missing)
    splitter = CorrectorSplitter(
        {"name": "wfc", "functions": [], "correctors": [{"stream": "nowhere", "modes": 2}]}
    )
    try:
        splitter.write_stream("wfc", np.ones(2, dtype=np.float32))
        splitter.split()
        splitter.write_stream("wfc", np.ones(2, dtype=np.float32))
        splitter.split()
        assert caplog.text.count("does not exist yet") == 1
    finally:
        splitter.close()


def test_loop_converges_through_the_splitter_under_the_manager(tmp_path):
    """Manager build order, stream wiring and closed loop via a splitter (#58)."""
    import time

    import yaml

    from pyrtc import RTCManager, clear_shms
    from pyrtc.streams import open_stream
    from testsupport import private_synthetic_config

    config_path, names = private_synthetic_config(tmp_path, include_psf=False)
    config = yaml.safe_load(open(config_path))
    dm_stream = names["wfc"] + "_dm"
    # The synthetic DM moves to its own section and stream; the splitter takes
    # over the loop's `wfc` section and forwards to it.
    dm = config.pop("wfc")
    dm["input_streams"] = {"wfc": dm_stream}
    dm["output_streams"] = {"wfc": dm_stream, "wfc_2d": names["wfc_2d"]}
    config["dm"] = dm
    num_modes = int(dm["num_modes"])
    config["wfc"] = {
        "class_name": "pyrtc.corrector_splitter.CorrectorSplitter",
        "correctors": [{"name": "dm", "stream": dm_stream, "modes": num_modes}],
        "input_streams": {"wfc": names["wfc"]},
        "output_streams": {"wfc": names["wfc"]},
        "functions": ["split"],
    }
    config["wfs"]["input_streams"]["wfc"] = dm_stream  # the WFS sees the DM, not the loop
    config["loop"]["num_iters_im"] = 400  # DOCRIME needs enough frames
    for key in ("component_classes", "component_files"):
        config.get("manager", {}).get(key, {}).pop("wfc", None)
    yaml.safe_dump(config, open(config_path, "w"))
    streams = sorted(set(names.values()) | {dm_stream})
    clear_shms(streams)
    try:
        with RTCManager.from_config_file(config_path) as manager:
            manager.start()
            loop = manager.get_component("loop")
            signal = open_stream(names["signal"], readonly=True)
            try:
                loop.stop()
                loop.flatten()
                time.sleep(0.3)
                open_loop = float(np.sqrt(np.mean(signal.read() ** 2)))
                loop.compute_im()  # every poke goes loop -> splitter -> DM -> WFS
                loop.start()
                deadline = time.monotonic() + 20
                while True:
                    residual = float(np.sqrt(np.mean(signal.read() ** 2)))
                    if residual < 0.5 * open_loop or time.monotonic() > deadline:
                        break
                    time.sleep(0.3)
                loop.stop()
            finally:
                signal.close()
            assert manager.get_component("wfc").num_modes == num_modes
    finally:
        clear_shms(streams)
    assert open_loop > 0
    assert residual < 0.5 * open_loop, (open_loop, residual)
