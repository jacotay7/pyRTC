"""The synthetic system with a TorchImageReconstructor in the slopes section.

A PyTorch model replaces SlopesProcess and maps the WFS image straight to
the loop's 97 modes; the loop runs with an identity interaction matrix. The
test checks that the manager builds, validates and runs the system and that
frame ids flow WFS -> signal -> DM command.
"""

import numpy as np
import pytest
import yaml

torch = pytest.importorskip("torch")

from pyrtc import RTCManager, clear_shms  # noqa: E402
from pyrtc.calibration import save_calibration  # noqa: E402
from testsupport import private_synthetic_config  # noqa: E402

MODEL_SOURCE = """
import torch


def build(num_pixels, num_modes, seed=0):
    # A linear map with small weights: the input is flux-normalised (it sums
    # to 1), so every output stays below 1e-3 and the integrator cannot run
    # away, whatever the DM does.
    torch.manual_seed(seed)
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(num_pixels, num_modes))
    with torch.no_grad():
        model[1].weight.uniform_(-1e-3, 1e-3)
        model[1].bias.zero_()
    return model
"""


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=[
                pytest.mark.gpu,
                pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available"),
            ],
        ),
    ],
)
def test_loop_runs_on_a_torch_reconstructor(tmp_path, device):
    config_path, names = private_synthetic_config(tmp_path, include_psf=False)
    config = yaml.safe_load(config_path.read_text())
    num_modes = int(config["wfc"]["num_modes"])
    width, height = int(config["wfs"]["width"]), int(config["wfs"]["height"])

    model_file = tmp_path / "recon_model.py"
    model_file.write_text(MODEL_SOURCE, encoding="utf-8")
    config["slopes"] = {
        "class_name": "TorchImageReconstructor",
        "signal_size": num_modes,
        "model_factory": "build",
        "model_factory_file": str(model_file),
        "model_kwargs": {"num_pixels": width * height, "num_modes": num_modes},
        "flux_normalization": "sum",
        "device": device,
        "input_streams": {"wfs": names["wfs"]},
        "output_streams": {"signal": names["signal"]},
        "functions": ["compute_signal"],
    }
    for key in ("component_classes", "component_files"):
        config["manager"][key].pop("slopes", None)
    # The model outputs modal coefficients, so the IM (and CM) is the identity.
    im_path = tmp_path / "identity_im.npy"
    save_calibration(im_path, np.eye(num_modes, dtype=np.float32), "interaction_matrix")
    config["loop"]["im_file"] = str(im_path)
    config["loop"]["gain"] = 0.15
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    streams = sorted(set(names.values()))
    clear_shms(streams)
    try:
        with RTCManager.from_config_file(config_path) as manager:
            manager.start()
            reconstructor = manager.get_component("slopes")
            loop = manager.get_component("loop")
            assert type(reconstructor).__name__ == "TorchImageReconstructor"
            assert loop.signal_size == num_modes
            np.testing.assert_allclose(loop.cm, np.eye(num_modes), atol=1e-5)

            report = manager.latency(samples=32, timeout_seconds=30.0)
            assert report["stream_path"] == [names["wfs"], names["signal"], names["wfc"]]
            assert report["total"]["alignment"] == "frame_id"

            wfc = manager.get_component("wfc")
            command = np.asarray(wfc.read_stream("wfc", block=False), dtype=np.float64)
            assert np.all(np.isfinite(command))
            assert np.any(command != 0), "the loop never moved the DM"
            assert reconstructor.timing_stats()["count"] > 0
    finally:
        clear_shms(streams)
