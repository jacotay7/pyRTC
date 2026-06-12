import importlib

import numpy as np

from testsupport import DummySHM


loop_mod = importlib.import_module("pyrtc.loop")
wfc_mod = importlib.import_module("pyrtc.wavefront_corrector")


def test_loop_wfc_system_flow_smoke(monkeypatch):
    streams = {}

    def _make_stream(name, shape, dtype, gpu_device=None):
        stream = DummySHM(name, shape, dtype, gpu_device=gpu_device)
        streams[name] = stream
        return stream

    def _open_existing(name, gpu_device=None):
        return streams[name]

    monkeypatch.setattr(wfc_mod, "create_stream", _make_stream)
    monkeypatch.setattr(loop_mod, "open_stream", _open_existing)

    streams["signal"] = DummySHM("signal", (4,), np.float32)

    wfc = wfc_mod.WavefrontCorrector(
        {
            "name": "wfc-smoke",
            "num_actuators": 4,
            "num_modes": 4,
            "functions": [],
        }
    )

    loop = loop_mod.Loop(
        {
            "functions": [],
            "num_dropped_modes": 1,
            "gain": 0.5,
        }
    )

    loop.im = np.eye(loop.signal_size, loop.num_modes, dtype=np.float32)
    loop.compute_cm()
    loop.set_gain(0.5)

    streams["signal"].write(np.array([0.2, -0.1, 0.4, 0.3], dtype=np.float32))
    loop.standard_integrator_pol()

    correction = streams["wfc"].read()
    assert correction.shape == (4,)
    assert np.any(np.abs(correction[: loop.num_active_modes]) > 0)
    assert np.allclose(correction[loop.num_active_modes :], 0.0)

    # Ensure objects are referenced so constructors/destructors are exercised in test scope.
    assert wfc is not None
    assert loop is not None
