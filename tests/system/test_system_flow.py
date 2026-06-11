import importlib

import numpy as np

from testsupport import DummySHM


loop_mod = importlib.import_module("pyRTC.Loop")
wfc_mod = importlib.import_module("pyRTC.WavefrontCorrector")


def test_loop_wfc_system_flow_smoke(monkeypatch):
    streams = {}

    def _make_stream(name, shape, dtype, gpuDevice=None):
        stream = DummySHM(name, shape, dtype, gpuDevice=gpuDevice)
        streams[name] = stream
        return stream

    def _open_existing(name, gpuDevice=None):
        return streams[name]

    monkeypatch.setattr(wfc_mod, "create_stream", _make_stream)
    monkeypatch.setattr(loop_mod, "open_stream", _open_existing)

    streams["signal"] = DummySHM("signal", (4,), np.float32)

    wfc = wfc_mod.WavefrontCorrector(
        {
            "name": "wfc-smoke",
            "numActuators": 4,
            "numModes": 4,
            "functions": [],
        }
    )

    loop = loop_mod.Loop(
        {
            "functions": [],
            "numDroppedModes": 1,
            "gain": 0.5,
        }
    )

    loop.IM = np.eye(loop.signalSize, loop.numModes, dtype=np.float32)
    loop.computeCM()
    loop.setGain(0.5)

    streams["signal"].write(np.array([0.2, -0.1, 0.4, 0.3], dtype=np.float32))
    loop.standardIntegratorPOL()

    correction = streams["wfc"].read()
    assert correction.shape == (4,)
    assert np.any(np.abs(correction[: loop.numActiveModes]) > 0)
    assert np.allclose(correction[loop.numActiveModes :], 0.0)

    # Ensure objects are referenced so constructors/destructors are exercised in test scope.
    assert wfc is not None
    assert loop is not None
