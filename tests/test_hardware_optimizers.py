import importlib
import numpy as np


class _Trial:
    def suggest_float(self, name, low, high):
        return (low + high) / 2

    def suggest_int(self, name, low, high):
        return low


class _Study:
    def __init__(self, best_params):
        self.best_params = best_params


class _Stream:
    def __init__(self, value=1.0, shape=(1,), dtype=np.float32):
        self.value = value
        self.shape = shape
        self.dtype = np.dtype(dtype)
        self.writes = []
        self.count = 1
        self.write_time = 1.0

    def read(self):
        return self.value

    def read_new(self, timeout=None):
        return self.value

    def write(self, arr):
        self.writes.append(np.asarray(arr))


class _Loop:
    def __init__(self):
        self.properties = {"running": False, "pGain": 0.1}
        self.calls = []

    def setProperty(self, name, value):
        self.properties[name] = value
        self.calls.append(("set", name, value))

    def getProperty(self, name):
        return self.properties[name]

    def run(self, name):
        self.calls.append(("run", name))


class _Slopes:
    def __init__(self):
        self.properties = {"refSlopesFile": "ref.npy", "validSubApsFile": "valid.npy"}
        self.calls = []

    def setProperty(self, name, value):
        self.properties[name] = value
        self.calls.append(("set", name, value))

    def getProperty(self, name):
        return self.properties[name]

    def run(self, name):
        self.calls.append(("run", name))


def test_pid_optimizer_apply_trial_and_optimum(monkeypatch):
    module = importlib.import_module("pyRTC.hardware.PIDOptimizer")
    monkeypatch.setattr(module, "open_stream", lambda name: _Stream())

    loop = _Loop()
    optimizer = module.PIDOptimizer({"numSteps": 2, "functions": []}, loop)
    optimizer.study = _Study({"pGain": 0.2, "iGain": 0.01, "dGain": 0.02})

    optimizer.applyTrial(_Trial())
    assert loop.properties["pGain"] > 0
    assert loop.properties["iGain"] >= 0
    assert loop.properties["dGain"] >= 0

    optimizer.applyOptimum()
    assert loop.properties["pGain"] == 0.2
    assert loop.properties["iGain"] == 0.01
    assert loop.properties["dGain"] == 0.02


def test_loop_optimizer_apply_trial_and_optimum(monkeypatch):
    module = importlib.import_module("pyRTC.hardware.loopHyperparamsOptimizer")
    monkeypatch.setattr(module, "open_stream", lambda name: _Stream())

    loop = _Loop()
    optimizer = module.loopOptimizer({"numSteps": 2, "functions": []}, loop)
    optimizer.study = _Study({"numDroppedModes": 1, "gain": 0.4, "leakyGain": 0.02})

    optimizer.applyTrial(_Trial())
    assert ("run", "loadIM") in loop.calls

    optimizer.applyOptimum()
    assert loop.properties["numDroppedModes"] == 1
    assert loop.properties["gain"] == 0.4
    assert loop.properties["leakyGain"] == 0.02


def test_ncpa_optimizer_apply_trial_open_loop(monkeypatch):
    module = importlib.import_module("pyRTC.hardware.NCPAOptimizer")
    wfc_stream = _Stream(shape=(6,), dtype=np.float32)

    def _open(name):
        if name == "wfc":
            return wfc_stream
        return _Stream(0.8)

    monkeypatch.setattr(module, "open_stream", _open)

    loop = _Loop()
    slopes = _Slopes()
    optimizer = module.NCPAOptimizer({"numSteps": 2, "functions": [], "endMode": 3}, loop, slopes)
    optimizer.applyTrial(_Trial())

    assert len(wfc_stream.writes) == 1
    assert wfc_stream.writes[0].shape == (6,)