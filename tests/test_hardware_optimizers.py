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
        self.properties = {"running": False, "p_gain": 0.1}
        self.calls = []

    def set_property(self, name, value):
        self.properties[name] = value
        self.calls.append(("set", name, value))

    def get_property(self, name):
        return self.properties[name]

    def run(self, name):
        self.calls.append(("run", name))


class _Slopes:
    def __init__(self):
        self.properties = {"ref_slopes_file": "ref.npy", "valid_sub_aps_file": "valid.npy"}
        self.calls = []

    def set_property(self, name, value):
        self.properties[name] = value
        self.calls.append(("set", name, value))

    def get_property(self, name):
        return self.properties[name]

    def run(self, name):
        self.calls.append(("run", name))


def test_pid_optimizer_apply_trial_and_optimum(monkeypatch):
    module = importlib.import_module("pyrtc.hardware.pid_optimizer")
    monkeypatch.setattr(module, "open_stream", lambda name: _Stream())

    loop = _Loop()
    optimizer = module.PIDOptimizer({"num_steps": 2, "functions": []}, loop)
    optimizer.study = _Study({"p_gain": 0.2, "i_gain": 0.01, "d_gain": 0.02})

    optimizer.apply_trial(_Trial())
    assert loop.properties["p_gain"] > 0
    assert loop.properties["i_gain"] >= 0
    assert loop.properties["d_gain"] >= 0

    optimizer.apply_optimum()
    assert loop.properties["p_gain"] == 0.2
    assert loop.properties["i_gain"] == 0.01
    assert loop.properties["d_gain"] == 0.02


def test_loop_optimizer_apply_trial_and_optimum(monkeypatch):
    module = importlib.import_module("pyrtc.hardware.loop_hyperparams_optimizer")
    monkeypatch.setattr(module, "open_stream", lambda name: _Stream())

    loop = _Loop()
    optimizer = module.LoopOptimizer({"num_steps": 2, "functions": []}, loop)
    optimizer.study = _Study({"num_dropped_modes": 1, "gain": 0.4, "leaky_gain": 0.02})

    optimizer.apply_trial(_Trial())
    assert ("run", "load_im") in loop.calls

    optimizer.apply_optimum()
    assert loop.properties["num_dropped_modes"] == 1
    assert loop.properties["gain"] == 0.4
    assert loop.properties["leaky_gain"] == 0.02


def test_ncpa_optimizer_apply_trial_open_loop(monkeypatch):
    module = importlib.import_module("pyrtc.hardware.ncpa_optimizer")
    wfc_stream = _Stream(shape=(6,), dtype=np.float32)

    def _open(name):
        if name == "wfc":
            return wfc_stream
        return _Stream(0.8)

    monkeypatch.setattr(module, "open_stream", _open)

    loop = _Loop()
    slopes = _Slopes()
    optimizer = module.NCPAOptimizer({"num_steps": 2, "functions": [], "end_mode": 3}, loop, slopes)
    optimizer.apply_trial(_Trial())

    assert len(wfc_stream.writes) == 1
    assert wfc_stream.writes[0].shape == (6,)
