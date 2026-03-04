import numpy as np
import importlib

from pyRTC.Modulator import Modulator
from pyRTC.Optimizer import Optimizer
from pyRTC.pyRTCComponent import pyRTCComponent

opt_mod = importlib.import_module("pyRTC.Optimizer")


class DummyComponent(pyRTCComponent):
    pass


def test_pyrtc_component_start_stop():
    c = DummyComponent({"functions": []})
    assert c.running is False
    c.start()
    assert c.running is True
    c.stop()
    assert c.running is False


def test_modulator_name_default_and_custom():
    m1 = Modulator({})
    assert m1.name == "modulator"
    m2 = Modulator({"name": "m"})
    assert m2.name == "m"


def test_optimizer_apply_next_and_reset_study(monkeypatch):
    class FakeStudy:
        def __init__(self):
            self.asked = False

        def ask(self):
            self.asked = True
            return {"trial": 1}

        def optimize(self, objective, n_trials):
            for _ in range(n_trials):
                objective()

    def fake_create_study(direction, sampler):
        return FakeStudy()

    monkeypatch.setattr(opt_mod.optuna, "create_study", fake_create_study)

    class TOptimizer(Optimizer):
        def __init__(self, conf):
            self.applied = None
            self.objective_calls = 0
            super().__init__(conf)

        def objective(self):
            self.objective_calls += 1
            return 1.0

        def applyTrial(self, trial):
            self.applied = trial

    opt = TOptimizer({"numSteps": 2, "functions": []})
    assert opt.objective() == 1.0
    opt.optimize()
    assert opt.objective_calls >= 2
    opt.applyNext()
    assert opt.applied == {"trial": 1}
    assert opt.applyOptimum() is None
    assert opt.applyTrial({}) is None
    old = opt.study
    opt.resetStudy()
    assert opt.study is not old
