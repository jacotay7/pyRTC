import importlib

import pytest

from pyRTC.Modulator import Modulator
from pyRTC.Optimizer import Optimizer
from pyRTC.pyRTCComponent import pyRTCComponent

opt_mod = importlib.import_module("pyRTC.Optimizer")


class DummyComponent(pyRTCComponent):
    pass


class DummyModulator(Modulator):
    def __init__(self, conf):
        self.position = None
        self.restarted = False
        super().__init__(conf)

    def set_position(self, position):
        self.position = tuple(position)
        return 1

    def restart(self):
        self.restarted = True
        return 1


def test_pyrtc_component_start_stop():
    c = DummyComponent({"functions": []})
    assert c.running is False
    c.start()
    assert c.running is True
    c.stop()
    assert c.running is False


def test_modulator_name_default_and_custom():
    with pytest.raises(TypeError):
        Modulator({})

    m1 = DummyModulator({"functions": []})
    assert m1.name == "modulator"
    assert m1.goTo((1, 2)) == 1
    assert m1.position == (1, 2)

    m2 = DummyModulator({"name": "m", "functions": []})
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
