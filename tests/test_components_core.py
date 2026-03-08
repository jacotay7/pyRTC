import importlib
from pathlib import Path

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


def test_pyrtc_component_init_and_lifecycle_error_paths(monkeypatch):
    component_module = importlib.import_module("pyRTC.pyRTCComponent")

    def bad_validate(conf, class_names):
        raise RuntimeError("invalid")

    with monkeypatch.context() as mp:
        mp.setattr(component_module, "validate_component_config", bad_validate)
        with pytest.raises(RuntimeError, match="invalid"):
            DummyComponent({"functions": []})

    c = DummyComponent({"functions": []})

    class BadLogger:
        def info(self, message):
            raise RuntimeError("log failed")

        def exception(self, message):
            return None

    c.logger = BadLogger()
    with pytest.raises(RuntimeError, match="log failed"):
        c.start()
    with pytest.raises(RuntimeError, match="log failed"):
        c.stop()

    def bad_stop():
        raise RuntimeError("stop failed")

    c.stop = bad_stop
    c.__del__()
    assert c.alive is False


def test_pyrtc_component_creates_worker_threads(monkeypatch):
    component_module = importlib.import_module("pyRTC.pyRTCComponent")
    created = []

    class FakeThread:
        def __init__(self, target, args, daemon):
            self.target = target
            self.args = args
            self.daemon = daemon
            self.started = False
            created.append(self)

        def start(self):
            self.started = True

    monkeypatch.setattr(component_module.threading, "Thread", FakeThread)
    monkeypatch.setattr(component_module.os, "cpu_count", lambda: 8)

    component = DummyComponent({"affinity": 3, "functions": ["first", "second"]})

    assert len(component.workThreads) == 2
    assert [thread.args[1] for thread in created] == ["first", "second"]
    assert [thread.args[2] for thread in created] == [3, 4]
    assert all(thread.daemon for thread in created)
    assert all(thread.started for thread in created)


def test_pyrtc_component_main_invokes_launch_component(monkeypatch):
    pipeline_module = importlib.import_module("pyRTC.Pipeline")
    component_module = importlib.import_module("pyRTC.pyRTCComponent")
    called = {}

    def fake_launch_component(component_class, component_name, start=False):
        called["component_class"] = component_class
        called["component_name"] = component_name
        called["start"] = start

    monkeypatch.setattr(pipeline_module, "launchComponent", fake_launch_component)
    source = Path(component_module.__file__).read_text(encoding="utf-8")
    exec(compile(source, component_module.__file__, "exec"), {"__name__": "__main__", "__file__": component_module.__file__})

    assert called["component_class"].__name__ == "pyRTCComponent"
    assert called["component_name"] == "component"
    assert called["start"] is True


def test_modulator_start_stop_restart_and_error_paths(monkeypatch):
    modulator_module = importlib.import_module("pyRTC.Modulator")

    modulator = DummyModulator({"functions": []})
    modulator.start()
    assert modulator.running is True
    modulator.stop()
    assert modulator.running is False
    assert modulator.restart() == 1
    assert modulator.restarted is True

    def bad_component_init(self, conf):
        raise RuntimeError("mod init failed")

    monkeypatch.setattr(modulator_module.pyRTCComponent, "__init__", bad_component_init)
    with pytest.raises(RuntimeError, match="mod init failed"):
        DummyModulator({"functions": []})


def test_modulator_start_stop_error_paths(monkeypatch):
    modulator_module = importlib.import_module("pyRTC.Modulator")
    modulator = DummyModulator({"functions": []})

    class QuietLogger:
        def info(self, *args, **kwargs):
            return None

        def exception(self, *args, **kwargs):
            return None

    modulator.logger = QuietLogger()

    def bad_start(self):
        raise RuntimeError("start failed")

    def bad_stop(self):
        raise RuntimeError("stop failed")

    monkeypatch.setattr(modulator_module.pyRTCComponent, "start", bad_start)
    with pytest.raises(RuntimeError, match="start failed"):
        modulator.start()

    monkeypatch.setattr(modulator_module.pyRTCComponent, "stop", bad_stop)
    with pytest.raises(RuntimeError, match="stop failed"):
        modulator.stop()


def test_optimizer_base_methods_and_error_paths(monkeypatch):
    class FakeStudy:
        def __init__(self):
            self.best_params = {}

        def ask(self):
            return {"trial": 1}

        def optimize(self, objective, n_trials):
            for _ in range(n_trials):
                objective()

    monkeypatch.setattr(opt_mod.optuna, "create_study", lambda direction, sampler: FakeStudy())

    optimizer = Optimizer({"numSteps": 1, "functions": []})
    assert optimizer.objective() is None
    assert optimizer.applyOptimum() is None
    assert optimizer.applyTrial({"trial": 2}) is None

    def bad_optimize(objective, n_trials):
        raise RuntimeError("optimize failed")

    optimizer.study.optimize = bad_optimize
    with pytest.raises(RuntimeError, match="optimize failed"):
        optimizer.optimize()

    def bad_ask():
        raise RuntimeError("ask failed")

    optimizer.study.ask = bad_ask
    with pytest.raises(RuntimeError, match="ask failed"):
        optimizer.applyNext()

    monkeypatch.setattr(opt_mod.optuna, "create_study", lambda direction, sampler: (_ for _ in ()).throw(RuntimeError("reset failed")))
    with pytest.raises(RuntimeError, match="reset failed"):
        optimizer.resetStudy()


def test_optimizer_init_failure_logs_and_raises(monkeypatch):
    monkeypatch.setattr(
        opt_mod.optuna,
        "create_study",
        lambda direction, sampler: (_ for _ in ()).throw(RuntimeError("study failed")),
    )

    with pytest.raises(RuntimeError, match="study failed"):
        Optimizer({"functions": []})
