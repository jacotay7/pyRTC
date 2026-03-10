import copy
from pathlib import Path

import numpy as np
import pytest
import yaml

from pyRTC.Pipeline import RTCManager, clear_shms
from pyRTC.config_schema import read_system_config


REPO_ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_CONFIG_PATH = REPO_ROOT / "examples" / "synthetic_shwfs" / "config.yaml"
DEFAULT_STREAMS = [
    "wfs",
    "wfsRaw",
    "wfc",
    "wfc2D",
    "signal",
    "signal2D",
    "psfShort",
    "psfLong",
    "strehl",
    "tiptilt",
]


def _write_runtime_synthetic_config(tmp_path: Path) -> Path:
    config = read_system_config(SYNTHETIC_CONFIG_PATH, validate=False)
    im_path = tmp_path / "synthetic_identity_im.npy"
    np.save(im_path, np.eye(32, dtype=np.float32))
    config["loop"]["IMFile"] = str(im_path)

    config_path = tmp_path / "synthetic_runtime_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def _configure_hard_manager(manager: RTCManager, launcher_cls, *, log_dir=None, manager_overrides=None) -> RTCManager:
    manager.config = copy.deepcopy(manager.config)
    manager.config["manager"] = {
        "mode": "hard-rtc",
        "healthCheckInterval": 60.0,
        "componentClasses": {
            "wfs": "pyRTC.WavefrontSensor",
            "slopes": "pyRTC.SlopesProcess",
            "loop": "pyRTC.Loop",
            "wfc": "pyRTC.WavefrontCorrector",
            "psf": "pyRTC.ScienceCamera",
        },
        "componentFiles": {
            "wfs": str(REPO_ROOT / "pyRTC" / "WavefrontSensor.py"),
            "slopes": str(REPO_ROOT / "pyRTC" / "SlopesProcess.py"),
            "loop": str(REPO_ROOT / "pyRTC" / "Loop.py"),
            "wfc": str(REPO_ROOT / "pyRTC" / "WavefrontCorrector.py"),
            "psf": str(REPO_ROOT / "pyRTC" / "ScienceCamera.py"),
        },
        "ports": {
            "wfs": 5601,
            "slopes": 5602,
            "loop": 5603,
            "wfc": 5604,
            "psf": 5605,
        },
    }
    if log_dir is not None:
        manager.config["manager"]["logDir"] = str(log_dir)
    if manager_overrides:
        manager.config["manager"].update(manager_overrides)
    manager.launcher_cls = launcher_cls
    return manager


def test_manager_launches_soft_synthetic_system(tmp_path):
    clear_shms(DEFAULT_STREAMS)
    manager = RTCManager.from_config_file(_write_runtime_synthetic_config(tmp_path))

    try:
        manager.start()
        status = manager.status()

        assert status["state"] == "running"
        assert status["components"]["loop"]["state"] == "running"
        assert status["components"]["wfs"]["mode"] == "soft-rtc"
    finally:
        manager.stop()
        clear_shms(DEFAULT_STREAMS)


def test_manager_stop_is_idempotent_for_soft_system():
    class FakeRuntime:
        def __init__(self):
            self.calls = 0

        def stop(self):
            self.calls += 1

        def status(self):
            return {"state": "stopped", "mode": "soft-rtc"}

    manager = RTCManager.from_config_file(SYNTHETIC_CONFIG_PATH)
    runtime = FakeRuntime()
    manager.runtimes = {"wfs": runtime}
    manager.state = "running"

    manager.stop()
    manager.stop()

    assert manager.status()["state"] == "stopped"
    assert runtime.calls == 2


def test_manager_mode_override_uses_hard_runtime_with_short_alias(monkeypatch):
    calls = []

    class FakeLauncher:
        def __init__(self, hardwareFile, configFile, port, timeout=None):
            self.hardwareFile = hardwareFile
            self.configFile = configFile
            self.port = port
            self.timeout = timeout

        def launch(self):
            calls.append(("launch", self.hardwareFile, self.port))

        def run(self, function, *args, timeout=None):
            calls.append(("run", function, self.port))
            return 1

        def shutdown(self):
            calls.append(("shutdown", self.hardwareFile, self.port))
            return 1

    manager = RTCManager.from_config_file(SYNTHETIC_CONFIG_PATH, mode="hard", launcher_cls=FakeLauncher)

    manager.start()
    status = manager.status()
    manager.stop()

    assert status["mode"] == "hard-rtc"
    assert status["components"]["loop"]["mode"] == "hard-rtc"
    assert any(entry[:2] == ("run", "start") for entry in calls)
    assert any(entry[0] == "shutdown" for entry in calls)


def test_manager_uses_hard_runtime_with_launcher_integration(monkeypatch):
    calls = []

    class FakeLauncher:
        def __init__(self, hardwareFile, configFile, port, timeout=None):
            self.hardwareFile = hardwareFile
            self.configFile = configFile
            self.port = port
            self.timeout = timeout

        def launch(self):
            calls.append(("launch", self.hardwareFile, self.port))

        def run(self, function, *args, timeout=None):
            calls.append(("run", function, self.port))
            return 1

        def shutdown(self):
            calls.append(("shutdown", self.hardwareFile, self.port))
            return 1

    manager = RTCManager.from_config_file(SYNTHETIC_CONFIG_PATH, launcher_cls=FakeLauncher)
    manager.config = copy.deepcopy(manager.config)
    manager.config["manager"] = {
        "mode": "hard-rtc",
        "componentClasses": {
            "wfs": "pyRTC.WavefrontSensor",
            "slopes": "pyRTC.SlopesProcess",
            "loop": "pyRTC.Loop",
            "wfc": "pyRTC.WavefrontCorrector",
            "psf": "pyRTC.ScienceCamera",
        },
        "componentFiles": {
            "wfs": str(REPO_ROOT / "pyRTC" / "WavefrontSensor.py"),
            "slopes": str(REPO_ROOT / "pyRTC" / "SlopesProcess.py"),
            "loop": str(REPO_ROOT / "pyRTC" / "Loop.py"),
            "wfc": str(REPO_ROOT / "pyRTC" / "WavefrontCorrector.py"),
            "psf": str(REPO_ROOT / "pyRTC" / "ScienceCamera.py"),
        },
        "ports": {
            "wfs": 5601,
            "slopes": 5602,
            "loop": 5603,
            "wfc": 5604,
            "psf": 5605,
        },
    }

    manager.start()
    status = manager.status()
    manager.stop()

    assert status["state"] == "running"
    assert status["components"]["loop"]["mode"] == "hard-rtc"
    assert status["components"]["loop"]["port"] == 5603
    assert any(entry[:2] == ("launch", str(REPO_ROOT / "pyRTC" / "Loop.py")) for entry in calls)
    assert any(entry[:2] == ("run", "start") for entry in calls)
    assert any(entry[0] == "shutdown" for entry in calls)


def test_manager_requires_config_path_for_hard_mode_from_dict():
    manager = RTCManager.from_config(
        {
            "wfs": {"name": "wavefrontSensor", "width": 16, "height": 16, "darkCount": 1, "functions": ["expose"]},
            "slopes": {"type": "SHWFS", "signalType": "slopes", "subApSpacing": 8, "subApOffsetX": 0, "subApOffsetY": 0, "functions": ["computeSignal"]},
            "wfc": {"name": "dm", "numActuators": 8, "numModes": 8, "functions": ["sendToHardware"]},
            "loop": {"gain": 0.1, "numDroppedModes": 0, "functions": ["standardIntegrator"]},
            "manager": {
                "mode": "hard-rtc",
                "componentClasses": {
                    "wfs": "pyRTC.WavefrontSensor",
                    "slopes": "pyRTC.SlopesProcess",
                    "loop": "pyRTC.Loop",
                    "wfc": "pyRTC.WavefrontCorrector",
                },
            },
        }
    )

    with pytest.raises(ValueError, match="config_path"):
        manager.start()


def test_manager_supports_explicit_manager_declared_sections():
    calls = []

    class FakeLauncher:
        def __init__(self, hardwareFile, configFile, port, timeout=None):
            self.hardwareFile = hardwareFile
            self.configFile = configFile
            self.port = port
            self.timeout = timeout

        def launch(self):
            calls.append(("launch", self.hardwareFile, self.port))

        def run(self, function, *args, timeout=None):
            calls.append(("run", function, self.port))
            return 1

        def shutdown(self):
            calls.append(("shutdown", self.hardwareFile, self.port))
            return 1

    manager = RTCManager.from_config_file(SYNTHETIC_CONFIG_PATH, launcher_cls=FakeLauncher)
    manager.config = copy.deepcopy(manager.config)
    manager.config["modulator"] = {"name": "tutorial-modulator", "frequency": 300, "amplitude": 600}
    manager.config["manager"] = {
        "mode": "hard-rtc",
        "componentClasses": {
            "modulator": "pyRTC.pyRTCComponent.pyRTCComponent",
            "wfs": "pyRTC.WavefrontSensor",
            "slopes": "pyRTC.SlopesProcess",
            "loop": "pyRTC.Loop",
            "wfc": "pyRTC.WavefrontCorrector",
            "psf": "pyRTC.ScienceCamera",
        },
        "componentFiles": {
            "modulator": str(REPO_ROOT / "pyRTC" / "pyRTCComponent.py"),
            "wfs": str(REPO_ROOT / "pyRTC" / "WavefrontSensor.py"),
            "slopes": str(REPO_ROOT / "pyRTC" / "SlopesProcess.py"),
            "loop": str(REPO_ROOT / "pyRTC" / "Loop.py"),
            "wfc": str(REPO_ROOT / "pyRTC" / "WavefrontCorrector.py"),
            "psf": str(REPO_ROOT / "pyRTC" / "ScienceCamera.py"),
        },
        "ports": {
            "modulator": 5600,
            "wfs": 5601,
            "slopes": 5602,
            "loop": 5603,
            "wfc": 5604,
            "psf": 5605,
        },
    }

    manager.start()
    manager.stop()

    assert any(entry[:2] == ("launch", str(REPO_ROOT / "pyRTC" / "pyRTCComponent.py")) for entry in calls)


def test_manager_status_includes_health_metadata_for_hard_runtime(tmp_path):
    class HealthLauncher:
        def __init__(self, hardwareFile, configFile, port, timeout=None):
            self.hardwareFile = hardwareFile
            self.configFile = configFile
            self.port = port
            self.timeout = timeout
            self.pid = 7000 + port
            self.health_state = "running"
            self.last_contact_time = 100.0

        def launch(self):
            return 1

        def run(self, function, *args, timeout=None):
            return 1

        def shutdown(self):
            return 1

        def close(self, force=False):
            return None

        def health_check(self, timeout=None):
            self.last_contact_time += 1.0
            return {
                "state": self.health_state,
                "pid": self.pid,
                "last_contact_time": self.last_contact_time,
                "error": None,
            }

    manager = _configure_hard_manager(
        RTCManager.from_config_file(SYNTHETIC_CONFIG_PATH, launcher_cls=HealthLauncher),
        HealthLauncher,
        log_dir=tmp_path,
    )

    manager.start()
    try:
        status = manager.status()
    finally:
        manager.stop()

    loop_status = status["components"]["loop"]
    assert status["state"] == "running"
    assert loop_status["pid"] == 12603
    assert loop_status["start_time"] is not None
    assert loop_status["uptime_seconds"] >= 0.0
    assert loop_status["last_heartbeat_time"] == 101.0
    assert loop_status["last_success_time"] == 101.0
    assert loop_status["restart_count"] == 0
    assert loop_status["restart_policy"] == "never"
    assert loop_status["log_file"].endswith("pyrtc-loop_loop_12603.log")


def test_manager_marks_component_degraded_when_health_check_fails():
    class DegradedLauncher:
        def __init__(self, hardwareFile, configFile, port, timeout=None):
            self.port = port
            self.health_state = "running"
            self.pid = 8000 + port

        def launch(self):
            return 1

        def run(self, function, *args, timeout=None):
            return 1

        def shutdown(self):
            return 1

        def close(self, force=False):
            return None

        def health_check(self, timeout=None):
            if self.health_state == "degraded":
                return {
                    "state": "degraded",
                    "pid": self.pid,
                    "last_contact_time": 50.0,
                    "error": "health check RPC failed",
                }
            return {
                "state": "running",
                "pid": self.pid,
                "last_contact_time": 50.0,
                "error": None,
            }

    manager = _configure_hard_manager(
        RTCManager.from_config_file(SYNTHETIC_CONFIG_PATH, launcher_cls=DegradedLauncher),
        DegradedLauncher,
    )

    manager.start()
    try:
        manager.get_component("loop").health_state = "degraded"
        status = manager.status()
    finally:
        manager.stop()

    assert status["state"] == "degraded"
    assert status["components"]["loop"]["state"] == "degraded"
    assert "health check RPC failed" in status["components"]["loop"]["error"]


def test_manager_restarts_failed_child_when_policy_is_on_failure():
    class RestartingLauncher:
        launches = 0
        loop_failed_once = False

        def __init__(self, hardwareFile, configFile, port, timeout=None):
            type(self).launches += 1
            self.port = port
            self.pid = 9000 + type(self).launches
            self.fail_health_check = port == 5603 and not type(self).loop_failed_once
            if self.fail_health_check:
                type(self).loop_failed_once = True

        def launch(self):
            return 1

        def run(self, function, *args, timeout=None):
            return 1

        def shutdown(self):
            return 1

        def close(self, force=False):
            return None

        def health_check(self, timeout=None):
            if self.fail_health_check:
                return {
                    "state": "failed",
                    "pid": self.pid,
                    "last_contact_time": 25.0,
                    "error": "child process exited with code 1",
                }
            return {
                "state": "running",
                "pid": self.pid,
                "last_contact_time": 26.0,
                "error": None,
            }

    manager = _configure_hard_manager(
        RTCManager.from_config_file(SYNTHETIC_CONFIG_PATH, launcher_cls=RestartingLauncher),
        RestartingLauncher,
        manager_overrides={"restartPolicy": "on-failure"},
    )

    manager.start()
    try:
        status = manager.refresh_health()
    finally:
        manager.stop()

    loop_status = status["components"]["loop"]
    assert status["state"] == "running"
    assert loop_status["state"] == "running"
    assert loop_status["restart_count"] == 1
    assert loop_status["last_error"] == "child process exited with code 1"
    assert RestartingLauncher.launches >= 6


def test_manager_repeated_failures_increment_restart_count_and_preserve_last_error():
    class FlappingLauncher:
        launches = 0

        def __init__(self, hardwareFile, configFile, port, timeout=None):
            type(self).launches += 1
            self.port = port
            self.pid = 10000 + type(self).launches

        def launch(self):
            return 1

        def run(self, function, *args, timeout=None):
            return 1

        def shutdown(self):
            return 1

        def close(self, force=False):
            return None

        def health_check(self, timeout=None):
            return {
                "state": "failed",
                "pid": self.pid,
                "last_contact_time": 75.0,
                "error": "child process exited with code 2",
            }

    manager = _configure_hard_manager(
        RTCManager.from_config_file(SYNTHETIC_CONFIG_PATH, launcher_cls=FlappingLauncher),
        FlappingLauncher,
        manager_overrides={"restartPolicy": "on-failure"},
    )

    manager.start()
    try:
        manager.refresh_health()
        status = manager.refresh_health()
    finally:
        manager.stop()

    loop_status = status["components"]["loop"]
    assert loop_status["restart_count"] == 2
    assert loop_status["last_error"] == "child process exited with code 2"
    assert loop_status["state"] == "running"