import copy
import socket
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

from testsupport import private_synthetic_config, publishing_chain
from pyrtc.manager import HardComponentRuntime, RTCManager
from pyrtc.rpc import _socket_read_json, _socket_send_json
from pyrtc.streams import (
    clear_shms,
    create_stream,
    open_stream,
    expected_output_shm_specs_for_config,
    reconcile_expected_output_shms,
)
from pyrtc.config_schema import read_system_config
from pyrtc.hardware.synthetic_systems import (
    _default_wfc_layout,
    build_synthetic_shwfs_response_matrix,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_CONFIG_PATH = REPO_ROOT / "examples" / "synthetic_shwfs" / "config.yaml"


def _write_runtime_synthetic_config(tmp_path: Path) -> Path:
    config = read_system_config(SYNTHETIC_CONFIG_PATH, validate=False)
    im_path = tmp_path / "synthetic_identity_im.npy"
    layout = _default_wfc_layout(int(config["wfc"]["num_actuators"]))
    response = build_synthetic_shwfs_response_matrix(7, int(config["wfc"]["num_modes"]), layout)
    np.save(im_path, response.astype(np.float32))
    config["loop"]["im_file"] = str(im_path)

    config_path = tmp_path / "synthetic_runtime_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def _configure_hard_manager(
    manager: RTCManager, launcher_cls, *, log_dir=None, manager_overrides=None
) -> RTCManager:
    manager.config = copy.deepcopy(manager.config)
    manager.config["manager"] = {
        "mode": "hard-rtc",
        "health_check_interval": 60.0,
        "component_classes": {
            "wfs": "pyrtc.wavefront_sensor.WavefrontSensor",
            "slopes": "pyrtc.slopes_process.SlopesProcess",
            "loop": "pyrtc.loop.Loop",
            "wfc": "pyrtc.wavefront_corrector.WavefrontCorrector",
            "psf": "pyrtc.science_camera.ScienceCamera",
        },
        "component_files": {
            "wfs": str(REPO_ROOT / "pyrtc" / "wavefront_sensor.py"),
            "slopes": str(REPO_ROOT / "pyrtc" / "slopes_process.py"),
            "loop": str(REPO_ROOT / "pyrtc" / "loop.py"),
            "wfc": str(REPO_ROOT / "pyrtc" / "wavefront_corrector.py"),
            "psf": str(REPO_ROOT / "pyrtc" / "science_camera.py"),
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
        manager.config["manager"]["log_dir"] = str(log_dir)
    if manager_overrides:
        manager.config["manager"].update(manager_overrides)
    manager.launcher_cls = launcher_cls
    return manager


@pytest.fixture(scope="module")
def private_config_path(tmp_path_factory):
    """The synthetic config with private stream names.

    ``build()`` reconciles (and may clear) every output stream the config
    names, so tests never point it at the canonical names another system on
    the host may be using.
    """
    config_path, _names = private_synthetic_config(tmp_path_factory.mktemp("manager_config"))
    return config_path


@pytest.fixture
def private_system(tmp_path):
    """Yield ``(config_path, names)`` for a runnable private synthetic system."""
    config_path, names = private_synthetic_config(tmp_path)
    streams = sorted(set(names.values()))
    clear_shms(streams)
    yield config_path, names
    clear_shms(streams)


def test_manager_launches_soft_synthetic_system(private_system):
    config_path, _names = private_system

    with RTCManager.from_config_file(config_path) as manager:
        manager.start()
        status = manager.status()

        assert status["state"] == "running"
        assert status["components"]["loop"]["state"] == "running"
        assert status["components"]["wfs"]["mode"] == "soft-rtc"

    assert manager.state == "closed"
    assert manager.runtimes == {}


def test_manager_start_clears_stale_output_shms(private_system):
    config_path, names = private_system
    # Leave mismatched segments behind the way an exited run would: no handle
    # stays open (on Windows the names then vanish, as they would in practice).
    create_stream(names["wfc"], (1,), np.int8).close()
    create_stream(names["signal"], (1,), np.int8).close()
    manager = RTCManager.from_config_file(config_path)

    try:
        manager.start()
        status = manager.status()
        assert status["state"] == "running"
        assert status["components"]["loop"]["state"] == "running"
    finally:
        manager.close()


def test_manager_build_creates_components_before_start(private_system):
    config_path, _names = private_system
    manager = RTCManager.from_config_file(config_path)

    try:
        status = manager.build()

        assert status["state"] == "built"
        assert manager.get_component("wfs") is not None
        assert status["components"]["wfs"]["state"] == "built"

        manager.start()
        running_status = manager.status()
        assert running_status["state"] == "running"

        manager.stop()
        built_status = manager.status()
        assert built_status["state"] == "built"
    finally:
        manager.close()


def _registered_handles(component):
    return list(component._stream_inputs.values()) + list(component._stream_outputs.values())


def test_manager_close_ends_workers_and_closes_stream_handles(private_system):
    config_path, _names = private_system
    manager = RTCManager.from_config_file(config_path)
    manager.start()
    components = {section: manager.get_component(section) for section in manager.runtimes}
    threads = [thread for comp in components.values() for thread in comp.work_threads]
    handles = [handle for comp in components.values() for handle in _registered_handles(comp)]
    assert threads and all(thread.is_alive() for thread in threads)
    assert handles

    manager.stop()
    # stop() only pauses: workers and handles stay alive for start().
    assert all(thread.is_alive() for thread in threads)
    assert all(comp.alive for comp in components.values())

    manager.close()

    assert manager.state == "closed"
    assert manager.runtimes == {}
    assert not any(thread.is_alive() for thread in threads)
    assert not any(comp.alive for comp in components.values())
    for handle in handles:
        with pytest.raises(RuntimeError, match="closed shared memory"):
            handle.read()
    manager.close()  # idempotent
    assert manager.state == "closed"


def test_manager_close_releases_blocked_readers_while_running(private_system):
    config_path, _names = private_system
    manager = RTCManager.from_config_file(config_path)
    manager.start()
    loop = manager.get_component("loop")
    # Stop the producers only: the loop's worker now blocks on the signal.
    for section in ("wfs", "slopes"):
        manager.stop_component(section)

    started = time.monotonic()
    manager.close()

    assert time.monotonic() - started < 5.0
    assert not any(thread.is_alive() for thread in loop.work_threads)


def test_manager_can_start_again_after_close(private_system):
    config_path, _names = private_system
    manager = RTCManager.from_config_file(config_path)
    manager.start()
    first_loop = manager.get_component("loop")
    manager.close()

    try:
        manager.start()
        assert manager.status()["state"] == "running"
        assert manager.get_component("loop") is not first_loop
    finally:
        manager.close()


def test_frame_ids_propagate_through_the_synthetic_chain(private_system):
    """Every registered input carries the WFS frame id down to the DM (#35)."""
    config_path, names = private_system
    with RTCManager.from_config_file(config_path) as manager:
        manager.start()
        # Straight after start: latency() waits for the chain to be live (#112).
        report = manager.latency(
            stream_path=[names["wfs"], names["signal"], names["wfc"]],
            samples=32,
            timeout_seconds=30.0,
        )
        observers = {name: open_stream(names[name], readonly=True) for name in names}
        try:
            # Read the WFS last: every downstream frame id was produced first.
            order = sorted(observers, key=lambda name: name == "wfs")
            publications = {name: observers[name].read_publication() for name in order}
        finally:
            for shm in observers.values():
                shm.close()

    assert report["total"]["alignment"] == "frame_id"
    assert [segment["alignment"] for segment in report["segments"]] == ["frame_id", "frame_id"]
    # Downstream streams carry ids of frames the WFS already produced.
    wfs_id = publications["wfs"].frame_id
    assert wfs_id is not None and wfs_id > 0
    for name in ("signal", "signal_2d", "wfc", "wfc_2d"):
        frame_id = publications[name].frame_id
        assert frame_id is not None and 0 < frame_id <= wfs_id, name
    # The WFS writes wfs_raw just before wfs, so it can be one frame ahead.
    raw_id = publications["wfs_raw"].frame_id
    assert raw_id is not None and 0 < raw_id <= wfs_id + 1
    # The science camera reads the signal (a registered input) and stamps it.
    assert publications["strehl"].frame_id is not None


def test_latency_infers_the_configured_stream_names(private_system):
    """Without stream_path, latency() follows the renamed streams (#119)."""
    config_path, names = private_system
    with RTCManager.from_config_file(config_path) as manager:
        manager.start()
        report = manager.latency(samples=16, timeout_seconds=30.0)

    assert report["inferred_path"] is True
    assert report["stream_path"][0] == names["wfs"]
    assert set(report["stream_path"]) <= set(names.values())
    assert report["total"]["alignment"] == "frame_id"


def test_reconcile_expected_output_shms_reuses_matching_streams(monkeypatch):
    config = read_system_config(SYNTHETIC_CONFIG_PATH, validate=False)
    specs = expected_output_shm_specs_for_config(config)
    cleared = []

    monkeypatch.setattr(
        "pyrtc.streams._existing_shm_spec",
        lambda name: (
            (tuple(specs[name]["shape"]), np.dtype(specs[name]["dtype"])) if name in specs else None
        ),
    )
    monkeypatch.setattr("pyrtc.streams.clear_shms", lambda names: cleared.extend(names))

    rebuilt, reused = reconcile_expected_output_shms(config)

    assert rebuilt == []
    assert "wfc" in reused
    assert "signal" in reused
    assert cleared == []


def test_reconcile_expected_output_shms_clears_only_mismatched_streams(monkeypatch):
    config = read_system_config(SYNTHETIC_CONFIG_PATH, validate=False)
    specs = expected_output_shm_specs_for_config(config)
    cleared = []

    def _existing(name):
        if name == "wfc":
            return ((1,), np.dtype(np.int8))
        if name in specs:
            return (tuple(specs[name]["shape"]), np.dtype(specs[name]["dtype"]))
        return None

    monkeypatch.setattr("pyrtc.streams._existing_shm_spec", _existing)
    monkeypatch.setattr("pyrtc.streams.clear_shms", lambda names: cleared.extend(names))

    rebuilt, reused = reconcile_expected_output_shms(config)

    assert rebuilt == ["wfc"]
    assert "signal" in reused
    assert cleared == ["wfc"]


def test_expected_output_shm_specs_include_pywfs_signal2d():
    config = read_system_config(
        REPO_ROOT / "examples" / "pywfs" / "pywfs_OOPAO_config.yaml", validate=False
    )

    specs = expected_output_shm_specs_for_config(config)

    assert specs["signal"]["dtype"] == np.float32
    assert specs["signal_2d"]["dtype"] == np.float32
    assert specs["signal_2d"]["shape"] == (24, 48)


def test_expected_output_shm_specs_include_oopao_shwfs_signal2d():
    config = read_system_config(
        REPO_ROOT / "examples" / "shwfs" / "shwfs_OOPAO_config.yaml", validate=False
    )

    specs = expected_output_shm_specs_for_config(config)

    assert specs["signal"]["dtype"] == np.float32
    assert specs["signal_2d"]["dtype"] == np.float32
    assert specs["signal_2d"]["shape"] == (40, 20)


def test_expected_output_shm_specs_include_specula_shwfs_signal2d():
    config = read_system_config(
        REPO_ROOT / "examples" / "shwfs" / "shwfs_SPECULA_config.yaml", validate=False
    )

    specs = expected_output_shm_specs_for_config(config)

    assert specs["wfs"]["shape"] == (160, 160)
    assert specs["signal"]["dtype"] == np.float32
    assert specs["signal_2d"]["shape"] == (40, 20)


def test_socket_json_helpers_handle_back_to_back_messages():
    left, right = socket.socketpair()
    try:
        _socket_send_json(left, {"type": "get", "property": "gain"})
        _socket_send_json(left, {"status": "OK", "property": 1.25})

        buffer = ""
        first, buffer = _socket_read_json(right, buffer)
        second, buffer = _socket_read_json(right, buffer)

        assert first == {"type": "get", "property": "gain"}
        assert second == {"status": "OK", "property": 1.25}
        assert buffer == ""
    finally:
        left.close()
        right.close()


def test_manager_latency_infers_loop_path(monkeypatch, tmp_path):
    from pyrtc import latency

    with publishing_chain(["wfs", "signal", "wfc"]) as opener:
        monkeypatch.setattr(latency, "open_stream", opener)

        manager = RTCManager.from_config_file(_write_runtime_synthetic_config(tmp_path))
        report = manager.latency(samples=8)

        assert report["stream_path"] == ["wfs", "signal", "wfc"]
        assert report["inferred_path"] is True
        assert report["total"]["source_shm"] == "wfs"
        assert report["total"]["target_shm"] == "wfc"
        assert report["total"]["alignment"] == "frame_id"
        assert len(report["segments"]) == 2


def test_manager_latency_uses_explicit_pair_when_requested(monkeypatch, tmp_path):
    from pyrtc import latency

    with publishing_chain(["signal", "wfc"]) as opener:
        monkeypatch.setattr(latency, "open_stream", opener)

        manager = RTCManager.from_config_file(_write_runtime_synthetic_config(tmp_path))
        report = manager.latency(source_shm="signal", target_shm="wfc", samples=8)

        assert report["stream_path"] == ["signal", "wfc"]
        assert report["total"]["source_shm"] == "signal"
        assert report["total"]["target_shm"] == "wfc"


def test_manager_stop_is_idempotent_for_soft_system(private_config_path):
    class FakeRuntime:
        def __init__(self):
            self.calls = 0

        def stop(self):
            self.calls += 1

        def status(self):
            return {"state": "stopped", "mode": "soft-rtc"}

    manager = RTCManager.from_config_file(private_config_path)
    runtime = FakeRuntime()
    manager.runtimes = {"wfs": runtime}
    manager.state = "running"

    manager.stop()
    manager.stop()

    assert manager.status()["state"] == "built"
    assert runtime.calls == 2


def test_manager_mode_override_uses_hard_runtime_with_short_alias(monkeypatch, private_config_path):
    calls = []

    class FakeLauncher:
        def __init__(self, hardware_file, config_file, port, timeout=None):
            self.hardware_file = hardware_file
            self.config_file = config_file
            self.port = port
            self.timeout = timeout

        def launch(self):
            calls.append(("launch", self.hardware_file, self.port))

        def run(self, function, *args, timeout=None):
            calls.append(("run", function, self.port))
            return 1

        def shutdown(self):
            calls.append(("shutdown", self.hardware_file, self.port))
            return 1

    manager = RTCManager.from_config_file(
        private_config_path, mode="hard", launcher_cls=FakeLauncher
    )

    manager.start()
    status = manager.status()
    manager.stop()

    assert status["mode"] == "hard-rtc"
    assert status["components"]["loop"]["mode"] == "hard-rtc"
    assert any(entry[:2] == ("run", "start") for entry in calls)
    assert any(entry[0] == "shutdown" for entry in calls)


def test_hard_runtime_stays_stopped_after_manual_stop():
    calls = []

    class FakeLauncher:
        def __init__(self, hardware_file, config_file, port, timeout=None):
            self.port = port

        def launch(self):
            calls.append(("launch", self.port))

        def run(self, function, *args, timeout=None):
            calls.append(("run", function, self.port))
            return 1

        def shutdown(self):
            calls.append(("shutdown", self.port))
            return 1

    runtime = HardComponentRuntime(
        "loop",
        component_class=type("LoopComponent", (), {}),
        script_path="loop.py",
        config_path="config.yaml",
        port=5603,
        launcher_cls=FakeLauncher,
    )

    runtime.start()
    runtime.stop()
    state = runtime.refresh_health()

    assert state == "stopped"
    assert runtime.state == "stopped"
    assert runtime.desired_running is False
    assert runtime.launcher is None


def test_manager_uses_hard_runtime_with_launcher_integration(monkeypatch, private_config_path):
    calls = []

    class FakeLauncher:
        def __init__(self, hardware_file, config_file, port, timeout=None):
            self.hardware_file = hardware_file
            self.config_file = config_file
            self.port = port
            self.timeout = timeout

        def launch(self):
            calls.append(("launch", self.hardware_file, self.port))

        def run(self, function, *args, timeout=None):
            calls.append(("run", function, self.port))
            return 1

        def shutdown(self):
            calls.append(("shutdown", self.hardware_file, self.port))
            return 1

    manager = RTCManager.from_config_file(private_config_path, launcher_cls=FakeLauncher)
    manager.config = copy.deepcopy(manager.config)
    manager.config["manager"] = {
        "mode": "hard-rtc",
        "component_classes": {
            "wfs": "pyrtc.wavefront_sensor.WavefrontSensor",
            "slopes": "pyrtc.slopes_process.SlopesProcess",
            "loop": "pyrtc.loop.Loop",
            "wfc": "pyrtc.wavefront_corrector.WavefrontCorrector",
            "psf": "pyrtc.science_camera.ScienceCamera",
        },
        "component_files": {
            "wfs": str(REPO_ROOT / "pyrtc" / "wavefront_sensor.py"),
            "slopes": str(REPO_ROOT / "pyrtc" / "slopes_process.py"),
            "loop": str(REPO_ROOT / "pyrtc" / "loop.py"),
            "wfc": str(REPO_ROOT / "pyrtc" / "wavefront_corrector.py"),
            "psf": str(REPO_ROOT / "pyrtc" / "science_camera.py"),
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
    assert any(entry[:2] == ("launch", str(REPO_ROOT / "pyrtc" / "loop.py")) for entry in calls)
    assert any(entry[:2] == ("run", "start") for entry in calls)
    assert any(entry[0] == "shutdown" for entry in calls)


def test_manager_requires_config_path_for_hard_mode_from_dict():
    manager = RTCManager.from_config(
        {
            "wfs": {
                "name": "wavefrontSensor",
                "width": 16,
                "height": 16,
                "dark_count": 1,
                "functions": ["expose"],
            },
            "slopes": {
                "type": "SHWFS",
                "signal_type": "slopes",
                "sub_ap_spacing": 8,
                "sub_ap_offset_x": 0,
                "sub_ap_offset_y": 0,
                "functions": ["compute_signal"],
            },
            "wfc": {
                "name": "dm",
                "num_actuators": 8,
                "num_modes": 8,
                "functions": ["send_to_hardware"],
            },
            "loop": {"gain": 0.1, "num_dropped_modes": 0, "functions": ["standard_integrator"]},
            "manager": {
                "mode": "hard-rtc",
                "component_classes": {
                    "wfs": "pyrtc.wavefront_sensor.WavefrontSensor",
                    "slopes": "pyrtc.slopes_process.SlopesProcess",
                    "loop": "pyrtc.loop.Loop",
                    "wfc": "pyrtc.wavefront_corrector.WavefrontCorrector",
                },
            },
        }
    )

    with pytest.raises(ValueError, match="config_path"):
        manager.start()


def test_manager_supports_explicit_manager_declared_sections(private_config_path):
    calls = []

    class FakeLauncher:
        def __init__(self, hardware_file, config_file, port, timeout=None):
            self.hardware_file = hardware_file
            self.config_file = config_file
            self.port = port
            self.timeout = timeout

        def launch(self):
            calls.append(("launch", self.hardware_file, self.port))

        def run(self, function, *args, timeout=None):
            calls.append(("run", function, self.port))
            return 1

        def shutdown(self):
            calls.append(("shutdown", self.hardware_file, self.port))
            return 1

    manager = RTCManager.from_config_file(private_config_path, launcher_cls=FakeLauncher)
    manager.config = copy.deepcopy(manager.config)
    manager.config["modulator"] = {"name": "tutorial-modulator", "frequency": 300, "amplitude": 600}
    manager.config["manager"] = {
        "mode": "hard-rtc",
        "component_classes": {
            "modulator": "pyrtc.component.Component",
            "wfs": "pyrtc.wavefront_sensor.WavefrontSensor",
            "slopes": "pyrtc.slopes_process.SlopesProcess",
            "loop": "pyrtc.loop.Loop",
            "wfc": "pyrtc.wavefront_corrector.WavefrontCorrector",
            "psf": "pyrtc.science_camera.ScienceCamera",
        },
        "component_files": {
            "modulator": str(REPO_ROOT / "pyrtc" / "Component.py"),
            "wfs": str(REPO_ROOT / "pyrtc" / "wavefront_sensor.py"),
            "slopes": str(REPO_ROOT / "pyrtc" / "slopes_process.py"),
            "loop": str(REPO_ROOT / "pyrtc" / "loop.py"),
            "wfc": str(REPO_ROOT / "pyrtc" / "wavefront_corrector.py"),
            "psf": str(REPO_ROOT / "pyrtc" / "science_camera.py"),
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

    assert any(
        entry[:2] == ("launch", str(REPO_ROOT / "pyrtc" / "Component.py")) for entry in calls
    )


def test_manager_injects_shared_resources_into_soft_runtimes(tmp_path):
    class FakeResource:
        def __init__(self, conf, system_conf):
            self.conf = conf
            self.system_conf = system_conf

    class FakeComponent:
        def __init__(self, conf, resource):
            self.conf = conf
            self.resource = resource
            self.alive = True
            self.running = False

        def start(self):
            self.running = True

        def stop(self):
            self.running = False

    manager = RTCManager.from_config(
        {
            "demo": {
                "class_name": "ignored",
                "resource": "shared",
                "input_streams": {},
                "output_streams": {},
            },
            "resources": {
                "shared": {
                    "class_name": "ignored",
                }
            },
            "manager": {
                "mode": "soft-rtc",
                "component_classes": {"demo": "ignored"},
            },
        },
        config_path=str(tmp_path / "resource_demo.yaml"),
    )
    manager.validated = True
    manager.state = "validated"
    manager._resolve_resource_class = lambda resource_name: FakeResource
    manager._resolve_component_class = lambda section_name: FakeComponent

    manager.start()
    try:
        runtime = manager.runtimes["demo"]
        assert isinstance(runtime.component, FakeComponent)
        assert isinstance(runtime.component.resource, FakeResource)
        assert runtime.state == "running"
    finally:
        manager.close()
    assert manager.resources == {}


def test_manager_injects_component_provider_resources_into_soft_runtimes(tmp_path):
    starts = []
    closes = []

    class FakeProvider:
        def __init__(self, conf):
            self.conf = conf
            self.alive = True
            self.running = False

        def start(self):
            self.running = True
            starts.append("provider")

        def stop(self):
            self.running = False

        def close(self):
            closes.append("provider")

    class FakeConsumer:
        def __init__(self, conf, resource):
            self.conf = conf
            self.resource = resource
            self.alive = True
            self.running = False

        def start(self):
            self.running = True
            starts.append("consumer")

        def stop(self):
            self.running = False

        def close(self):
            closes.append("consumer")

    manager = RTCManager.from_config(
        {
            "provider": {
                "class_name": "ignored",
                "input_streams": {},
                "output_streams": {},
            },
            "consumer": {
                "class_name": "ignored",
                "resource": "provider",
                "input_streams": {},
                "output_streams": {},
            },
            "manager": {
                "mode": "soft-rtc",
                "component_classes": {"provider": "ignored", "consumer": "ignored"},
            },
        },
        config_path=str(tmp_path / "component_resource_demo.yaml"),
    )
    manager.validated = True
    manager.state = "validated"
    manager._resolve_component_class = lambda section_name: (
        FakeProvider if section_name == "provider" else FakeConsumer
    )

    manager.start()
    try:
        runtime = manager.runtimes["consumer"]
        assert isinstance(runtime.component.resource, FakeProvider)
        assert starts == ["provider", "consumer"]
    finally:
        manager.close()
    # Consumers close before the provider they depend on.
    assert closes == ["consumer", "provider"]


def test_manager_status_includes_health_metadata_for_hard_runtime(tmp_path, private_config_path):
    class HealthLauncher:
        def __init__(self, hardware_file, config_file, port, timeout=None):
            self.hardware_file = hardware_file
            self.config_file = config_file
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
        RTCManager.from_config_file(private_config_path, launcher_cls=HealthLauncher),
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


def test_manager_marks_component_degraded_when_health_check_fails(private_config_path):
    class DegradedLauncher:
        def __init__(self, hardware_file, config_file, port, timeout=None):
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
        RTCManager.from_config_file(private_config_path, launcher_cls=DegradedLauncher),
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


def test_manager_restarts_failed_child_when_policy_is_on_failure(private_config_path):
    class RestartingLauncher:
        launches = 0
        loop_failed_once = False

        def __init__(self, hardware_file, config_file, port, timeout=None):
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
        RTCManager.from_config_file(private_config_path, launcher_cls=RestartingLauncher),
        RestartingLauncher,
        manager_overrides={"restart_policy": "on-failure"},
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


def test_manager_repeated_failures_increment_restart_count_and_preserve_last_error(
    private_config_path,
):
    class FlappingLauncher:
        launches = 0

        def __init__(self, hardware_file, config_file, port, timeout=None):
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
        RTCManager.from_config_file(private_config_path, launcher_cls=FlappingLauncher),
        FlappingLauncher,
        manager_overrides={"restart_policy": "on-failure"},
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


def _installed_pyrtc_loop_path() -> Path:
    """Return the path to the installed pyrtc.loop module on disk.

    Tests must target the copy of pyrtc that's actually importable in
    the current environment, which is the installed wheel on CI
    (location reported by importlib.util.find_spec) — not the
    local source checkout. The two diverge as soon as pip install .
    runs in CI, so referencing REPO_ROOT directly causes duplicate
    class objects to be exec'd.
    """
    import importlib.util
    import os

    spec = importlib.util.find_spec("pyrtc.loop")
    if spec is None or spec.origin is None:
        raise RuntimeError("pyrtc.loop is not importable")
    return Path(os.path.normpath(spec.origin))


def test_import_symbol_from_file_reuses_canonical_pyrtc_module():
    from pyrtc.loop import Loop
    from pyrtc.component_loading import import_symbol_from_file as _import_symbol_from_file

    resolved = _import_symbol_from_file(str(_installed_pyrtc_loop_path()), "Loop")

    assert resolved is Loop


def test_manager_latency_infers_path_for_classfile_components(monkeypatch):
    from pyrtc import latency

    with publishing_chain(["wfs", "signal", "wfc"]) as opener:
        monkeypatch.setattr(latency, "open_stream", opener)

        # Use the real example config: its components are loaded via class_file,
        # which used to produce duplicate class objects with empty descriptors
        # and break stream-path inference.
        manager = RTCManager.from_config_file(SYNTHETIC_CONFIG_PATH)
        report = manager.latency(samples=8)

        assert report["stream_path"] == ["wfs", "signal", "wfc"]
        assert report["inferred_path"] is True
