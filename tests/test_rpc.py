"""Tests for the hard-RTC RPC message protocol (pyrtc.rpc)."""

import os

import numpy as np
import pytest

import pyrtc.rpc as rpc
from pyrtc.logging_utils import configure_logging
from pyrtc.rpc import Listener, PROTOCOL_VERSION, _coerce_property_value, HardwareLauncher


class _Hardware:
    def __init__(self):
        self.gain = 0.1
        self.enabled = False
        self.label = "dm"
        self.num_modes = 10
        self.modes = [1, 2]
        self.calls = []
        self.unset = None
        self._secret = "hidden"

    def start(self):
        self.calls.append("start")

    def echo(self, value):
        self.calls.append(("echo", value))
        return value

    def measure(self):
        return np.float64(2.5)

    def matrix(self):
        return np.eye(2, dtype=np.float32)

    def opaque(self):
        return object()

    def _private_helper(self):
        self.calls.append("private")

    def boom(self):
        raise RuntimeError("hardware fault")

    def __del__(self):
        self.calls.append("shutdown")


def _listener(hardware=None) -> Listener:
    listener = Listener.__new__(Listener)
    listener.hardware = hardware if hardware is not None else _Hardware()
    listener.running = True
    listener.OKMessage = {"status": "OK", "protocol": PROTOCOL_VERSION}
    listener.BadMessage = {"status": "BAD", "protocol": PROTOCOL_VERSION}
    return listener


# request dispatch


def test_get_returns_property():
    listener = _listener()
    reply = listener.handle_request(
        {"type": "get", "property": "gain", "protocol": PROTOCOL_VERSION}
    )
    assert reply["status"] == "OK"
    assert reply["property"] == 0.1


def test_set_coerces_onto_current_type():
    listener = _listener()
    reply = listener.handle_request(
        {"type": "set", "property": "num_modes", "value": 12.0, "protocol": PROTOCOL_VERSION}
    )
    assert reply["status"] == "OK"
    assert listener.hardware.num_modes == 12
    assert isinstance(listener.hardware.num_modes, int)


def test_set_bool_string_false_is_false():
    listener = _listener()
    reply = listener.handle_request(
        {"type": "set", "property": "enabled", "value": "False", "protocol": PROTOCOL_VERSION}
    )
    assert reply["status"] == "OK"
    assert listener.hardware.enabled is False

    reply = listener.handle_request(
        {"type": "set", "property": "enabled", "value": True, "protocol": PROTOCOL_VERSION}
    )
    assert reply["status"] == "OK"
    assert listener.hardware.enabled is True


def test_set_unknown_property_reports_error():
    listener = _listener()
    reply = listener.handle_request(
        {"type": "set", "property": "missing", "value": 1, "protocol": PROTOCOL_VERSION}
    )
    assert reply["status"] == "BAD"
    assert "missing" in reply["error"]


def test_run_invokes_function_with_args_and_returns_value():
    listener = _listener()
    reply = listener.handle_request(
        {"type": "run", "function": "echo", "args": [42], "protocol": PROTOCOL_VERSION}
    )
    assert reply["status"] == "OK"
    assert reply["property"] == 42
    assert listener.hardware.calls == [("echo", 42)]


def test_run_converts_numpy_results():
    listener = _listener()
    scalar_reply = listener.handle_request(
        {"type": "run", "function": "measure", "protocol": PROTOCOL_VERSION}
    )
    assert scalar_reply["property"] == 2.5

    matrix_reply = listener.handle_request(
        {"type": "run", "function": "matrix", "protocol": PROTOCOL_VERSION}
    )
    assert matrix_reply["property"] == [[1.0, 0.0], [0.0, 1.0]]


def test_run_drops_unserializable_results_but_succeeds():
    listener = _listener()
    reply = listener.handle_request(
        {"type": "run", "function": "opaque", "protocol": PROTOCOL_VERSION}
    )
    assert reply["status"] == "OK"
    assert "property" not in reply


def test_run_failure_carries_error_message():
    listener = _listener()
    reply = listener.handle_request(
        {"type": "run", "function": "boom", "protocol": PROTOCOL_VERSION}
    )
    assert reply["status"] == "BAD"
    assert "hardware fault" in reply["error"]


def test_protocol_mismatch_is_rejected():
    listener = _listener()
    reply = listener.handle_request(
        {"type": "get", "property": "gain", "protocol": PROTOCOL_VERSION + 1}
    )
    assert reply["status"] == "BAD"
    assert "protocol" in reply["error"]


def test_unknown_type_and_missing_type_are_rejected():
    listener = _listener()
    assert (
        listener.handle_request({"type": "noop", "protocol": PROTOCOL_VERSION})["status"] == "BAD"
    )
    assert listener.handle_request({"protocol": PROTOCOL_VERSION})["status"] == "BAD"
    assert listener.handle_request("not a dict")["status"] == "BAD"


@pytest.mark.parametrize("name", ["_secret", "__class__", "__dict__", "__init__", "_"])
def test_get_rejects_private_and_dunder_names(name):
    listener = _listener()
    reply = listener.handle_request({"type": "get", "property": name, "protocol": PROTOCOL_VERSION})
    assert reply["status"] == "BAD"
    assert "private" in reply["error"]
    assert "property" not in reply


@pytest.mark.parametrize("name", ["_secret", "__class__", "__dict__"])
def test_set_rejects_private_and_dunder_names(name):
    hardware = _Hardware()
    original_class = hardware.__class__
    listener = _listener(hardware)
    reply = listener.handle_request(
        {"type": "set", "property": name, "value": "x", "protocol": PROTOCOL_VERSION}
    )
    assert reply["status"] == "BAD"
    assert "private" in reply["error"]
    assert hardware._secret == "hidden"
    assert hardware.__class__ is original_class


@pytest.mark.parametrize("name", ["_private_helper", "__del__", "__init__", "__class__"])
def test_run_rejects_private_and_dunder_names(name):
    hardware = _Hardware()
    listener = _listener(hardware)
    reply = listener.handle_request({"type": "run", "function": name, "protocol": PROTOCOL_VERSION})
    assert reply["status"] == "BAD"
    assert "private" in reply["error"]
    assert hardware.calls == []


@pytest.mark.parametrize(
    "request_type,key", [("get", "property"), ("set", "property"), ("run", "function")]
)
@pytest.mark.parametrize("name", [None, "", 5, ["gain"]])
def test_non_string_names_are_rejected(request_type, key, name):
    listener = _listener()
    reply = listener.handle_request(
        {"type": request_type, key: name, "value": 1, "protocol": PROTOCOL_VERSION}
    )
    assert reply["status"] == "BAD"
    assert "non-empty string" in reply["error"]


def test_run_rejects_non_callable_attribute():
    listener = _listener()
    reply = listener.handle_request(
        {"type": "run", "function": "gain", "protocol": PROTOCOL_VERSION}
    )
    assert reply["status"] == "BAD"
    assert "not callable" in reply["error"]


def test_set_refuses_to_overwrite_methods():
    hardware = _Hardware()
    listener = _listener(hardware)
    reply = listener.handle_request(
        {"type": "set", "property": "start", "value": "x", "protocol": PROTOCOL_VERSION}
    )
    assert reply["status"] == "BAD"
    assert "method" in reply["error"]
    assert callable(hardware.start)


def test_launcher_surfaces_private_name_rejection():
    hardware = _Hardware()
    listener = _listener(hardware)
    launcher = HardwareLauncher("dummy.py", "c.yaml", 9999)
    launcher.running = True
    launcher.write = lambda message: setattr(launcher, "_pending", listener.handle_request(message))
    launcher.read = lambda: launcher._pending

    assert launcher.get_property("__dict__") == -1
    assert "private" in launcher.last_error
    assert launcher.set_property("_secret", "leak") == -1
    assert "private" in launcher.last_error
    assert launcher.run("__del__") == -1
    assert "private" in launcher.last_error
    assert hardware.calls == []

    # Public access still works and clears the error.
    assert launcher.get_property("gain") == 0.1
    assert launcher.last_error is None
    assert launcher.run("start") == 1
    assert hardware.calls == ["start"]


def test_shutdown_stops_listener():
    listener = _listener()
    reply = listener.handle_request({"type": "shutdown", "protocol": PROTOCOL_VERSION})
    assert reply["status"] == "OK"
    assert listener.running is False
    assert "shutdown" in listener.hardware.calls


# value coercion unit tests


@pytest.mark.parametrize(
    "current, incoming, expected",
    [
        (False, "true", True),
        (True, 0, False),
        (3, 4.0, 4),
        (1.5, "2.5", 2.5),
        ("a", 5, "5"),
        ([1, 2], [3, 4], [3, 4]),
        ((1, 2), [3, 4], (3, 4)),
        (None, {"x": 1}, {"x": 1}),
    ],
)
def test_coerce_property_value(current, incoming, expected):
    assert _coerce_property_value(current, incoming) == expected


def test_coerce_property_value_rejects_ambiguous_bool():
    with pytest.raises(ValueError):
        _coerce_property_value(True, "maybe")


# launcher side


class _FakeSocket:
    def __init__(self):
        self.timeouts = []
        self._timeout = 0.25

    def gettimeout(self):
        return self._timeout

    def settimeout(self, value):
        self._timeout = value
        self.timeouts.append(value)


def test_launcher_run_honors_per_call_timeout(monkeypatch):
    launcher = HardwareLauncher("dummy.py", "c.yaml", 9999)
    launcher.running = True
    launcher.process_socket = _FakeSocket()
    sent = []
    launcher.write = lambda message: sent.append(message)
    launcher.read = lambda: {"status": "OK", "protocol": PROTOCOL_VERSION}

    assert launcher.run("compute_im", timeout=60.0) == 1

    assert sent[0]["type"] == "run"
    assert sent[0]["args"] == []
    assert sent[0]["protocol"] == PROTOCOL_VERSION
    # timeout applied for the call, then restored
    assert launcher.process_socket.timeouts == [60.0, 0.25]


def test_launcher_surfaces_child_error_message():
    launcher = HardwareLauncher("dummy.py", "c.yaml", 9999)
    launcher.running = True
    launcher.write = lambda message: None
    launcher.read = lambda: {
        "status": "BAD",
        "error": "run 'boom' failed: hardware fault",
        "protocol": PROTOCOL_VERSION,
    }

    assert launcher.run("boom") == -1
    assert "hardware fault" in launcher.last_error


def test_launcher_round_trip_through_listener_dispatch():
    hardware = _Hardware()
    listener = _listener(hardware)
    launcher = HardwareLauncher("dummy.py", "c.yaml", 9999)
    launcher.running = True
    launcher.write = lambda message: setattr(launcher, "_pending", listener.handle_request(message))
    launcher.read = lambda: launcher._pending

    assert launcher.set_property("gain", 0.3) == 1
    assert hardware.gain == 0.3
    assert launcher.get_property("gain") == 0.3
    assert launcher.run("echo", 7) == 7


# launcher lifecycle / health


class _FakeProcess:
    def __init__(self, exit_code=None):
        self._exit_code = exit_code
        self.terminated = False
        self.killed = False
        self.pid = 4242

    def poll(self):
        return self._exit_code

    def terminate(self):
        self.terminated = True
        self._exit_code = -15

    def wait(self, timeout=None):
        return self._exit_code

    def kill(self):
        self.killed = True


def test_launcher_close_terminates_live_process_when_forced():
    launcher = HardwareLauncher("dummy.py", "c.yaml", 9999)
    launcher.running = True
    launcher.process_socket = _FakeSocket()
    process = _FakeProcess(exit_code=None)
    launcher.process = process

    launcher.close(force=True)

    assert launcher.running is False
    assert launcher.process_socket is None
    assert launcher.process is None
    assert process.terminated is True


def test_launcher_close_without_force_leaves_process_untouched():
    launcher = HardwareLauncher("dummy.py", "c.yaml", 9999)
    launcher.running = True
    process = _FakeProcess(exit_code=None)
    launcher.process = process

    launcher.close(force=False)

    assert process.terminated is False
    assert launcher.process is None


def test_health_check_reports_dead_child():
    launcher = HardwareLauncher("dummy.py", "c.yaml", 9999)
    launcher.process = _FakeProcess(exit_code=3)

    health = launcher.health_check()

    assert health["state"] == "failed"
    assert "exited with code 3" in health["error"]


def test_health_check_degraded_when_rpc_fails():
    launcher = HardwareLauncher("dummy.py", "c.yaml", 9999)
    launcher.process = _FakeProcess(exit_code=None)
    launcher.running = True
    launcher.process_socket = _FakeSocket()
    launcher.write = lambda message: (_ for _ in ()).throw(ConnectionError("gone"))

    health = launcher.health_check(timeout=0.5)

    assert health["state"] == "degraded"
    # the per-call timeout was applied and restored
    assert launcher.process_socket.timeouts == [0.5, 0.25]


def test_health_check_degraded_when_component_not_running():
    launcher = HardwareLauncher("dummy.py", "c.yaml", 9999)
    launcher.process = _FakeProcess(exit_code=None)
    launcher.running = True
    launcher.write = lambda message: None
    launcher.read = lambda: {"status": "OK", "property": False, "protocol": PROTOCOL_VERSION}

    health = launcher.health_check()

    assert health["state"] == "degraded"
    assert "running=False" in health["error"]


def test_health_check_running():
    launcher = HardwareLauncher("dummy.py", "c.yaml", 9999)
    launcher.process = _FakeProcess(exit_code=None)
    launcher.running = True
    launcher.write = lambda message: None
    launcher.read = lambda: {"status": "OK", "property": True, "protocol": PROTOCOL_VERSION}

    health = launcher.health_check()

    assert health["state"] == "running"
    assert health["error"] is None


def test_launcher_shutdown_sends_message_and_closes():
    launcher = HardwareLauncher("dummy.py", "c.yaml", 9999)
    launcher.running = True
    sent = []
    launcher.write = lambda message: sent.append(message)
    launcher.read = lambda: {"status": "OK", "protocol": PROTOCOL_VERSION}

    assert launcher.shutdown() == 1
    assert sent[0]["type"] == "shutdown"
    assert launcher.running is False


# listener over a real socket pair


def _connected_listener(hardware):
    import socket as socket_module

    listener = _listener(hardware)
    rtc_side, child_side = socket_module.socketpair()
    listener.RTCsocket = child_side
    listener._read_buffer = ""
    return listener, rtc_side


def test_listener_listen_round_trip_over_socket():
    hardware = _Hardware()
    listener, rtc_side = _connected_listener(hardware)
    try:
        rpc._socket_send_json(
            rtc_side, {"type": "get", "property": "gain", "protocol": PROTOCOL_VERSION}
        )
        listener.listen()
        reply, _ = rpc._socket_read_json(rtc_side, "")
        assert reply["status"] == "OK"
        assert reply["property"] == 0.1
    finally:
        rtc_side.close()
        listener.RTCsocket.close()


def test_listener_stops_when_rtc_disconnects():
    hardware = _Hardware()
    listener, rtc_side = _connected_listener(hardware)
    try:
        rtc_side.close()
        listener.listen()
        assert listener.running is False
    finally:
        listener.RTCsocket.close()


def test_hardware_launcher_write_and_read():
    hl = HardwareLauncher("dummy.py", "c.yaml", 9999)
    calls = []

    def _write(msg):
        calls.append(msg)

    def _read():
        return {"status": "OK", "property": 42}

    hl.running = True
    hl.write = _write
    hl.read = _read
    assert hl.get_property("x") == 42
    assert calls[0]["type"] == "get"


def test_hardware_launcher_inherits_logging_env(monkeypatch, tmp_path):
    configure_logging(
        app_name="pyrtc-test-launcher",
        component_name="parent",
        level="DEBUG",
        log_dir=tmp_path,
        color=False,
        console=False,
    )

    captured = {}

    class _DummyProcess:
        pass

    class _DummySocket:
        def settimeout(self, timeout):
            captured["timeout"] = timeout

        def connect(self, address):
            captured["address"] = address

    def _fake_popen(command, stdin=None, stdout=None, text=None, bufsize=None, env=None):
        captured["command"] = command
        captured["env"] = env
        return _DummyProcess()

    monkeypatch.setattr(rpc, "Popen", _fake_popen)
    monkeypatch.setattr(rpc.socket, "socket", lambda *args, **kwargs: _DummySocket())
    monkeypatch.setattr(rpc.time, "sleep", lambda _seconds: None)

    launcher = HardwareLauncher("child.py", "config.yaml", 4567, timeout=1.5)
    launcher.launch()

    assert captured["command"][0] == os.sys.executable
    assert captured["address"] == ("127.0.0.1", 4567)
    assert captured["env"]["PYRTC_LOG_LEVEL"] == "DEBUG"
    assert os.path.basename(captured["env"]["PYRTC_LOG_DIR"]) == os.path.basename(str(tmp_path))
