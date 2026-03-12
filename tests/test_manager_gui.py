from pathlib import Path

from pyRTC import RTCManager
from pyRTC.config_schema import read_system_config
from pyRTC.gui.manager_adapter import ManagerAdapter, _coerce_runtime_value, _is_live_runtime_field, _ordered_sections
from pyRTC.scripts import manager_gui


REPO_ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_CONFIG_PATH = REPO_ROOT / "examples" / "synthetic_shwfs" / "config.yaml"


def test_manager_gui_parser_supports_mode_theme_and_refresh():
    parser = manager_gui._build_arg_parser()
    args = parser.parse_args([str(SYNTHETIC_CONFIG_PATH), "--mode", "hard", "--theme", "light", "--refresh-ms", "1500"])

    assert args.config == str(SYNTHETIC_CONFIG_PATH)
    assert args.mode == "hard"
    assert args.theme == "light"
    assert args.refresh_ms == 1500


def test_manager_adapter_builds_graph_snapshot_from_synthetic_config():
    adapter = ManagerAdapter()
    adapter.load_config(str(SYNTHETIC_CONFIG_PATH))

    snapshot = adapter.build_graph_snapshot()

    section_names = [node.section_name for node in snapshot.nodes]
    edges = {(edge.source_section, edge.target_section, edge.source_stream) for edge in snapshot.edges}

    assert snapshot.config_path == str(SYNTHETIC_CONFIG_PATH.resolve())
    assert {"wfs", "slopes", "loop", "wfc", "psf"}.issubset(section_names)
    assert ("wfs", "slopes", "wfs") in edges
    assert ("slopes", "loop", "signal") in edges
    assert ("loop", "wfc", "wfc") in edges


def test_manager_adapter_set_parameter_updates_loaded_config():
    adapter = ManagerAdapter()
    adapter.load_config(str(SYNTHETIC_CONFIG_PATH))

    updated = adapter.set_parameter("loop", "gain", "0.55")

    assert updated == 0.55
    assert adapter.config["loop"]["gain"] == 0.55
    assert adapter.get_parameter("loop", "gain") == 0.55


def test_manager_adapter_prefers_common_viewer_streams():
    adapter = ManagerAdapter()
    adapter.config = read_system_config(SYNTHETIC_CONFIG_PATH, validate=False)
    adapter.config_path = str(SYNTHETIC_CONFIG_PATH.resolve())
    adapter.manager = RTCManager.from_config(adapter.config, config_path=adapter.config_path)

    streams = adapter.suggested_viewer_streams()

    assert streams == ["wfsRaw", "wfs", "signal2D", "wfc2D", "psfShort", "psfLong"]


def test_runtime_field_policy_keeps_functions_config_only():
    assert _is_live_runtime_field("gain") is True
    assert _is_live_runtime_field("functions") is False
    assert _is_live_runtime_field("IMFile") is False


def test_manager_adapter_skips_hard_runtime_rpc_for_config_only_fields():
    class _FakeLauncher:
        def __init__(self):
            self.calls = []
            self.lastError = None

        def getProperty(self, name):
            self.calls.append(name)
            return -1

    class _FakeRuntime:
        mode = "hard-rtc"

    adapter = ManagerAdapter()
    adapter.config = read_system_config(SYNTHETIC_CONFIG_PATH, validate=False)
    adapter.config_path = str(SYNTHETIC_CONFIG_PATH.resolve())
    adapter.manager = RTCManager.from_config(adapter.config, config_path=adapter.config_path)
    adapter.manager.state = "running"
    launcher = _FakeLauncher()
    adapter.manager.runtimes = {"loop": _FakeRuntime()}
    adapter.manager.get_component = lambda section_name: launcher

    value = adapter.get_parameter("loop", "functions", fallback=["standardIntegrator"])

    assert value == ["standardIntegrator"]
    assert launcher.calls == []


def test_ordered_sections_excludes_manager_metadata_sections():
    config = read_system_config(SYNTHETIC_CONFIG_PATH, validate=False)

    sections = _ordered_sections(config)

    assert "wfs" in sections
    assert "loop" in sections
    assert "manager" not in sections


def test_manager_adapter_mode_switch_rebuilds_manager_when_stopped():
    adapter = ManagerAdapter()
    adapter.load_config(str(SYNTHETIC_CONFIG_PATH), mode="soft-rtc")

    status = adapter.set_mode("hard-rtc")

    assert status["mode"] == "hard-rtc"
    assert adapter.selected_mode() == "hard-rtc"


def test_manager_adapter_rejects_restart_for_non_runtime_section():
    adapter = ManagerAdapter()
    adapter.load_config(str(SYNTHETIC_CONFIG_PATH))

    try:
        adapter.restart_component("manager")
    except KeyError as exc:
        assert exc.args == ("manager",)
    else:
        raise AssertionError("Expected KeyError for non-runtime section")


def test_manager_adapter_component_start_stop_round_trip():
    class _FakeManager:
        def __init__(self):
            self._status = {"components": {"loop": {"state": "stopped"}}, "state": "stopped"}

        def start_component(self, section_name):
            self._status = {"components": {section_name: {"state": "running"}}, "state": "running"}

        def stop_component(self, section_name):
            self._status = {"components": {section_name: {"state": "stopped"}}, "state": "stopped"}

        def status(self):
            return self._status

    adapter = ManagerAdapter()
    adapter.config = read_system_config(SYNTHETIC_CONFIG_PATH, validate=False)
    adapter.manager = _FakeManager()

    start_status = adapter.start_component("loop")
    stop_status = adapter.stop_component("loop")

    assert start_status["components"]["loop"]["state"] == "running"
    assert stop_status["components"]["loop"]["state"] == "stopped"


def test_coerce_runtime_value_parses_infinite_float_lists():
    value = _coerce_runtime_value("[-inf, inf]", "list[float]")

    assert value[0] == float("-inf")
    assert value[1] == float("inf")