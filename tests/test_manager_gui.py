from pathlib import Path

from pyRTC import RTCManager
from pyRTC.config_schema import read_system_config
from pyRTC.gui.manager_adapter import ManagerAdapter, _coerce_runtime_value, _is_live_runtime_field, _ordered_sections
from pyRTC.hardware.SyntheticSystems import _default_wfc_layout, build_synthetic_shwfs_response_matrix
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


def test_manager_adapter_uses_persisted_graph_positions_from_config():
    adapter = ManagerAdapter()
    adapter.load_config(str(SYNTHETIC_CONFIG_PATH))
    adapter.set_component_position("loop", 321.5, 654.25)

    snapshot = adapter.build_graph_snapshot()
    loop_node = next(node for node in snapshot.nodes if node.section_name == "loop")

    assert loop_node.x == 321.5
    assert loop_node.y == 654.25
    assert snapshot.metadata["positions"]["loop"] == {"x": 321.5, "y": 654.25}


def test_manager_adapter_can_initialize_empty_config_for_build_mode():
    adapter = ManagerAdapter()

    config = adapter.initialize_empty_config(mode="hard-rtc")

    assert config["manager"]["mode"] == "hard-rtc"
    assert config["streams"] == {}
    assert config["metadata"] == {}
    assert adapter.is_loaded() is True


def test_manager_adapter_set_parameter_updates_loaded_config():
    adapter = ManagerAdapter()
    adapter.load_config(str(SYNTHETIC_CONFIG_PATH))

    updated = adapter.set_parameter("loop", "gain", "0.55")

    assert updated == 0.55
    assert adapter.config["loop"]["gain"] == 0.55
    assert adapter.get_parameter("loop", "gain") == 0.55


def test_manager_adapter_build_exposes_built_state(tmp_path):
    adapter = ManagerAdapter()
    config_path = tmp_path / "synthetic_runtime_config.yaml"
    config = read_system_config(SYNTHETIC_CONFIG_PATH, validate=False)
    np_path = tmp_path / "synthetic_identity_im.npy"
    import numpy as np

    layout = _default_wfc_layout(int(config["wfc"]["numActuators"]))
    response = build_synthetic_shwfs_response_matrix(7, int(config["wfc"]["numModes"]), layout)
    np.save(np_path, response.astype(np.float32))
    config["loop"]["IMFile"] = str(np_path)
    config_path.write_text(__import__("yaml").safe_dump(config, sort_keys=False), encoding="utf-8")

    adapter.load_config(str(config_path))
    status = adapter.build()

    assert status["state"] == "built"
    assert status["components"]["wfs"]["state"] == "built"


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


def test_manager_adapter_lists_zero_arg_component_functions():
    adapter = ManagerAdapter()
    adapter.load_config(str(SYNTHETIC_CONFIG_PATH))

    functions = adapter.get_component_functions("wfc")
    names = {row["name"] for row in functions}

    assert "flatten" in names
    assert "sendToHardware" not in names
    assert "start" not in names


def test_manager_adapter_runs_zero_arg_component_function_on_soft_runtime():
    class _FakeComponent:
        def __init__(self):
            self.flatten_calls = 0

        def flatten(self):
            self.flatten_calls += 1
            return 123

    class _FakeRuntime:
        mode = "soft-rtc"
        component_class = _FakeComponent

    class _FakeManager:
        def __init__(self, component):
            self.runtimes = {"wfc": _FakeRuntime()}
            self._component = component

        def get_component(self, section_name):
            return self._component

        def status(self):
            return {"components": {"wfc": {"state": "running"}}, "state": "running"}

    component = _FakeComponent()
    adapter = ManagerAdapter()
    adapter.config = read_system_config(SYNTHETIC_CONFIG_PATH, validate=False)
    adapter.manager = _FakeManager(component)

    result = adapter.run_component_function("wfc", "flatten")

    assert result == 123
    assert component.flatten_calls == 1


def test_manager_adapter_runs_zero_arg_component_function_on_hard_runtime():
    class _FakeLauncher:
        def __init__(self):
            self.calls = []
            self.lastError = None

        def run(self, function_name):
            self.calls.append(function_name)
            return 1

    class _FakeComponentClass:
        def flatten(self):
            return None

    class _FakeRuntime:
        mode = "hard-rtc"
        component_class = _FakeComponentClass

    class _FakeManager:
        def __init__(self, launcher):
            self.runtimes = {"wfc": _FakeRuntime()}
            self._launcher = launcher

        def get_component(self, section_name):
            return self._launcher

        def status(self):
            return {"components": {"wfc": {"state": "running"}}, "state": "running"}

    launcher = _FakeLauncher()
    adapter = ManagerAdapter()
    adapter.config = read_system_config(SYNTHETIC_CONFIG_PATH, validate=False)
    adapter.manager = _FakeManager(launcher)

    result = adapter.run_component_function("wfc", "flatten")

    assert result == 1
    assert launcher.calls == ["flatten"]


def test_manager_adapter_adds_custom_component_and_connection():
    adapter = ManagerAdapter()
    adapter.load_config(str(SYNTHETIC_CONFIG_PATH))

    adapter.add_component("science_copy", template_section="psf")
    adapter.add_connection(
        "science_copy_psf",
        output_component="science_copy",
        input_components=["loop"],
        component_stream="psfShort",
    )

    assert "science_copy" in adapter.config
    assert adapter.config["manager"]["componentClasses"]["science_copy"] == "pyRTC.ScienceCamera.ScienceCamera"
    assert adapter.config["streams"]["science_copy_psf"]["outputComponent"] == "science_copy"
    assert adapter.config["streams"]["science_copy_psf"]["componentStream"] == "psfShort"


def test_manager_adapter_snapshot_includes_custom_stream_connections():
    adapter = ManagerAdapter()
    adapter.load_config(str(SYNTHETIC_CONFIG_PATH))
    adapter.add_component("science_copy", template_section="psf")
    adapter.add_connection(
        "science_copy_psf",
        output_component="science_copy",
        input_components=["loop"],
        component_stream="psfShort",
    )

    snapshot = adapter.build_graph_snapshot(runtime_controls_enabled=False)

    sections = {node.section_name for node in snapshot.nodes}
    edges = {(edge.source_section, edge.target_section, edge.source_stream) for edge in snapshot.edges}

    assert "science_copy" in sections
    assert ("science_copy", "loop", "science_copy_psf") in edges


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