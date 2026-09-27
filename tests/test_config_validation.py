import copy
import json
from pathlib import Path

import pytest
import yaml

import logging

from pyrtc.config_schema import (
    collect_config_warnings,
    read_system_config,
    validate_system_config,
)
from pyrtc.scripts import validate_config as validate_config_cli
from pyrtc.utils import (
    ConfigValidationError,
    validate_loop_config,
    validate_wfc_config,
    validate_wfs_config,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_CONFIG_PATH = REPO_ROOT / "examples" / "synthetic_shwfs" / "config.yaml"


def test_validate_wfs_config_accepts_valid_defaults():
    validate_wfs_config({"width": 16, "height": 16, "dark_count": 10})


def test_validate_wfs_config_rejects_invalid_width():
    with pytest.raises(ConfigValidationError):
        validate_wfs_config({"width": 0, "height": 16})


def test_validate_wfc_config_requires_keys():
    with pytest.raises(ConfigValidationError):
        validate_wfc_config({"num_actuators": 97, "num_modes": 80})


def test_validate_wfc_config_rejects_bad_modes():
    with pytest.raises(ConfigValidationError):
        validate_wfc_config({"name": "dm", "num_actuators": 97, "num_modes": 0})


def test_validate_wfc_config_accepts_valid_minimal():
    validate_wfc_config({"name": "dm", "num_actuators": 97, "num_modes": 80})


def test_validate_loop_config_allows_negative_gain():
    validate_loop_config({"gain": -0.1, "p_gain": -0.2, "i_gain": -0.3, "d_gain": -0.4})


def test_validate_loop_config_rejects_bad_limits_shape():
    with pytest.raises(ConfigValidationError):
        validate_loop_config({"control_limits": [0.0, 1.0, 2.0]})


def test_validate_loop_config_accepts_valid_config():
    validate_loop_config(
        {
            "cm_method": "tikhonov",
            "conditioning": 1000,
            "gain": 0.1,
            "leaky_gain": 0.01,
            "num_dropped_modes": 0,
            "tikhonov_reg": 0.001,
            "control_limits": [-1.0, 1.0],
            "integral_limits": [-0.5, 0.5],
            "absolute_limits": [-2.0, 2.0],
        }
    )


def test_validate_system_config_accepts_float_shwfs_subap_spacing():
    conf = read_system_config(SYNTHETIC_CONFIG_PATH, validate=False)
    conf["slopes"]["sub_ap_spacing"] = 8.5

    normalized = validate_system_config(conf, config_path=SYNTHETIC_CONFIG_PATH)

    assert normalized["slopes"]["sub_ap_spacing"] == 8.5


def test_validate_loop_config_rejects_bad_cm_method():
    with pytest.raises(ConfigValidationError, match="cm_method"):
        validate_loop_config({"cm_method": "ridge-ish"})


def test_validate_loop_config_rejects_bad_tikhonov_regularization():
    with pytest.raises(ConfigValidationError, match="tikhonov_reg"):
        validate_loop_config({"tikhonov_reg": -0.1})


def test_validate_system_config_accepts_synthetic_example():
    normalized = read_system_config(SYNTHETIC_CONFIG_PATH)

    assert normalized["manager"]["mode"] == "soft-rtc"
    assert normalized["wfs"]["class_name"] == "SyntheticSHWFS"
    assert normalized["wfc"]["num_modes"] == 97
    assert Path(normalized["metadata"]["config_path"]).resolve() == SYNTHETIC_CONFIG_PATH.resolve()


def test_validate_system_config_rejects_invalid_component_class_name():
    conf = read_system_config(SYNTHETIC_CONFIG_PATH, validate=False)
    conf["loop"]["class_name"] = "DefinitelyNotARealComponent"

    with pytest.raises(ConfigValidationError, match="class_name"):
        validate_system_config(conf)


def test_validate_system_config_resolves_relative_file_paths_against_config_file():
    conf = read_system_config(SYNTHETIC_CONFIG_PATH, validate=False)
    conf["loop"]["im_file"] = "synthetic_identity_im.npy"
    conf["manager"] = {"mode": "hard-rtc", "component_files": {"loop": "../../pyrtc/loop.py"}}

    normalized = validate_system_config(conf, config_path=SYNTHETIC_CONFIG_PATH)

    assert normalized["loop"]["im_file"] == str(
        (SYNTHETIC_CONFIG_PATH.parent / "synthetic_identity_im.npy").resolve()
    )
    assert normalized["manager"]["component_files"]["loop"] == str(
        (SYNTHETIC_CONFIG_PATH.parent / "../../pyrtc/loop.py").resolve()
    )


def test_validate_system_config_rejects_missing_required_section():
    conf = read_system_config(SYNTHETIC_CONFIG_PATH, validate=False)
    conf.pop("loop")

    with pytest.raises(ConfigValidationError, match="missing required top-level section"):
        validate_system_config(conf)


def test_validate_system_config_rejects_invalid_function_name():
    conf = read_system_config(SYNTHETIC_CONFIG_PATH, validate=False)
    conf["loop"]["functions"] = ["notARealLoopMethod"]

    with pytest.raises(ConfigValidationError, match="notARealLoopMethod"):
        validate_system_config(conf)


def test_validate_system_config_rejects_descriptor_type_mismatch():
    conf = read_system_config(SYNTHETIC_CONFIG_PATH, validate=False)
    conf["psf"]["dark_count"] = "16"

    with pytest.raises(ConfigValidationError, match="dark_count"):
        validate_system_config(conf)


def test_validate_system_config_rejects_too_many_dropped_modes():
    conf = read_system_config(SYNTHETIC_CONFIG_PATH, validate=False)
    conf["loop"]["num_dropped_modes"] = conf["wfc"]["num_modes"]

    with pytest.raises(ConfigValidationError, match="num_dropped_modes"):
        validate_system_config(conf)


def test_validate_system_config_rejects_manager_restart_policy():
    conf = read_system_config(SYNTHETIC_CONFIG_PATH, validate=False)
    conf["manager"] = {"mode": "soft-rtc", "restart_policy": "sometimes"}

    with pytest.raises(ConfigValidationError, match="restart_policy"):
        validate_system_config(conf)


def test_validate_system_config_accepts_manager_supervision_fields(tmp_path):
    conf = read_system_config(SYNTHETIC_CONFIG_PATH, validate=False)
    conf["manager"] = {
        "mode": "hard-rtc",
        "restart_policy": "on-failure",
        "component_restart_policies": {"loop": "always"},
        "health_check_interval": 0.5,
        "heartbeat_timeout": 3.0,
        "rpc_timeout": 0.2,
        "log_dir": str(tmp_path),
        "log_file": str(tmp_path / "manager.log"),
        "component_files": {"loop": "../../pyrtc/loop.py"},
    }

    normalized = validate_system_config(conf, config_path=SYNTHETIC_CONFIG_PATH)

    assert normalized["manager"]["restart_policy"] == "on-failure"
    assert normalized["manager"]["component_restart_policies"]["loop"] == "always"


def test_validate_system_config_rejects_hard_rtc_for_resource_backed_component():
    conf = read_system_config(SYNTHETIC_CONFIG_PATH, validate=False)
    conf["resources"] = {
        "shared": {
            "class_name": "pyrtc.component.Component",
        }
    }
    conf["wfs"]["resource"] = "shared"
    conf["manager"] = {
        "mode": "hard-rtc",
    }

    with pytest.raises(ConfigValidationError, match="soft-rtc"):
        validate_system_config(conf)


def test_validate_system_config_accepts_component_section_as_resource_provider():
    conf = read_system_config(SYNTHETIC_CONFIG_PATH, validate=False)
    conf["shared"] = {
        "class_name": "pyrtc.component.Component",
    }
    conf["wfs"]["resource"] = "shared"
    conf["manager"] = {
        "mode": "soft-rtc",
        "component_classes": {"shared": "pyrtc.component.Component"},
        "component_files": {"shared": "pyrtc/component.py"},
    }

    normalized = validate_system_config(conf)

    assert normalized["wfs"]["resource"] == "shared"


def test_validate_system_config_rejects_component_restart_policy_for_unknown_section():
    conf = read_system_config(SYNTHETIC_CONFIG_PATH, validate=False)
    conf["manager"] = {
        "mode": "soft-rtc",
        "component_restart_policies": {"notAComponent": "on-failure"},
    }

    with pytest.raises(ConfigValidationError, match="component_restart_policies"):
        validate_system_config(conf)


def test_validate_system_config_rejects_signal_stream_shape_mismatch():
    conf = read_system_config(SYNTHETIC_CONFIG_PATH, validate=False)
    conf["streams"] = {
        "signal": {
            "shape": [31],
            "dtype": "float32",
            "output_component": "slopes",
            "input_components": ["loop"],
        }
    }

    with pytest.raises(ConfigValidationError, match="streams.signal"):
        validate_system_config(conf)


def test_validate_system_config_accepts_stream_lineage_overrides():
    conf = read_system_config(SYNTHETIC_CONFIG_PATH, validate=False)
    conf["streams"] = {
        "signal": {
            "shape": [98],
            "dtype": "float32",
            "output_component": "slopes",
            "input_components": ["loop"],
            "sourceStreams": ["wfs"],
            "lineageSource": "wfs",
        },
        "wfc": {
            "shape": [97],
            "dtype": "float32",
            "output_component": "loop",
            "input_components": ["wfc"],
            "sourceStreams": ["signal"],
            "lineageSource": "signal",
        },
    }

    normalized = validate_system_config(conf)

    assert normalized["streams"]["signal"]["lineageSource"] == "wfs"


def test_validate_system_config_accepts_manager_declared_custom_section():
    conf = read_system_config(SYNTHETIC_CONFIG_PATH, validate=False)
    conf["modulator"] = {
        "name": "tutorial-modulator",
        "frequency": 300,
        "amplitude": 600,
    }
    conf["manager"] = {
        "mode": "hard-rtc",
        "component_classes": {"modulator": "pyrtc.component.Component"},
        "component_files": {"modulator": "pyrtc/component.py"},
        "ports": {"modulator": 6001},
    }

    normalized = validate_system_config(conf)

    assert normalized["manager"]["component_classes"]["modulator"] == "pyrtc.component.Component"


def test_validate_config_cli_text_success(capsys):
    code = validate_config_cli.main([str(SYNTHETIC_CONFIG_PATH)])
    captured = capsys.readouterr()

    assert code == 0
    assert "Config valid" in captured.out
    assert "Components: wfs, slopes, loop, wfc, psf" in captured.out


def test_validate_config_cli_json_failure(capsys, tmp_path):
    bad_conf = copy.deepcopy(read_system_config(SYNTHETIC_CONFIG_PATH, validate=False))
    bad_conf.pop("wfc")
    bad_config_path = tmp_path / "bad_config.yaml"
    bad_config_path.write_text(
        "\n".join(
            [
                "wfs:",
                "  name: SyntheticSHWFS",
                "  width: 32",
                "  height: 32",
                "  dark_count: 16",
                "slopes:",
                "  type: SHWFS",
                "  signal_type: slopes",
                "  sub_ap_spacing: 8",
                "  sub_ap_offset_x: 0",
                "  sub_ap_offset_y: 0",
                "loop:",
                "  gain: 0.1",
            ]
        ),
        encoding="utf-8",
    )

    code = validate_config_cli.main([str(bad_config_path), "--format", "json"])
    captured = capsys.readouterr()

    payload = json.loads(captured.out)
    assert code == 1
    assert payload["valid"] is False
    assert "wfc" in payload["error"]


@pytest.mark.parametrize("signal_type", ["phase", "Slopes2", "intensity"])
def test_validate_system_config_rejects_unsupported_signal_type(signal_type):
    conf = read_system_config(SYNTHETIC_CONFIG_PATH, validate=False)
    conf["slopes"]["signal_type"] = signal_type

    with pytest.raises(ConfigValidationError, match="signal_type"):
        validate_system_config(conf, config_path=SYNTHETIC_CONFIG_PATH)


def test_validate_system_config_accepts_slopes_types_case_insensitively():
    conf = read_system_config(SYNTHETIC_CONFIG_PATH, validate=False)
    conf["slopes"]["type"] = "shwfs"
    conf["slopes"]["signal_type"] = "SLOPES"

    validate_system_config(conf, config_path=SYNTHETIC_CONFIG_PATH)


def test_synthetic_example_has_no_config_warnings():
    normalized = read_system_config(SYNTHETIC_CONFIG_PATH)

    assert collect_config_warnings(normalized) == []


def test_collect_config_warnings_reports_unknown_keys():
    conf = read_system_config(SYNTHETIC_CONFIG_PATH, validate=False)
    conf["loop"]["method"] = "push-pull"
    conf["loop"]["_private"] = 1
    conf["wfs"]["frame_rate"] = 10.0  # SyntheticSHWFS reads frame_rate_hz

    normalized = validate_system_config(conf, config_path=SYNTHETIC_CONFIG_PATH)
    warnings = collect_config_warnings(normalized)

    assert warnings == [
        "wfs: unknown config key 'frame_rate' is ignored by SyntheticSHWFS"
        " (did you mean 'frame_rate_hz'?)",
        "loop: unknown config key 'method' is ignored by Loop (did you mean 'im_method'?)",
    ]


def _write_config_with_loop_typo(tmp_path):
    conf = read_system_config(SYNTHETIC_CONFIG_PATH, validate=False)
    conf["loop"]["method"] = "push-pull"
    for section in ("wfs", "slopes", "loop", "wfc", "psf"):
        conf[section].pop("class_file", None)
    conf["manager"].pop("component_files", None)
    conf["loop"].pop("im_file", None)
    config_path = tmp_path / "typo.yaml"
    config_path.write_text(yaml.safe_dump(conf), encoding="utf-8")
    return config_path


def test_validate_config_cli_prints_unknown_key_warnings(capsys, tmp_path):
    config_path = _write_config_with_loop_typo(tmp_path)

    code = validate_config_cli.main([str(config_path)])
    captured = capsys.readouterr()

    assert code == 0
    assert "Config valid" in captured.out
    assert "Warning: loop: unknown config key 'method'" in captured.out


def test_validate_config_cli_json_includes_warnings(capsys, tmp_path):
    config_path = _write_config_with_loop_typo(tmp_path)

    code = validate_config_cli.main([str(config_path), "--format", "json"])
    payload = json.loads(capsys.readouterr().out)

    assert code == 0
    assert payload["valid"] is True
    assert payload["warnings"] == [
        "loop: unknown config key 'method' is ignored by Loop (did you mean 'im_method'?)"
    ]


def test_component_construction_logs_unknown_config_keys(tmp_path):
    from pyrtc.telemetry import Telemetry

    records = []

    class _Collect(logging.Handler):
        def emit(self, record):
            records.append(record)

    logger = logging.getLogger("pyrtc.telemetry.Telemetry")
    handler = _Collect(level=logging.WARNING)
    logger.addHandler(handler)
    try:
        Telemetry(
            {
                "_sectionName": "telemetry",
                "data_dir": str(tmp_path),
                "stream": ["wfs"],
                "class_name": "Telemetry",
            }
        )
    finally:
        logger.removeHandler(handler)

    messages = [record.getMessage() for record in records if record.levelno == logging.WARNING]
    assert messages == [
        "telemetry: unknown config key 'stream' is ignored by Telemetry (did you mean 'streams'?)"
    ]
