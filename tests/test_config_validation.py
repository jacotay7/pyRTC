import copy
import json
from pathlib import Path

import pytest

from pyRTC.config_schema import read_system_config, validate_system_config
from pyRTC.scripts import validate_config as validate_config_cli
from pyRTC.utils import (
    ConfigValidationError,
    validate_loop_config,
    validate_wfc_config,
    validate_wfs_config,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_CONFIG_PATH = REPO_ROOT / "examples" / "synthetic_shwfs" / "config.yaml"


def test_validate_wfs_config_accepts_valid_defaults():
    validate_wfs_config({"width": 16, "height": 16, "darkCount": 10})


def test_validate_wfs_config_rejects_invalid_width():
    with pytest.raises(ConfigValidationError):
        validate_wfs_config({"width": 0, "height": 16})


def test_validate_wfc_config_requires_keys():
    with pytest.raises(ConfigValidationError):
        validate_wfc_config({"numActuators": 97, "numModes": 80})


def test_validate_wfc_config_rejects_bad_modes():
    with pytest.raises(ConfigValidationError):
        validate_wfc_config({"name": "dm", "numActuators": 97, "numModes": 0})


def test_validate_wfc_config_accepts_valid_minimal():
    validate_wfc_config({"name": "dm", "numActuators": 97, "numModes": 80})


def test_validate_loop_config_allows_negative_gain():
    validate_loop_config({"gain": -0.1, "pGain": -0.2, "iGain": -0.3, "dGain": -0.4})


def test_validate_loop_config_rejects_bad_limits_shape():
    with pytest.raises(ConfigValidationError):
        validate_loop_config({"controlLimits": [0.0, 1.0, 2.0]})


def test_validate_loop_config_accepts_valid_config():
    validate_loop_config(
        {
            "CMMethod": "tikhonov",
            "conditioning": 1000,
            "gain": 0.1,
            "leakyGain": 0.01,
            "numDroppedModes": 0,
            "tikhonovReg": 0.001,
            "controlLimits": [-1.0, 1.0],
            "integralLimits": [-0.5, 0.5],
            "absoluteLimits": [-2.0, 2.0],
        }
    )


def test_validate_system_config_accepts_float_shwfs_subap_spacing():
    conf = read_system_config(SYNTHETIC_CONFIG_PATH, validate=False)
    conf["slopes"]["subApSpacing"] = 8.5

    normalized = validate_system_config(conf, config_path=SYNTHETIC_CONFIG_PATH)

    assert normalized["slopes"]["subApSpacing"] == 8.5


def test_validate_loop_config_rejects_bad_cm_method():
    with pytest.raises(ConfigValidationError, match="CMMethod"):
        validate_loop_config({"CMMethod": "ridge-ish"})


def test_validate_loop_config_rejects_bad_tikhonov_regularization():
    with pytest.raises(ConfigValidationError, match="tikhonovReg"):
        validate_loop_config({"tikhonovReg": -0.1})


def test_validate_system_config_accepts_synthetic_example():
    normalized = read_system_config(SYNTHETIC_CONFIG_PATH)

    assert normalized["manager"]["mode"] == "soft-rtc"
    assert normalized["wfc"]["numModes"] == 32
    assert Path(normalized["metadata"]["configPath"]).resolve() == SYNTHETIC_CONFIG_PATH.resolve()


def test_validate_system_config_resolves_relative_file_paths_against_config_file():
    conf = read_system_config(SYNTHETIC_CONFIG_PATH, validate=False)
    conf["loop"]["IMFile"] = "synthetic_identity_im.npy"
    conf["manager"] = {"mode": "hard-rtc", "componentFiles": {"loop": "../../pyRTC/Loop.py"}}

    normalized = validate_system_config(conf, config_path=SYNTHETIC_CONFIG_PATH)

    assert normalized["loop"]["IMFile"] == str((SYNTHETIC_CONFIG_PATH.parent / "synthetic_identity_im.npy").resolve())
    assert normalized["manager"]["componentFiles"]["loop"] == str((SYNTHETIC_CONFIG_PATH.parent / "../../pyRTC/Loop.py").resolve())


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
    conf["psf"]["darkCount"] = "16"

    with pytest.raises(ConfigValidationError, match="darkCount"):
        validate_system_config(conf)


def test_validate_system_config_rejects_too_many_dropped_modes():
    conf = read_system_config(SYNTHETIC_CONFIG_PATH, validate=False)
    conf["loop"]["numDroppedModes"] = conf["wfc"]["numModes"]

    with pytest.raises(ConfigValidationError, match="numDroppedModes"):
        validate_system_config(conf)


def test_validate_system_config_rejects_manager_restart_policy():
    conf = read_system_config(SYNTHETIC_CONFIG_PATH, validate=False)
    conf["manager"] = {"mode": "soft-rtc", "restartPolicy": "sometimes"}

    with pytest.raises(ConfigValidationError, match="restartPolicy"):
        validate_system_config(conf)


def test_validate_system_config_accepts_manager_supervision_fields(tmp_path):
    conf = read_system_config(SYNTHETIC_CONFIG_PATH, validate=False)
    conf["manager"] = {
        "mode": "hard-rtc",
        "restartPolicy": "on-failure",
        "componentRestartPolicies": {"loop": "always"},
        "healthCheckInterval": 0.5,
        "heartbeatTimeout": 3.0,
        "rpcTimeout": 0.2,
        "logDir": str(tmp_path),
        "logFile": str(tmp_path / "manager.log"),
        "componentFiles": {"loop": "../../pyRTC/Loop.py"},
    }

    normalized = validate_system_config(conf, config_path=SYNTHETIC_CONFIG_PATH)

    assert normalized["manager"]["restartPolicy"] == "on-failure"
    assert normalized["manager"]["componentRestartPolicies"]["loop"] == "always"


def test_validate_system_config_rejects_component_restart_policy_for_unknown_section():
    conf = read_system_config(SYNTHETIC_CONFIG_PATH, validate=False)
    conf["manager"] = {
        "mode": "soft-rtc",
        "componentRestartPolicies": {"notAComponent": "on-failure"},
    }

    with pytest.raises(ConfigValidationError, match="componentRestartPolicies"):
        validate_system_config(conf)


def test_validate_system_config_rejects_signal_stream_shape_mismatch():
    conf = read_system_config(SYNTHETIC_CONFIG_PATH, validate=False)
    conf["streams"] = {
        "signal": {"shape": [31], "dtype": "float32", "outputComponent": "slopes", "inputComponents": ["loop"]}
    }

    with pytest.raises(ConfigValidationError, match="streams.signal"):
        validate_system_config(conf)


def test_validate_system_config_accepts_stream_lineage_overrides():
    conf = read_system_config(SYNTHETIC_CONFIG_PATH, validate=False)
    conf["streams"] = {
        "signal": {
            "shape": [32],
            "dtype": "float32",
            "outputComponent": "slopes",
            "inputComponents": ["loop"],
            "sourceStreams": ["wfs"],
            "lineageSource": "wfs",
        },
        "wfc": {
            "shape": [32],
            "dtype": "float32",
            "outputComponent": "loop",
            "inputComponents": ["wfc"],
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
        "componentClasses": {"modulator": "pyRTC.pyRTCComponent.pyRTCComponent"},
        "componentFiles": {"modulator": "pyRTC/pyRTCComponent.py"},
        "ports": {"modulator": 6001},
    }

    normalized = validate_system_config(conf)

    assert normalized["manager"]["componentClasses"]["modulator"] == "pyRTC.pyRTCComponent.pyRTCComponent"


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
                "  darkCount: 16",
                "slopes:",
                "  type: SHWFS",
                "  signalType: slopes",
                "  subApSpacing: 8",
                "  subApOffsetX: 0",
                "  subApOffsetY: 0",
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
