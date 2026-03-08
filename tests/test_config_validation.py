import pytest

from pyRTC.utils import (
    ConfigValidationError,
    validate_loop_config,
    validate_wfc_config,
    validate_wfs_config,
)


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
            "gain": 0.1,
            "leakyGain": 0.01,
            "numDroppedModes": 0,
            "controlLimits": [-1.0, 1.0],
            "integralLimits": [-0.5, 0.5],
            "absoluteLimits": [-2.0, 2.0],
        }
    )
