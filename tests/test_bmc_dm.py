"""Boston Micromachines DM adapter against a fake BMC SDK (#70)."""

import sys
import types

import numpy as np
import pytest

from testsupport import private_stream


class _BmcDm:
    instances = []
    actuators = 140
    open_error = 0

    def __init__(self):
        self.sent, self.closed, self.serial = [], False, None
        _BmcDm.instances.append(self)

    def open_dm(self, serial):
        self.serial = serial
        return _BmcDm.open_error

    def num_actuators(self):
        return _BmcDm.actuators

    def send_data(self, values):
        self.sent.append(np.array(values, copy=True))
        return 0

    def close_dm(self):
        self.closed = True

    def error_string(self, error):
        return "NO_DEVICE" if error == 5 else "?"


@pytest.fixture
def bmc(monkeypatch):
    _BmcDm.instances.clear()
    _BmcDm.actuators, _BmcDm.open_error = 140, 0
    monkeypatch.setitem(sys.modules, "bmc", types.SimpleNamespace(BmcDm=_BmcDm))
    import pyrtc.hardware.bmc_dm as module
    import pyrtc.wavefront_corrector as wavefront_corrector

    monkeypatch.setattr(wavefront_corrector, "create_stream", private_stream)
    return module


def _conf(**extra):
    return {"name": "wfc", "serial": "MultiDM-01", "num_modes": 4, "functions": [], **extra}


def test_layouts_of_standard_bmc_mirrors(bmc):
    multi = bmc.bmc_actuator_layout(140)
    assert multi.shape == (12, 12) and multi.sum() == 140
    assert not multi[0, 0] and not multi[-1, -1] and multi[0, 1]
    kilo = bmc.bmc_actuator_layout(1020)
    assert kilo.shape == (32, 32) and kilo.sum() == 1020
    assert bmc.bmc_actuator_layout(144).all()
    circular = bmc.bmc_actuator_layout(97)  # not a square family: centred disk
    assert circular.sum() == 97 and circular.shape == (11, 11)


def test_dm_opens_maps_commands_and_closes_unpowered(bmc):
    dm = bmc.BMCDM(_conf(num_actuators=12, command_cap=0.8))  # wrong count: overridden
    sdk = _BmcDm.instances[-1]
    try:
        assert sdk.serial == "MultiDM-01"
        assert dm.num_actuators == 140 and dm.layout.shape == (12, 12)
        dm.set_m2c(np.eye(140, 4, dtype=np.float32))
        dm.write(np.array([1.0, -1.0, 0.5, 0.0], dtype=np.float32))
        dm.send_to_hardware()
        values = sdk.sent[-1]
        assert values.dtype == np.float64 and values.shape == (140,)
        # bias 0.5 + 0.5 * command, after the 0.8 cap
        np.testing.assert_allclose(values[:4], [0.9, 0.1, 0.75, 0.5])
        np.testing.assert_allclose(values[4:], 0.5)
    finally:
        dm.close()
    assert sdk.closed
    np.testing.assert_array_equal(sdk.sent[-1], np.zeros(140))


def test_bias_scale_and_clipping(bmc):
    dm = bmc.BMCDM(_conf(bias=0.2, command_scale=1.0))
    try:
        np.testing.assert_allclose(
            dm.to_dm_values(np.array([-1.0, 0.3, 2.0] + [0.0] * 137)[:140])[:3], [0.0, 0.5, 1.0]
        )
    finally:
        dm.close()


def test_open_failure_reports_the_sdk_error(bmc):
    _BmcDm.open_error = 5
    with pytest.raises(RuntimeError, match="NO_DEVICE"):
        bmc.BMCDM(_conf())


def test_layout_file_must_match_the_mirror(bmc, tmp_path):
    path = tmp_path / "layout.npy"
    np.save(path, np.ones((10, 10), dtype=bool))
    with pytest.raises(ValueError, match="layout_file has 100"):
        bmc.BMCDM(_conf(layout_file=str(path)))
    assert _BmcDm.instances[-1].closed  # a failed init still releases the mirror


@pytest.mark.parametrize(
    ("key", "value", "match"), [("bias", 1.5, "bias"), ("command_scale", 0, "command_scale")]
)
def test_invalid_mapping_is_rejected(bmc, key, value, match):
    with pytest.raises(ValueError, match=match):
        bmc.BMCDM(_conf(**{key: value}))


def test_missing_sdk_names_it(monkeypatch):
    monkeypatch.setitem(sys.modules, "bmc", None)
    import pyrtc.hardware.bmc_dm as module

    with pytest.raises(ImportError, match="BMC DM SDK"):
        module._load_bmc_module()
