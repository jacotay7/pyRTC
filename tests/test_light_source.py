"""LightSource safety rules against a fake device."""

import pytest

from pyrtc.light_source import LightSource, LightSourceError
from pyrtc.manager import DEFAULT_COMPONENT_ORDER


class _FakeSource(LightSource):
    """A device that records calls; ``stuck`` makes emission ignore commands."""

    def _open(self, conf):
        self.device = conf.get("device") or {"emission": False, "power": 0.5}
        self.calls = []
        self.closed = False
        if conf.get("fail_open"):
            raise OSError("no such port")

    def _read_status(self):
        return dict(self.device)

    def _set_power(self, power):
        self.calls.append(("power", power))
        self.device["power"] = power

    def _set_emission(self, enabled):
        self.calls.append(("emission", enabled))
        if not self.device.get("stuck"):
            self.device["emission"] = enabled

    def _close(self):
        self.closed = self.device["closed"] = True


def _conf(**extra):
    return {"name": "laser", "max_power": 1.0, "functions": [], **extra}


def test_max_power_is_required_and_positive():
    with pytest.raises(ValueError, match="max_power"):
        _FakeSource({"name": "laser", "functions": []})
    with pytest.raises(ValueError, match="positive"):
        _FakeSource(_conf(max_power=0.0))


def test_opening_never_emits_and_turns_a_lit_source_off():
    source = _FakeSource(_conf(device={"emission": True, "power": 0.5}, power=0.2))
    try:
        assert source.calls == [("emission", False), ("power", 0.2)]
        assert not source.emission and source.power == 0.2
    finally:
        source.close()


def test_power_is_capped_and_enable_checks_the_device_setpoint():
    source = _FakeSource(_conf())
    try:
        with pytest.raises(ValueError, match="outside 0-1.0 mW"):
            source.set_power(5.0)
        with pytest.raises(ValueError):
            source.power = float("nan")
        assert ("power", 5.0) not in source.calls

        source.device["power"] = 100.0  # set by some other program
        with pytest.raises(LightSourceError, match="set the power first"):
            source.enable()
        assert not source.device["emission"]

        source.power = 0.5
        source.enable()
        assert source.emission
        source.disable()
        assert not source.emission
    finally:
        source.close()


def test_stop_and_close_turn_emission_off():
    source = _FakeSource(_conf())
    source.enable()
    source.stop()
    assert not source.device["emission"]
    source.enable()
    source.close()
    assert not source.device["emission"] and source.closed
    source.close()  # idempotent


def test_readback_mismatch_raises():
    source = _FakeSource(_conf())
    try:
        source.device["stuck"] = True
        with pytest.raises(LightSourceError, match="did not turn on"):
            source.enable()
        source.device.update(stuck=False, emission=True)
        source.device["stuck"] = True
        with pytest.raises(LightSourceError, match="did not turn off"):
            source.disable()
    finally:
        source.device["stuck"] = False
        source.close()


def test_max_power_can_only_be_lowered():
    source = _FakeSource(_conf())
    try:
        source.max_power = 0.5
        with pytest.raises(ValueError, match="only be tightened"):
            source.max_power = 1.0
        with pytest.raises(ValueError, match="outside 0-0.5 mW"):
            source.set_power(0.8)
    finally:
        source.close()


def test_a_failed_open_releases_the_device():
    device = {"emission": False, "power": 0.5}
    with pytest.raises(OSError):
        _FakeSource(_conf(fail_open=True, device=device))
    assert device["closed"]


def test_source_is_built_last_so_it_is_torn_down_first():
    assert DEFAULT_COMPONENT_ORDER[-1] == "source"


def test_closing_writes_only_when_emission_is_on():
    source = _FakeSource(_conf())
    source.close()
    assert ("emission", False) not in source.calls  # already off: read-only

    unreadable = _FakeSource(_conf())

    def broken_status():
        raise OSError("no reply")

    unreadable._read_status = broken_status
    unreadable.close()
    assert ("emission", False) in unreadable.calls  # unknown state: turn it off


def test_a_setpoint_that_reads_back_wrong_raises():
    source = _FakeSource(_conf())
    try:
        source._set_power = lambda power: source.device.update(power=power * 10)  # wrong units
        with pytest.raises(LightSourceError, match="reads back"):
            source.set_power(0.05)
    finally:
        source.close()


def test_max_power_rejects_nan_and_negative_values():
    source = _FakeSource(_conf())
    try:
        for bad in (float("nan"), -1.0, float("inf")):
            with pytest.raises(ValueError):
                source.max_power = bad
        assert source.max_power == 1.0
    finally:
        source.close()


def test_the_source_is_built_after_any_other_section():
    from pyrtc.manager import RTCManager

    manager = RTCManager.__new__(RTCManager)
    manager.config = {"source": {}, "wfs": {}, "extra_camera": {}, "manager": {}}
    assert manager._component_sections()[-1] == "source"
