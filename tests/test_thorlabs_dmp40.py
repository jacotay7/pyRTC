"""Thorlabs DMP40 adapter against a fake dmp40 driver."""

import sys
import types

import numpy as np
import pytest

from testsupport import private_stream

BASELINE = np.linspace(90.0, 127.0, 40)
TILT = [100.0, 100.0, 100.0]


class _DMP40:
    instances = []
    next_measured = BASELINE

    def __init__(self, serial=None, index=0):
        self.requested_serial, self.sent, self.closed = serial, [], False
        self.measured = _DMP40.next_measured.copy()
        _DMP40.instances.append(self)

    def get_serial_number(self):
        return "M0000001"

    def get_instrument_name(self):
        return "DMP40"

    def get_measured_segment_voltages(self):
        return list(self.measured)

    def get_temperatures(self):
        return [30.0, 31.0, 32.7, 36.9]

    def set_voltages(self, segment_volts, tilt_volts):
        assert len(segment_volts) == 40 and len(tilt_volts) == 3
        self.sent.append((np.array(segment_volts), list(tilt_volts)))
        self.measured = np.array(segment_volts)

    def close(self):
        self.closed = True


@pytest.fixture
def dmp40(monkeypatch):
    _DMP40.instances.clear()
    _DMP40.next_measured = BASELINE.copy()
    monkeypatch.setitem(sys.modules, "dmp40", types.SimpleNamespace(DMP40=_DMP40))
    import pyrtc.hardware.thorlabs_dmp40 as module
    import pyrtc.wavefront_corrector as wavefront_corrector

    monkeypatch.setattr(wavefront_corrector, "create_stream", private_stream)
    monkeypatch.setattr(module, "create_stream", private_stream)
    return module


def _conf(**extra):
    return {"name": "wfc", "num_modes": 40, "functions": [], "tilt_volts": TILT, **extra}


def _send(dm, correction):
    dm.write(np.asarray(correction, dtype=np.float32))
    dm.send_to_hardware()
    return _DMP40.instances[-1].sent[-1]


def test_tilt_volts_are_required(dmp40):
    with pytest.raises(ValueError, match="tilt_volts is required"):
        dmp40.ThorlabsDMP40({"name": "wfc", "num_modes": 40, "functions": []})
    with pytest.raises(ValueError, match="tilt_volts must be finite"):
        dmp40.ThorlabsDMP40(_conf(tilt_volts=[100.0, 100.0, 250.0]))
    assert not _DMP40.instances  # rejected before the mirror was opened


def test_a_mirror_open_elsewhere_reports_busy(dmp40, monkeypatch):
    def busy(serial=None, index=0):
        try:
            raise OSError(16, "Resource busy")
        except OSError as cause:
            raise RuntimeError(
                "Could not open the DMP40 USB device. Install the udev rule"
            ) from cause

    monkeypatch.setattr(sys.modules["dmp40"], "DMP40", busy)
    with pytest.raises(RuntimeError, match="open in another program"):
        dmp40.ThorlabsDMP40(_conf())


def test_open_writes_nothing_and_holds_the_measured_shape(dmp40):
    dm = dmp40.ThorlabsDMP40(_conf(serial="M0000001", num_actuators=12))
    driver = _DMP40.instances[-1]
    try:
        assert driver.requested_serial == "M0000001" and not driver.sent
        assert dm.num_actuators == 40 and dm.command_cap == 89.0
        np.testing.assert_allclose(dm.flat, BASELINE - 100.0, atol=1e-5)
        np.testing.assert_allclose(dm.current_shape, dm.flat)

        segments, tilt = _send(dm, np.zeros(40))
        np.testing.assert_allclose(segments, BASELINE, atol=1e-4)
        assert tilt == TILT

        correction = np.zeros(40)
        correction[3] = 1.5
        segments, _ = _send(dm, correction)
        np.testing.assert_allclose(segments[3], BASELINE[3] + 1.5, atol=1e-4)
        np.testing.assert_allclose(np.delete(segments, 3), np.delete(BASELINE, 3), atol=1e-4)
        np.testing.assert_allclose(dm.readback(), segments)
        assert dm.temperatures()[2] == 32.7
    finally:
        dm.close()
    # Closing restores the frame measured at open, and never relaxes the mirror.
    segments, tilt = driver.sent[-1]
    np.testing.assert_allclose(segments, BASELINE)
    assert tilt == TILT and driver.closed


def test_step_and_offset_limits_clip_commands(dmp40):
    dm = dmp40.ThorlabsDMP40(_conf(max_step_v=2.0, max_offset_v=6.0))
    try:
        correction = np.zeros(40)
        correction[0], correction[1] = 10.0, -1.0
        segments, _ = _send(dm, correction)
        assert segments[0] == pytest.approx(BASELINE[0] + 2.0, abs=1e-4)  # slew
        assert segments[1] == pytest.approx(BASELINE[1] - 1.0, abs=1e-4)
        assert dm.limited_actuators == 1
        np.testing.assert_allclose(dm.current_shape[0], BASELINE[0] + 2.0 - 100.0, atol=1e-4)
        for _ in range(5):
            segments, _ = _send(dm, correction)
        assert segments[0] == pytest.approx(BASELINE[0] + 6.0, abs=1e-4)  # offset envelope
    finally:
        dm.close()


def test_command_cap_and_hardware_range(dmp40):
    dm = dmp40.ThorlabsDMP40(_conf(command_cap=30.0))
    try:
        segments, _ = _send(dm, np.full(40, 500.0))
        np.testing.assert_allclose(segments, 130.0, atol=1e-4)
    finally:
        dm.close()


def test_a_cap_that_would_clip_the_flat_is_refused(dmp40):
    with pytest.raises(ValueError, match="outside command_cap 20.0"):
        dmp40.ThorlabsDMP40(_conf(command_cap=20.0))  # the flat reaches 127 V
    assert not _DMP40.instances[-1].sent and _DMP40.instances[-1].closed


def test_close_without_commands_writes_nothing_and_hold_keeps_the_shape(dmp40):
    dmp40.ThorlabsDMP40(_conf()).close()
    assert not _DMP40.instances[-1].sent and _DMP40.instances[-1].closed

    dm = dmp40.ThorlabsDMP40(_conf(on_close="hold"))
    _send(dm, np.full(40, 1.0))
    dm.close()
    driver = _DMP40.instances[-1]
    assert len(driver.sent) == 1
    np.testing.assert_allclose(driver.sent[-1][0], BASELINE + 1.0, atol=1e-4)


def test_flat_file_overrides_the_measured_shape(dmp40, tmp_path):
    flat = tmp_path / "flat.npy"
    np.save(flat, np.full(40, 5.0))
    dm = dmp40.ThorlabsDMP40(_conf(flat_file=str(flat)))
    try:
        segments, _ = _send(dm, np.zeros(40))
        np.testing.assert_allclose(segments, 105.0, atol=1e-4)
    finally:
        dm.close()


def test_limits_only_tighten_and_tilt_is_fixed(dmp40):
    dm = dmp40.ThorlabsDMP40(_conf(max_step_v=2.0))
    try:
        dm.max_step_v = 1.0  # tightening is fine
        dm.max_offset_v = 4.0  # so is adding a limit
        dm.command_cap = 50.0
        for name, value in (("max_step_v", 3.0), ("max_offset_v", None), ("command_cap", 89.0)):
            with pytest.raises(ValueError, match="only be tightened"):
                setattr(dm, name, value)
        assert (dm.max_step_v, dm.max_offset_v, dm.command_cap) == (1.0, 4.0, 50.0)
        with pytest.raises(AttributeError):
            dm.tilt_volts = [0.0, 0.0, 0.0]
        with pytest.raises(AttributeError):
            dm.bias_v = 0.0
        dm.tilt_volts[0] = 0.0  # a copy: the frame sent is unchanged
        _, tilt = _send(dm, np.zeros(40))
        assert tilt == TILT
    finally:
        dm.close()


def test_clipped_writes_are_counted(dmp40):
    dm = dmp40.ThorlabsDMP40(_conf(max_step_v=2.0, command_cap=30.0))
    try:
        _send(dm, np.zeros(40))
        assert (dm.writes, dm.limited_writes) == (1, 0)
        step = np.zeros(40)
        step[0] = 3.0  # beyond max_step_v
        _send(dm, step)
        _send(dm, np.full(40, 500.0))  # beyond command_cap
        assert (dm.writes, dm.limited_writes) == (3, 2)
    finally:
        dm.close()


def test_a_step_of_exactly_max_step_v_is_not_clipped(dmp40):
    # Read-back voltages are not float32-representable, so +1 V -> -1 V used
    # to come out a few tens of nV over a 2 V limit; 100.776699... V is a
    # measured read-back where that happened.
    _DMP40.next_measured = BASELINE.copy()
    _DMP40.next_measured[13] = 100.77669902912622
    dm = dmp40.ThorlabsDMP40(_conf(max_step_v=2.0, max_offset_v=6.0))
    try:
        for amplitude in (0.0, 1.0, -1.0, 1.0, -1.0, 0.0):
            command = np.zeros(40)
            command[13] = amplitude
            segments, _ = _send(dm, command)
            assert segments[13] == pytest.approx(100.77669902912622 + amplitude, abs=1e-5)
        assert dm.limited_writes == 0
        command[13] = 2.5  # genuinely beyond the step limit from 0
        _send(dm, command)
        assert dm.limited_writes == 1
    finally:
        dm.close()


def test_wfc_volts_reports_every_write_for_stream_clients(dmp40):
    dm = dmp40.ThorlabsDMP40(_conf(max_step_v=2.0))
    try:
        stream = dm._stream_object("wfc_volts")
        record = np.asarray(stream.read())
        assert record[:2].tolist() == [0, 0]  # at open: the measured frame
        np.testing.assert_allclose(record[2:], BASELINE)
        command = np.zeros(40)
        command[5] = 1.0
        _send(dm, command)
        record = np.asarray(stream.read())
        assert record[:2].tolist() == [1, 0]
        assert record[2 + 5] == pytest.approx(BASELINE[5] + 1.0, abs=1e-5)
        command[5] = 4.0  # beyond max_step_v
        _send(dm, command)
        assert np.asarray(stream.read())[:2].tolist() == [2, 1]
    finally:
        dm.close()  # the restore write after close must not touch the closed stream


def test_an_index_map_layout_file_draws_the_rings(dmp40, tmp_path):
    index_map = -np.ones((10, 10), dtype=np.int16)
    index_map.flat[np.random.default_rng(0).permutation(100)[:40]] = np.arange(40)
    index_map[0, 0] = 7  # segment 7 on two cells
    path = tmp_path / "map.npy"
    np.save(path, index_map)
    dm = dmp40.ThorlabsDMP40(_conf(layout_file=str(path)))
    try:
        command = np.zeros(40)
        command[7] = 1.5
        _send(dm, command)
        wfc2d = np.asarray(dm._stream_object("wfc_2d").read())
        np.testing.assert_allclose(wfc2d[index_map == 7], 1.5, atol=1e-5)
        np.testing.assert_allclose(wfc2d[index_map != 7], 0.0, atol=1e-5)
    finally:
        dm.close()


def test_restore_on_close_steps_within_max_step_v(dmp40):
    dm = dmp40.ThorlabsDMP40(_conf(max_step_v=2.0, max_offset_v=6.0))
    for amplitude in (2.0, 4.0, 6.0):  # walk segment 3 out to +6 V, 2 V per write
        command = np.zeros(40)
        command[3] = amplitude
        _send(dm, command)
    driver = _DMP40.instances[-1]
    sent_before = len(driver.sent)
    dm.close()
    restore = [segments[3] for segments, _ in driver.sent[sent_before:]]
    assert restore[-1] == pytest.approx(BASELINE[3], abs=1e-6)
    steps = np.abs(np.diff([BASELINE[3] + 6.0] + restore))
    assert len(restore) == 3 and np.all(steps <= 2.0 + 1e-6)


def test_limits_reject_nan_and_negative(dmp40):
    dm = dmp40.ThorlabsDMP40(_conf(max_step_v=2.0))
    try:
        for bad in (float("nan"), -5.0):
            with pytest.raises(ValueError, match="finite, non-negative"):
                dm.max_step_v = bad
        assert dm.max_step_v == 2.0
    finally:
        dm.close()
