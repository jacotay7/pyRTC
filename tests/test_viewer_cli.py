from pyRTC.scripts import view
from pyRTC.scripts import clear_shms, view_launch_all


def test_clear_shms_default(monkeypatch):
    called = {}

    def _clear(names):
        called["names"] = list(names)

    monkeypatch.setattr(clear_shms.Pipeline, "clear_shms", _clear)
    code = clear_shms.main([])

    assert code == 0
    assert called["names"] == clear_shms.DEFAULT_SHM_NAMES


def test_clear_shms_custom(monkeypatch):
    called = {}

    def _clear(names):
        called["names"] = list(names)

    monkeypatch.setattr(clear_shms.Pipeline, "clear_shms", _clear)
    clear_shms.main(["foo", "bar"])

    assert called["names"] == ["foo", "bar"]


def test_view_launch_all_uses_pyrtc_view_commands():
    spawned = []

    def _popen(cmd):
        spawned.append(cmd)
        return object()

    view_launch_all.launch_all(popen_fn=_popen)

    assert len(spawned) == len(view_launch_all.DEFAULT_VIEW_COMMANDS)
    assert all(cmd[0] == "pyrtc-view" for cmd in spawned)


def test_view_launch_all_main_invokes_launcher(monkeypatch):
    launched = {"ok": False}

    def _launch_all(commands=None, popen_fn=None):
        launched["ok"] = True
        return []

    monkeypatch.setattr(view_launch_all, "launch_all", _launch_all)
    code = view_launch_all.main([])

    assert code == 0
    assert launched["ok"]


def test_view_split_targets_and_limits_supports_legacy_vmin_vmax():
    shms, vmin, vmax = view._split_targets_and_limits(["signal2D", "-1", "1"])

    assert shms == ["signal2D"]
    assert vmin == -1.0
    assert vmax == 1.0


def test_view_split_targets_and_limits_supports_multiple_shms():
    shms, vmin, vmax = view._split_targets_and_limits(["wfs", "signal2D", "wfc2D"])

    assert shms == ["wfs", "signal2D", "wfc2D"]
    assert vmin is None
    assert vmax is None


def test_view_resolve_grid_variants():
    assert view._resolve_grid(3, "row") == (1, 3)
    assert view._resolve_grid(3, "column") == (3, 1)
    assert view._resolve_grid(5, "square") == (2, 3)
    assert view._resolve_grid(5, "2x3") == (2, 3)


def test_view_rejects_too_small_explicit_grid():
    try:
        view._resolve_grid(5, "2x2")
    except ValueError as exc:
        assert "does not have enough cells" in str(exc)
    else:
        raise AssertionError("Expected ValueError for undersized explicit grid")
