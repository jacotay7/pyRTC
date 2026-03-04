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
