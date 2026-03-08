from pyRTC.scripts import view
from pyRTC.scripts import clear_shms, view_launch_all
from pyRTC.scripts.viewer_helpers import StreamConnection
from pyRTC.scripts.viewer_helpers import format_shape
from pyRTC.scripts.viewer_core import MosaicViewerWindow


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
    assert spawned[0][1:6] == ["wfs", "signal2D", "wfc2D", "psfShort", "psfLong"]


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


def test_view_parser_supports_theme_flag():
    parser = view._build_arg_parser()
    args = parser.parse_args(["wfs", "signal2D", "--geometry", "row", "--theme", "light"])

    assert args.items == ["wfs", "signal2D"]
    assert args.geometry == "row"
    assert args.theme == "light"


def test_view_compute_window_size_starts_smaller_than_previous_large_defaults():
    width, height = view._compute_window_size([
        view._normalize_frame(__import__("numpy").zeros((64, 64))),
        view._normalize_frame(__import__("numpy").zeros((64, 64))),
        view._normalize_frame(__import__("numpy").zeros((64, 64))),
        view._normalize_frame(__import__("numpy").zeros((64, 64))),
    ], 2, 2, 12.0)

    assert width <= 1320
    assert height <= 900


def test_stream_connection_reports_paused_without_new_frame():
    class _Meta:
        def __init__(self):
            self.calls = 0

        def read_noblock(self):
            self.calls += 1
            if self.calls == 1:
                return [1, 10.0]
            return [1, 10.0]

        def close(self):
            return None

    class _Shm:
        def read_noblock(self):
            return [[1.0, 2.0], [3.0, 4.0]]

        def close(self):
            return None

    connection = StreamConnection.__new__(StreamConnection)
    connection.name = "signal2D"
    connection.metadata_shm = _Meta()
    connection.shm = _Shm()
    connection.last_count = 1
    connection.last_time = 10.0
    connection.last_fps_text = "1.0 FPS"
    connection.cached_frame = view._normalize_frame([[1.0, 2.0], [3.0, 4.0]])

    snapshot = connection.poll()

    assert snapshot["changed"] is False
    assert snapshot["status_changed"] is True
    assert snapshot["fps_text"] == "PAUSED"


def test_format_shape_joins_all_dimensions():
    assert format_shape((8, 8)) == "8x8"
    assert format_shape((64, 32, 4)) == "64x32x4"


def test_apply_to_panels_collects_errors_without_raising():
    class _Panel:
        def __init__(self, name, should_fail=False):
            self.connection = type("Connection", (), {"name": name})()
            self.should_fail = should_fail
            self.calls = 0

        def toggle(self):
            self.calls += 1
            if self.should_fail:
                raise RuntimeError("boom")

    window = MosaicViewerWindow.__new__(MosaicViewerWindow)
    ok_panel = _Panel("ok")
    bad_panel = _Panel("bad", should_fail=True)
    window.panels = {0: ok_panel, 1: bad_panel}
    window._last_panel_errors = 0

    error_count = MosaicViewerWindow._apply_to_panels(window, "toggle", lambda panel: panel.toggle())

    assert error_count == 1
    assert window._last_panel_errors == 1
    assert ok_panel.calls == 1
    assert bad_panel.calls == 1


def test_stream_connection_close_is_idempotent():
    class _Handle:
        def __init__(self):
            self.calls = 0

        def close(self):
            self.calls += 1

    connection = StreamConnection.__new__(StreamConnection)
    connection.shm = _Handle()
    connection.metadata_shm = _Handle()
    connection._closed = False

    connection.close()
    connection.close()

    assert connection.shm.calls == 1
    assert connection.metadata_shm.calls == 1
