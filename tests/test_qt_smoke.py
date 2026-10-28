"""Offscreen smoke tests for the Qt6 manager GUI and stream viewer.

These build the real windows on Qt's ``offscreen`` platform and render frames
from a private pyshmem stream. They skip when qtpy or a Qt6 binding (PySide6
or PyQt6) is not installed, as in the default CI test install.
"""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from pyrtc.qt_compat import require_qt6, unavailable_qt_class
from testsupport import private_stream

REPO_ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_CONFIG_PATH = REPO_ROOT / "examples" / "synthetic_shwfs" / "config.yaml"

try:
    QT_BINDING = require_qt6()
except ImportError as exc:  # pragma: no cover - depends on the environment
    QT_BINDING = None
    _QT_SKIP_REASON = f"Qt6 GUI dependencies unavailable: {exc}"
else:
    _QT_SKIP_REASON = ""

requires_qt6 = pytest.mark.skipif(QT_BINDING is None, reason=_QT_SKIP_REASON)


@pytest.fixture(scope="module")
def qapp():
    # Must be set before the QApplication exists; without a display the default
    # platform plugin aborts the process instead of raising.
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from qtpy.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app
    app.processEvents()


def _process_events(app, rounds=5):
    for _ in range(rounds):
        app.processEvents()


def test_unavailable_qt_class_imports_but_refuses_to_instantiate():
    cause = ImportError("no binding")
    stand_in = unavailable_qt_class("install the gui extra", cause)

    class Window(stand_in):
        pass

    flags = stand_in.AlignmentFlag.AlignLeft | stand_in.AlignmentFlag.AlignTop
    assert flags is stand_in.AlignmentFlag
    with pytest.raises(ImportError, match="install the gui extra") as excinfo:
        Window()
    assert excinfo.value.__cause__ is cause


@pytest.mark.skipif(
    importlib.util.find_spec("qtpy") is None or importlib.util.find_spec("PyQt5") is None,
    reason="needs qtpy and PyQt5 installed",
)
def test_require_qt6_rejects_qt5_binding():
    env = dict(os.environ, QT_API="pyqt5")
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from pyrtc.qt_compat import require_qt6\n"
            "try:\n    require_qt6()\nexcept ImportError as exc:\n    print('rejected', exc)\n",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.startswith("rejected"), completed.stdout
    if "qtpy selected" not in completed.stdout:
        pytest.skip(f"qtpy could not load PyQt5: {completed.stdout.strip()}")
    assert "PyQt5" in completed.stdout


@requires_qt6
def test_require_qt6_prefers_a_qt6_binding():
    import qtpy

    assert qtpy.QT6
    assert QT_BINDING in {"PySide6", "PyQt6"}


@requires_qt6
def test_viewer_renders_private_stream_offscreen(qapp):
    from pyrtc.scripts import viewer_core
    from qtpy.QtGui import QAction

    stream = private_stream("qtview", (8, 12), np.float32)
    first = np.arange(8 * 12, dtype=np.float32).reshape(8, 12)
    stream.write(first)

    window = viewer_core.MosaicViewerWindow(
        [stream.name], fps=30, geometry="square", pixel_scale=12.0
    )
    try:
        window.show()
        _process_events(qapp)
        assert window.isVisible()
        assert sorted(window.panels) == [0]
        assert window.placeholders == {}
        panel = window.panels[0]
        assert isinstance(panel.colorbar_action, QAction)
        np.testing.assert_array_equal(np.asarray(panel.image.get_array()), first)

        second = np.flipud(first) * 2.0
        stream.write(second)
        window.refresh_panels()
        panel.canvas.draw()
        np.testing.assert_array_equal(np.asarray(panel.image.get_array()), second)

        pixels = np.asarray(panel.canvas.buffer_rgba())
        assert pixels.shape[0] > 0 and pixels.shape[1] > 0
        assert np.unique(pixels.reshape(-1, 4), axis=0).shape[0] > 10

        # Panel menus, colorbars, and layout growth go through the Qt6 enums.
        panel.colorbar_action.setChecked(True)
        assert panel.colorbar is not None
        window.toggle_theme()
        window.add_column()
        assert window.cols == 2 and sorted(window.placeholders) == [1]
        _process_events(qapp)
        assert not window.grab().isNull()
    finally:
        window.close()
        _process_events(qapp)
    assert window.panels == {}


@requires_qt6
def test_viewer_waits_for_a_stream_that_does_not_exist_yet(qapp):
    """Launched before the system is built, the viewer waits instead of crashing."""

    import uuid

    import pyshmem
    from pyrtc.scripts import viewer_core

    name = f"qtwait_{uuid.uuid4().hex[:8]}"
    window = viewer_core.MosaicViewerWindow([name], fps=30, geometry="square", pixel_scale=4.0)
    stream = None
    try:
        window.show()
        _process_events(qapp)
        assert window.panels == {}
        assert isinstance(window.placeholders[0], viewer_core.UnavailableStreamPlaceholder)

        assert window.placeholders[0].waiting
        stream = pyshmem.create(name, shape=(4, 6), dtype=np.float32)
        stream.write(np.ones((4, 6), dtype=np.float32))
        window._next_reconnect = 0.0
        window.refresh_panels()
        assert sorted(window.panels) == [0] and window.placeholders == {}

        # A stream that exists but fails to open is not retried automatically.
        rebuilds = []
        original = viewer_core.StreamConnection

        def broken(name, *args, **kwargs):
            raise ValueError("unsupported stream")

        viewer_core.StreamConnection = broken
        try:
            window.rebuild_grid()
            assert not window.placeholders[0].waiting
            window.rebuild_grid = lambda: rebuilds.append(1)
            window._next_reconnect = 0.0
            window.refresh_panels()
            assert rebuilds == []
        finally:
            viewer_core.StreamConnection = original
    finally:
        window.close()
        _process_events(qapp)
        if stream is not None:
            stream.close()
            pyshmem.unlink(name)


@requires_qt6
def test_manager_gui_toolbar_follows_the_system_state(qapp, tmp_path, monkeypatch):
    from pyrtc.gui import main_window

    monkeypatch.chdir(tmp_path)
    window = main_window.ManagerMainWindow(
        config_path=str(SYNTHETIC_CONFIG_PATH), theme_name="dark", refresh_ms=60_000
    )
    try:
        actions = ("build", "start", "stop", "shutdown")

        def enabled(state):
            window._update_action_states({"state": state, "components": {}})
            return {name for name in actions if getattr(window, f"{name}_action").isEnabled()}

        assert enabled("created") == {"build"}
        assert enabled("built") == {"start", "shutdown"}
        assert enabled("running") == {"stop", "shutdown"}
        # Stopped components still hold their hardware: start again or shut down.
        assert enabled("stopped") == {"start", "shutdown"}
        # A failed build or crash can be shut down and rebuilt without a GUI restart.
        assert enabled("failed") == {"stop", "shutdown"}
    finally:
        window.close()
        _process_events(qapp)


@requires_qt6
def test_manager_gui_edges_enter_inputs_and_feedback_runs_below(qapp, tmp_path, monkeypatch):
    from pyrtc.gui import main_window

    monkeypatch.chdir(tmp_path)
    window = main_window.ManagerMainWindow(
        config_path=str(SYNTHETIC_CONFIG_PATH), theme_name="dark", refresh_ms=60_000
    )
    try:
        window.show()
        _process_events(qapp)
        canvas = window.graph_canvas
        assert canvas._auto_fit  # fitted on load
        # A refresh after the fit leaves the user's view alone.
        transform = canvas.transform()
        from qtpy.QtWidgets import QAbstractSlider

        canvas.horizontalScrollBar().triggerAction(QAbstractSlider.SliderAction.SliderSingleStepAdd)
        assert not canvas._auto_fit  # the user's scroll wins
        window.refresh_view()
        assert canvas.transform() == transform
        canvas.fit_to_graph()
        assert canvas._auto_fit
        canvas.zoom_in()
        assert not canvas._auto_fit  # the user's zoom wins

        edge = next(
            item
            for item in canvas._edge_items
            if (item.edge.source_section, item.edge.target_section) == ("loop", "wfc")
        )
        loop_item, wfc_item = edge.source_item, edge.target_item
        wfc_item.setPos(loop_item.pos().x() - 400.0, loop_item.pos().y())
        points = edge.route()
        # Out of the source's right side, into the target's left side.
        assert points[0] == loop_item.connection_anchor("right")
        tip = wfc_item.connection_anchor("left")
        assert edge.arrow_item.path().elementAt(0).x == tip.x()
        # Feedback runs below every node it passes.
        lane_y = max(point.y() for point in points)
        assert all(lane_y > rect.bottom() for rect in canvas._node_rects())
    finally:
        window.close()
        _process_events(qapp)


@requires_qt6
def test_manager_gui_builds_graph_offscreen(qapp, tmp_path, monkeypatch):
    from pyrtc.gui import main_window
    from qtpy.QtCore import QPoint, Qt
    from qtpy.QtTest import QTest

    monkeypatch.chdir(tmp_path)
    window = main_window.ManagerMainWindow(
        config_path=str(SYNTHETIC_CONFIG_PATH), theme_name="light", refresh_ms=60_000
    )
    try:
        window.show()
        _process_events(qapp)
        canvas = window.graph_canvas
        assert {"wfs", "slopes", "loop", "wfc"}.issubset(canvas._items_by_section)

        # Moving and selecting a node drive itemChange (GraphicsItemChange enums).
        loop_item = canvas._items_by_section["loop"]
        loop_item.setPos(loop_item.pos().x() + 40.0, loop_item.pos().y() + 10.0)
        position = window.adapter.build_graph_snapshot().metadata["positions"]["loop"]
        assert position == {"x": loop_item.pos().x(), "y": loop_item.pos().y()}
        loop_item.setSelected(True)
        _process_events(qapp)
        assert window.selected_section == "loop"

        # A click on empty canvas clears the selection (QMouseEvent.position()).
        canvas.centerOn(-10_000.0, -10_000.0)
        canvas.setSceneRect(-20_000.0, -20_000.0, 40_000.0, 40_000.0)
        canvas.centerOn(-10_000.0, -10_000.0)
        QTest.mouseClick(canvas.viewport(), Qt.MouseButton.LeftButton, pos=QPoint(5, 5))
        _process_events(qapp)
        assert window.selected_section is None

        canvas.zoom_in()
        canvas.reset_zoom()
        window._set_theme("dark")
        _process_events(qapp)
        assert not window.grab().isNull()

        dialog = main_window.ComponentSettingsDialog(
            section_name="loop2",
            descriptor=next(iter(main_window.list_component_descriptors())),
            class_name="pyrtc.loop.Loop",
            parent=window,
        )
        assert dialog.values()["section_name"] == "loop2"
        dialog.deleteLater()
    finally:
        window.close()
        _process_events(qapp)


@requires_qt6
def test_launch_manager_gui_runs_event_loop(qapp, tmp_path, monkeypatch):
    from pyrtc.gui import main_window
    from qtpy.QtCore import QTimer

    monkeypatch.chdir(tmp_path)

    def _quit():
        for widget in qapp.topLevelWidgets():
            widget.close()
        qapp.quit()

    QTimer.singleShot(50, _quit)
    assert main_window.launch_manager_gui(refresh_ms=60_000) == 0


@requires_qt6
def test_launch_mosaic_viewer_runs_event_loop(qapp):
    from pyrtc.scripts import viewer_core
    from qtpy.QtCore import QTimer

    stream = private_stream("qtlaunch", (4, 4), np.float32)
    stream.write(np.ones((4, 4), dtype=np.float32))

    def _quit():
        for widget in qapp.topLevelWidgets():
            widget.close()
        qapp.quit()

    QTimer.singleShot(50, _quit)
    result = viewer_core.launch_mosaic_viewer(
        ["pyrtc-view"], [stream.name], 30, "square", 12.0, None, None, "dark"
    )
    assert result == 0


@requires_qt6
def test_graph_node_shows_safety_alerts(qapp):
    from pyrtc.gui.main_window import GraphNodeItem
    from pyrtc.gui.models import GraphNodeModel
    from pyrtc.gui.theme import get_theme

    theme = get_theme("dark")
    alerts = ("signal stale for 2.0 s (producer exited); action=open", "second alert")
    node = GraphNodeModel("loop", "Loop", "loop", 0.0, 0.0, state="running", alerts=alerts)
    item = GraphNodeItem(node, theme, lambda *_: None, lambda *_: None)

    text = item.state_item.text()
    assert text.startswith("RUNNING  ⚠ signal stale")
    assert text.endswith("(+1)") and len(text) < 60
    assert item.toolTip() == "\n".join(alerts)
    assert item.state_item.brush().color().name() == theme.degraded.lower()

    calm = GraphNodeItem(
        GraphNodeModel("wfs", "WFS", "wfs", 0.0, 0.0, state="running"),
        theme,
        lambda *_: None,
        lambda *_: None,
    )
    assert calm.state_item.text() == "RUNNING"
    assert calm.toolTip() == ""
