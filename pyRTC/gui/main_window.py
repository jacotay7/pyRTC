"""Qt main window for the pyRTC manager GUI."""

from __future__ import annotations

from collections import deque
import logging
from pathlib import Path
from types import SimpleNamespace

try:
    from PyQt5.QtCore import QPointF, QRectF, Qt, QTimer
    from PyQt5.QtGui import QBrush, QColor, QFont, QPainter, QPainterPath, QPen
    from PyQt5.QtWidgets import (
        QAction,
        QApplication,
        QComboBox,
        QDockWidget,
        QFileDialog,
        QFormLayout,
        QFrame,
        QGraphicsPathItem,
        QGraphicsRectItem,
        QGraphicsScene,
        QGraphicsSimpleTextItem,
        QGraphicsView,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QPlainTextEdit,
        QScrollArea,
        QSizePolicy,
        QSplitter,
        QStatusBar,
        QToolBar,
        QVBoxLayout,
        QWidget,
    )
    _GUI_IMPORT_ERROR = None
except ImportError as exc:
    _GUI_IMPORT_ERROR = exc

    class _QtUnavailableBase:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "pyrtc-manager-gui requires GUI dependencies. Install with: pip install pyRTC[gui]"
            ) from _GUI_IMPORT_ERROR

    QAction = QApplication = QComboBox = QDockWidget = QFileDialog = QFormLayout = QFrame = QGraphicsPathItem = (  # type: ignore[assignment]
        QGraphicsRectItem
    ) = QGraphicsScene = QGraphicsSimpleTextItem = QGraphicsView = QHBoxLayout = QLabel = QLineEdit = QListWidget = QListWidgetItem = QMainWindow = QMessageBox = QPushButton = QPlainTextEdit = QScrollArea = QSizePolicy = QSplitter = QStatusBar = QToolBar = QVBoxLayout = QWidget = QTimer = _QtUnavailableBase
    QPointF = QRectF = QBrush = QColor = QFont = QPainter = QPainterPath = QPen = _QtUnavailableBase
    Qt = SimpleNamespace(Horizontal=0, Vertical=0, AlignTop=0, LeftDockWidgetArea=0, BottomDockWidgetArea=0)

from pyRTC.logging_utils import get_logger

from .manager_adapter import ManagerAdapter
from .models import GraphEdgeModel, GraphNodeModel, GraphSnapshot
from .theme import build_main_window_stylesheet, get_theme


logger = get_logger(__name__)


class _GuiLogHandler(logging.Handler):
    def __init__(self, max_lines: int = 500) -> None:
        super().__init__()
        self._messages = deque(maxlen=max_lines)
        self.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(processName)s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._messages.append(self.format(record))
        except Exception:
            pass

    def render(self) -> str:
        return "\n".join(self._messages)


def _require_gui_backend() -> None:
    if _GUI_IMPORT_ERROR is not None:
        raise ImportError(
            "pyrtc-manager-gui requires GUI dependencies. Install with: pip install pyRTC[gui]"
        ) from _GUI_IMPORT_ERROR


class GraphEdgeItem:
    def __init__(self, scene, source_item, target_item, edge: GraphEdgeModel, theme):
        self.scene = scene
        self.source_item = source_item
        self.target_item = target_item
        self.edge = edge
        self.path_item = QGraphicsPathItem()
        self.label_item = QGraphicsSimpleTextItem(edge.label)
        self.path_item.setZValue(0)
        self.label_item.setZValue(0)
        self.scene.addItem(self.path_item)
        self.scene.addItem(self.label_item)
        self.apply_theme(theme)
        self.update_positions()

    def apply_theme(self, theme) -> None:
        self.path_item.setPen(QPen(QColor(theme.edge), 2.0))
        self.label_item.setBrush(QBrush(QColor(theme.subtext)))

    def update_positions(self) -> None:
        source_point = self.source_item.connection_anchor("right")
        target_point = self.target_item.connection_anchor("left")
        midpoint_x = source_point.x() + max((target_point.x() - source_point.x()) / 2.0, 40.0)
        path = QPainterPath(source_point)
        path.lineTo(midpoint_x, source_point.y())
        path.lineTo(midpoint_x, target_point.y())
        path.lineTo(target_point)
        self.path_item.setPath(path)
        midpoint = QPointF(midpoint_x, (source_point.y() + target_point.y()) / 2.0)
        self.label_item.setPos(midpoint.x() - 24.0, midpoint.y() - 18.0)


class GraphNodeButtonItem(QGraphicsRectItem):
    def __init__(self, label: str, x: float, y: float, width: float, callback, *, enabled: bool = True, parent=None):
        super().__init__(QRectF(x, y, width, 22.0), parent)
        self._callback = callback
        self._enabled = enabled
        self.label_item = QGraphicsSimpleTextItem(label, self)
        self.label_item.setPos(x + 10.0, y + 3.0)
        self.setAcceptedMouseButtons(Qt.LeftButton)
        self.setZValue(2)

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled

    def apply_theme(self, theme) -> None:
        if self._enabled:
            fill = QColor(theme.button_bg)
            text = QColor(theme.button_fg)
        else:
            fill = QColor(theme.panel_alt_bg)
            text = QColor(theme.subtext)
        self.setBrush(QBrush(fill))
        self.setPen(QPen(QColor(theme.border), 1.0))
        self.label_item.setBrush(QBrush(text))

    def mousePressEvent(self, event):
        if self._enabled and self._callback is not None:
            self._callback()
            event.accept()
            return
        super().mousePressEvent(event)


class GraphNodeItem(QGraphicsRectItem):
    def __init__(self, node: GraphNodeModel, theme, selection_callback, action_callback):
        super().__init__(QRectF(0.0, 0.0, 220.0, 152.0))
        self.node = node
        self.theme = theme
        self._selection_callback = selection_callback
        self._action_callback = action_callback
        self._edge_items = []
        self._blink_on = False
        self._drag_state_callback = None
        self._selection_guard = None
        self.setFlags(
            QGraphicsRectItem.ItemIsMovable
            | QGraphicsRectItem.ItemIsSelectable
            | QGraphicsRectItem.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)

        self.title_item = QGraphicsSimpleTextItem(node.title, self)
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(11)
        self.title_item.setFont(title_font)
        self.title_item.setPos(14.0, 10.0)

        self.subtitle_item = QGraphicsSimpleTextItem(node.subtitle, self)
        self.subtitle_item.setPos(14.0, 34.0)

        self.state_item = QGraphicsSimpleTextItem(node.state.upper(), self)
        self.state_item.setPos(14.0, 60.0)

        self.streams_item = QGraphicsSimpleTextItem(self._stream_text(), self)
        self.streams_item.setPos(14.0, 84.0)

        self.start_button = GraphNodeButtonItem(
            "Start",
            14.0,
            118.0,
            76.0,
            lambda: self._action_callback("start", self.node.section_name),
            enabled=self.node.can_start,
            parent=self,
        )
        self.stop_button = GraphNodeButtonItem(
            "Stop",
            102.0,
            118.0,
            76.0,
            lambda: self._action_callback("stop", self.node.section_name),
            enabled=self.node.can_stop,
            parent=self,
        )

        self.setPos(node.x, node.y)
        self.apply_theme(theme)

    def _stream_text(self) -> str:
        inputs = ", ".join(self.node.input_streams) or "-"
        outputs = ", ".join(self.node.output_streams) or "-"
        return f"in: {inputs}\nout: {outputs}"

    def apply_theme(self, theme) -> None:
        self.theme = theme
        fill = self._fill_color(theme)
        self.setBrush(QBrush(fill))
        border_color = theme.accent if self.isSelected() else theme.border
        self.setPen(QPen(QColor(border_color), 2.0))
        self.title_item.setBrush(QBrush(QColor(theme.text)))
        self.subtitle_item.setBrush(QBrush(QColor(theme.subtext)))
        self.state_item.setBrush(QBrush(QColor(self._state_color(theme))))
        self.streams_item.setBrush(QBrush(QColor(theme.subtext)))
        self.start_button.set_enabled(self.node.can_start)
        self.stop_button.set_enabled(self.node.can_stop)
        self.start_button.apply_theme(theme)
        self.stop_button.apply_theme(theme)

    def _state_color(self, theme):
        mapping = {
            "running": theme.running,
            "degraded": theme.degraded,
            "failed": theme.failed,
            "stopped": theme.stopped,
            "created": theme.stopped,
            "validated": theme.accent,
            "starting": theme.accent,
            "stopping": theme.degraded,
        }
        return mapping.get(self.node.state, theme.subtext)

    def _fill_color(self, theme):
        if self.node.state == "failed" and self._blink_on:
            return QColor(theme.failed).lighter(140)
        return QColor(theme.panel_bg)

    def set_blink_state(self, blink_on: bool) -> None:
        self._blink_on = blink_on
        self.apply_theme(self.theme)

    def add_edge(self, edge_item: GraphEdgeItem) -> None:
        self._edge_items.append(edge_item)

    def set_drag_state_callback(self, callback) -> None:
        self._drag_state_callback = callback

    def set_selection_guard(self, callback) -> None:
        self._selection_guard = callback

    def connection_anchor(self, side: str):
        rect = self.sceneBoundingRect()
        if side == "right":
            return QPointF(rect.right(), rect.center().y())
        return QPointF(rect.left(), rect.center().y())

    def itemChange(self, change, value):
        result = super().itemChange(change, value)
        if change == QGraphicsRectItem.ItemPositionHasChanged:
            for edge_item in self._edge_items:
                edge_item.update_positions()
        if change == QGraphicsRectItem.ItemSelectedHasChanged:
            self.apply_theme(self.theme)
            if bool(value) and (self._selection_guard is None or self._selection_guard()):
                self._selection_callback(self.node.section_name)
        return result

    def mousePressEvent(self, event):
        if self._drag_state_callback is not None:
            self._drag_state_callback(True)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        if self._drag_state_callback is not None:
            self._drag_state_callback(False)


class GraphCanvas(QGraphicsView):
    def __init__(self, selection_callback, deselection_callback, action_callback):
        super().__init__()
        self.selection_callback = selection_callback
        self.deselection_callback = deselection_callback
        self.action_callback = action_callback
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setFrameShape(QFrame.NoFrame)
        self._items_by_section = {}
        self._edge_items = []
        self._dragging = False
        self._suppress_selection_callback = False
        self._zoom = 1.0

    @property
    def is_dragging(self) -> bool:
        return self._dragging

    def set_dragging(self, dragging: bool) -> None:
        self._dragging = dragging

    def rebuild(self, snapshot: GraphSnapshot, theme, *, blink_on: bool) -> None:
        positions = {
            section_name: (item.pos().x(), item.pos().y())
            for section_name, item in self._items_by_section.items()
        }
        self.scene.clear()
        self._items_by_section = {}
        self._edge_items = []
        self.setBackgroundBrush(QBrush(QColor(theme.canvas_bg)))

        for node in snapshot.nodes:
            item = GraphNodeItem(node, theme, self.selection_callback, self.action_callback)
            item.set_drag_state_callback(self.set_dragging)
            item.set_selection_guard(self.should_emit_selection_callback)
            if node.section_name in positions:
                x, y = positions[node.section_name]
                item.setPos(x, y)
            self.scene.addItem(item)
            item.set_blink_state(blink_on)
            self._items_by_section[node.section_name] = item

        for edge in snapshot.edges:
            source_item = self._items_by_section.get(edge.source_section)
            target_item = self._items_by_section.get(edge.target_section)
            if source_item is None or target_item is None:
                continue
            edge_item = GraphEdgeItem(self.scene, source_item, target_item, edge, theme)
            source_item.add_edge(edge_item)
            target_item.add_edge(edge_item)
            self._edge_items.append(edge_item)

        self.scene.setSceneRect(self.scene.itemsBoundingRect().adjusted(-80.0, -80.0, 120.0, 120.0))

    def select_section(self, section_name: str) -> None:
        item = self._items_by_section.get(section_name)
        if item is None:
            return
        self._suppress_selection_callback = True
        try:
            self.scene.clearSelection()
            item.setSelected(True)
        finally:
            self._suppress_selection_callback = False

    def should_emit_selection_callback(self) -> bool:
        return not self._suppress_selection_callback

    def mousePressEvent(self, event):
        if self.itemAt(event.pos()) is None:
            self.scene.clearSelection()
            self.deselection_callback()
        super().mousePressEvent(event)

    def zoom_in(self) -> None:
        self._apply_zoom(1.15)

    def zoom_out(self) -> None:
        self._apply_zoom(1.0 / 1.15)

    def reset_zoom(self) -> None:
        if self._zoom == 0:
            return
        self.scale(1.0 / self._zoom, 1.0 / self._zoom)
        self._zoom = 1.0

    def _apply_zoom(self, factor: float) -> None:
        new_zoom = self._zoom * factor
        if new_zoom < 0.35 or new_zoom > 4.0:
            return
        self.scale(factor, factor)
        self._zoom = new_zoom

    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            self.zoom_in()
        else:
            self.zoom_out()
        event.accept()


class ManagerMainWindow(QMainWindow):
    def __init__(self, *, config_path: str | None = None, mode: str | None = None, theme_name: str = "dark", refresh_ms: int = 1000):
        _require_gui_backend()
        super().__init__()
        self.adapter = ManagerAdapter()
        self.theme_name = theme_name
        self.refresh_ms = refresh_ms
        self.selected_section: str | None = None
        self._blink_on = False
        self._field_inputs = {}
        self._field_types = {}
        self._function_buttons = []
        self._inspector_section: str | None = None
        self._gui_log_handler = _GuiLogHandler()
        get_logger().addHandler(self._gui_log_handler)

        self.setWindowTitle("pyRTC Manager GUI")
        self.resize(1500, 920)
        self._build_ui()
        self._build_toolbar(mode)
        self._apply_theme()

        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self.refresh_view)
        self.refresh_timer.start(self.refresh_ms)

        self.blink_timer = QTimer(self)
        self.blink_timer.timeout.connect(self._toggle_blink)
        self.blink_timer.start(700)

        if config_path:
            self.load_config(config_path, mode=mode)

    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(splitter)

        self.catalog_panel = QFrame()
        self.catalog_panel.setObjectName("Panel")
        catalog_layout = QVBoxLayout(self.catalog_panel)
        catalog_layout.setContentsMargins(12, 12, 12, 12)
        catalog_layout.setSpacing(10)
        catalog_title = QLabel("System Components")
        catalog_layout.addWidget(catalog_title)
        self.component_list = QListWidget()
        self.component_list.currentTextChanged.connect(self._handle_component_selected)
        catalog_layout.addWidget(self.component_list)

        center_panel = QFrame()
        center_panel.setObjectName("Panel")
        center_layout = QVBoxLayout(center_panel)
        center_layout.setContentsMargins(12, 12, 12, 12)
        center_layout.setSpacing(10)
        self.summary_label = QLabel("No config loaded")
        self.summary_label.setObjectName("SubtleText")
        center_layout.addWidget(self.summary_label)
        self.graph_canvas = GraphCanvas(self._select_section_from_canvas, self._clear_selection, self._handle_node_action)
        center_layout.addWidget(self.graph_canvas)

        self.inspector_panel = QFrame()
        self.inspector_panel.setObjectName("Panel")
        inspector_layout = QVBoxLayout(self.inspector_panel)
        inspector_layout.setContentsMargins(12, 12, 12, 12)
        inspector_layout.setSpacing(10)
        self.inspector_title = QLabel("Inspector")
        inspector_layout.addWidget(self.inspector_title)
        self.inspector_status = QLabel("Select a component")
        self.inspector_status.setObjectName("SubtleText")
        inspector_layout.addWidget(self.inspector_status)
        self.form_scroll = QScrollArea()
        self.form_scroll.setWidgetResizable(True)
        self.form_container = QWidget()
        self.form_layout = QFormLayout(self.form_container)
        self.form_scroll.setWidget(self.form_container)
        inspector_layout.addWidget(self.form_scroll)
        self.functions_title = QLabel("Functions")
        self.functions_title.hide()
        inspector_layout.addWidget(self.functions_title)
        self.functions_container = QWidget()
        self.functions_layout = QVBoxLayout(self.functions_container)
        self.functions_layout.setContentsMargins(0, 0, 0, 0)
        self.functions_layout.setSpacing(8)
        inspector_layout.addWidget(self.functions_container)
        button_row = QHBoxLayout()
        self.refresh_component_button = QPushButton("Refresh")
        self.refresh_component_button.clicked.connect(self._refresh_selected_component)
        button_row.addWidget(self.refresh_component_button)
        self.apply_button = QPushButton("Apply")
        self.apply_button.clicked.connect(self._apply_selected_component)
        button_row.addWidget(self.apply_button)
        inspector_layout.addLayout(button_row)

        splitter.addWidget(self.catalog_panel)
        splitter.addWidget(center_panel)
        splitter.addWidget(self.inspector_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)

        self.log_dock = QDockWidget("Logs", self)
        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)
        self.log_dock.setWidget(self.log_output)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.log_dock)

        self.setStatusBar(QStatusBar())

    def _build_toolbar(self, mode: str | None) -> None:
        toolbar = QToolBar("Manager")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        self.load_action = QAction("Load", self)
        self.load_action.triggered.connect(self.open_config_dialog)
        toolbar.addAction(self.load_action)

        self.validate_action = QAction("Validate", self)
        self.validate_action.triggered.connect(self.validate_config)
        toolbar.addAction(self.validate_action)

        self.start_action = QAction("Start", self)
        self.start_action.triggered.connect(self.start_system)
        toolbar.addAction(self.start_action)

        self.stop_action = QAction("Stop", self)
        self.stop_action.triggered.connect(self.stop_system)
        toolbar.addAction(self.stop_action)

        self.reset_action = QAction("Reset", self)
        self.reset_action.triggered.connect(self.reset_system)
        toolbar.addAction(self.reset_action)

        self.refresh_action = QAction("Refresh", self)
        self.refresh_action.triggered.connect(self.refresh_view)
        toolbar.addAction(self.refresh_action)

        self.restart_action = QAction("Restart Selected", self)
        self.restart_action.triggered.connect(self.restart_selected_component)
        toolbar.addAction(self.restart_action)

        self.viewer_action = QAction("Launch Viewer", self)
        self.viewer_action.triggered.connect(self.launch_viewer)
        toolbar.addAction(self.viewer_action)

        self.zoom_in_action = QAction("Zoom In", self)
        self.zoom_in_action.triggered.connect(self.graph_canvas.zoom_in)
        toolbar.addAction(self.zoom_in_action)

        self.zoom_out_action = QAction("Zoom Out", self)
        self.zoom_out_action.triggered.connect(self.graph_canvas.zoom_out)
        toolbar.addAction(self.zoom_out_action)

        self.zoom_reset_action = QAction("Reset Zoom", self)
        self.zoom_reset_action.triggered.connect(self.graph_canvas.reset_zoom)
        toolbar.addAction(self.zoom_reset_action)

        toolbar.addSeparator()

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["soft-rtc", "hard-rtc"])
        if mode in {"soft-rtc", "hard-rtc"}:
            self.mode_combo.setCurrentText(mode)
        self.mode_combo.currentTextChanged.connect(self._handle_mode_changed)
        toolbar.addWidget(self.mode_combo)

        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["dark", "light"])
        self.theme_combo.setCurrentText(self.theme_name)
        self.theme_combo.currentTextChanged.connect(self._set_theme)
        toolbar.addWidget(self.theme_combo)
        self._update_action_states(None)

    def _apply_theme(self) -> None:
        theme = get_theme(self.theme_name)
        self.setStyleSheet(build_main_window_stylesheet(theme))

    def _set_theme(self, theme_name: str) -> None:
        self.theme_name = theme_name
        self._apply_theme()
        self.refresh_view()

    def _handle_mode_changed(self, mode: str) -> None:
        if not self.adapter.is_loaded():
            return
        try:
            self.adapter.set_mode(mode)
        except Exception as exc:
            actual_mode = self.adapter.selected_mode()
            self.mode_combo.blockSignals(True)
            self.mode_combo.setCurrentText(actual_mode)
            self.mode_combo.blockSignals(False)
            self.statusBar().showMessage(str(exc), 4000)
            return
        self.statusBar().showMessage(f"Manager mode set to {mode}", 3000)
        self.refresh_view()

    def _toggle_blink(self) -> None:
        self._blink_on = not self._blink_on
        self.refresh_view()

    def open_config_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load pyRTC Config", "", "YAML Files (*.yaml *.yml)")
        if path:
            self.load_config(path, mode=self.mode_combo.currentText())

    def load_config(self, config_path: str, *, mode: str | None = None) -> None:
        try:
            self.adapter.load_config(config_path, mode=mode)
        except Exception as exc:
            self._show_error("Load failed", exc)
            return
        self.mode_combo.blockSignals(True)
        self.mode_combo.setCurrentText(self.adapter.selected_mode())
        self.mode_combo.blockSignals(False)
        self.statusBar().showMessage(f"Loaded {config_path}", 3000)
        self.refresh_view()

    def validate_config(self) -> None:
        try:
            self.adapter.validate()
        except Exception as exc:
            self._show_error("Validation failed", exc)
            return
        self.statusBar().showMessage("Configuration is valid", 3000)
        self.refresh_view()

    def start_system(self) -> None:
        try:
            self.adapter.start()
        except Exception as exc:
            self._show_error("Start failed", exc)
            return
        self.statusBar().showMessage("System started", 3000)
        self.refresh_view()

    def stop_system(self) -> None:
        try:
            self.adapter.stop()
        except Exception as exc:
            self._show_error("Stop failed", exc)
            return
        self.statusBar().showMessage("System stopped", 3000)
        self.refresh_view()

    def reset_system(self) -> None:
        try:
            self.adapter.reset()
        except Exception as exc:
            self._show_error("Reset failed", exc)
            return
        self.statusBar().showMessage("System reset", 3000)
        self.refresh_view()

    def restart_selected_component(self) -> None:
        if not self.selected_section:
            self.statusBar().showMessage("Select a component first", 2500)
            return
        try:
            self.adapter.restart_component(self.selected_section)
        except Exception as exc:
            self._show_error("Component restart failed", exc)
            return
        self.statusBar().showMessage(f"Restarted {self.selected_section}", 3000)
        self.refresh_view()

    def _handle_node_action(self, action: str, section_name: str) -> None:
        self.selected_section = section_name
        if action == "start":
            try:
                self.adapter.start_component(section_name)
            except Exception as exc:
                self._show_error("Component start failed", exc)
                return
            self.statusBar().showMessage(f"Started {section_name}", 3000)
        elif action == "stop":
            try:
                self.adapter.stop_component(section_name)
            except Exception as exc:
                self._show_error("Component stop failed", exc)
                return
            self.statusBar().showMessage(f"Stopped {section_name}", 3000)
        self.refresh_view()

    def launch_viewer(self) -> None:
        try:
            self.adapter.launch_viewer()
        except Exception as exc:
            self._show_error("Viewer launch failed", exc)
            return
        self.statusBar().showMessage("Viewer launched", 3000)

    def refresh_view(self) -> None:
        if not self.adapter.is_loaded():
            return
        try:
            status = self.adapter.status()
        except Exception:
            try:
                status = self.adapter.refresh()
            except Exception as exc:
                self._show_error("Status refresh failed", exc)
                return
        self._update_action_states(status)
        if self.graph_canvas.is_dragging:
            self.summary_label.setText(
                f"state={status.get('state', '-')}  mode={status.get('mode', '-')}  config={self.adapter.config_path or '-'}"
            )
            self._refresh_logs()
            return
        snapshot = self.adapter.build_graph_snapshot(status)
        self.summary_label.setText(
            f"state={snapshot.state}  mode={snapshot.mode}  config={snapshot.config_path or '-'}"
        )
        self._populate_component_list(snapshot)
        self.graph_canvas.rebuild(snapshot, get_theme(self.theme_name), blink_on=self._blink_on)
        if self.selected_section:
            self.graph_canvas.select_section(self.selected_section)
        self._refresh_logs()

    def _update_action_states(self, status: dict | None) -> None:
        is_loaded = self.adapter.is_loaded()
        state = None if status is None else status.get("state")
        is_running = state in {"running", "degraded", "failed"}
        runtime_sections = set() if status is None else set(status.get("components", {}))

        self.validate_action.setEnabled(is_loaded and not is_running)
        self.start_action.setEnabled(is_loaded and not is_running)
        self.stop_action.setEnabled(is_running)
        self.reset_action.setEnabled(is_running)
        self.refresh_action.setEnabled(is_running)
        self.restart_action.setEnabled(is_running and bool(self.selected_section) and self.selected_section in runtime_sections)
        self.viewer_action.setEnabled(is_loaded and is_running)

    def _populate_component_list(self, snapshot: GraphSnapshot) -> None:
        selected = self.selected_section
        self.component_list.blockSignals(True)
        self.component_list.clear()
        for node in snapshot.nodes:
            item = QListWidgetItem(f"{node.section_name} [{node.state}]")
            item.setData(Qt.UserRole, node.section_name)
            self.component_list.addItem(item)
            if node.section_name == selected:
                self.component_list.setCurrentItem(item)
        self.component_list.blockSignals(False)

    def _handle_component_selected(self, _text: str) -> None:
        item = self.component_list.currentItem()
        if item is None:
            return
        section_name = item.data(Qt.UserRole)
        self.selected_section = section_name
        self.graph_canvas.select_section(section_name)
        self._populate_inspector(section_name)

    def _select_section_from_canvas(self, section_name: str) -> None:
        self.selected_section = section_name
        for index in range(self.component_list.count()):
            item = self.component_list.item(index)
            if item.data(Qt.UserRole) == section_name:
                self.component_list.setCurrentItem(item)
                break
        self._populate_inspector(section_name)

    def _clear_selection(self) -> None:
        self.selected_section = None
        self._inspector_section = None
        self.component_list.blockSignals(True)
        self.component_list.clearSelection()
        self.component_list.setCurrentItem(None)
        self.component_list.blockSignals(False)
        self.inspector_title.setText("Inspector")
        self.inspector_status.setText("Select a component")
        self._clear_form()
        self._clear_function_buttons()
        self.functions_title.hide()
        self._update_action_states(self.adapter._last_status)

    def _clear_form(self) -> None:
        while self.form_layout.rowCount():
            self.form_layout.removeRow(0)
        self._field_inputs = {}
        self._field_types = {}

    def _clear_function_buttons(self) -> None:
        while self.functions_layout.count():
            item = self.functions_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._function_buttons = []

    def _populate_inspector(self, section_name: str) -> None:
        self._clear_form()
        self._clear_function_buttons()
        self._inspector_section = section_name
        self.inspector_title.setText(f"Inspector: {section_name}")
        try:
            rows = self.adapter.get_component_parameters(section_name)
            functions = self.adapter.get_component_functions(section_name)
            cached_status = self.adapter._last_status or {}
            status = cached_status.get("components", {}).get(section_name, {})
        except Exception as exc:
            self.inspector_status.setText(str(exc))
            return
        self.inspector_status.setText(
            f"state={status.get('state', '-')}  mode={status.get('mode', '-')}  restart={status.get('restart_policy', '-')}"
        )
        for row in rows:
            editor = QLineEdit("" if row["value"] is None else str(row["value"]))
            editor.setToolTip(row["description"])
            self.form_layout.addRow(f"{row['name']} ({row['type']})", editor)
            self._field_inputs[row["name"]] = editor
            self._field_types[row["name"]] = row["type"]

        if functions:
            self.functions_title.show()
            for function in functions:
                button = QPushButton(function["name"])
                button.setToolTip(function["description"])
                button.setEnabled(bool(function["enabled"]))
                button.clicked.connect(
                    lambda _checked=False, name=function["name"], section=section_name: self._run_component_function(section, name)
                )
                self.functions_layout.addWidget(button)
                self._function_buttons.append(button)
        else:
            self.functions_title.hide()

    def _refresh_selected_component(self) -> None:
        if self.selected_section:
            try:
                self.adapter.refresh()
            except Exception as exc:
                self._show_error("Inspector refresh failed", exc)
                return
            self._populate_inspector(self.selected_section)

    def _apply_selected_component(self) -> None:
        if not self.selected_section:
            return
        try:
            for name, editor in self._field_inputs.items():
                self.adapter.set_parameter(self.selected_section, name, editor.text())
        except Exception as exc:
            self._show_error("Parameter update failed", exc)
            return
        self.statusBar().showMessage(f"Updated {self.selected_section}", 3000)
        self._populate_inspector(self.selected_section)
        self.refresh_view()

    def _run_component_function(self, section_name: str, function_name: str) -> None:
        try:
            self.adapter.run_component_function(section_name, function_name)
        except Exception as exc:
            self._show_error(f"Function '{function_name}' failed", exc)
            return
        self.statusBar().showMessage(f"Ran {section_name}.{function_name}()", 3000)
        self.refresh_view()
        self._populate_inspector(section_name)

    def _refresh_logs(self) -> None:
        chunks = []
        in_process_logs = self._gui_log_handler.render()
        if in_process_logs:
            chunks.append(in_process_logs)
        for log_file in self.adapter.log_files():
            path = Path(log_file)
            if not path.exists():
                continue
            try:
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            except Exception:
                continue
            tail = lines[-30:]
            if tail:
                chunks.append(f"== {path.name} ==\n" + "\n".join(tail))
        text = "\n\n".join(chunks) if chunks else "No component log files available yet."
        if self.log_output.toPlainText() != text:
            self.log_output.setPlainText(text)
            scrollbar = self.log_output.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())

    def closeEvent(self, event):
        try:
            get_logger().removeHandler(self._gui_log_handler)
        except Exception:
            pass
        super().closeEvent(event)

    def _show_error(self, title: str, exc: Exception) -> None:
        logger.exception("%s", title)
        QMessageBox.critical(self, title, str(exc))
        self.statusBar().showMessage(f"{title}: {exc}", 5000)


def launch_manager_gui(*, config_path: str | None = None, mode: str | None = None, theme_name: str = "dark", refresh_ms: int = 1000) -> int:
    _require_gui_backend()
    app = QApplication.instance() or QApplication([])
    window = ManagerMainWindow(config_path=config_path, mode=mode, theme_name=theme_name, refresh_ms=refresh_ms)
    window.show()
    return app.exec_()