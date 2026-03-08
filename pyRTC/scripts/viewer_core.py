"""Qt-based shared-memory viewer used by the ``pyrtc-view`` CLI.

The viewer code in this module turns one or more pyRTC shared-memory streams
into a configurable mosaic of live 2D image panels. It is intentionally UI-
centric: the classes here manage layout, theming, refresh cadence, and per-panel
display controls rather than any AO-specific signal processing.
"""

from dataclasses import dataclass
import logging
from types import SimpleNamespace

import numpy as np
from matplotlib.colors import LogNorm, Normalize

try:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    from PyQt5.QtCore import Qt, QTimer
    from PyQt5.QtGui import QKeySequence
    from PyQt5.QtWidgets import (
        QAction,
        QApplication,
        QFrame,
        QGridLayout,
        QHBoxLayout,
        QInputDialog,
        QLabel,
        QMainWindow,
        QMenu,
        QPushButton,
        QScrollArea,
        QSizePolicy,
        QToolButton,
        QVBoxLayout,
        QWidget,
    )
    _VIEWER_BACKEND_IMPORT_ERROR = None
except ImportError as exc:
    _VIEWER_BACKEND_IMPORT_ERROR = exc

    class _QtUnavailableBase:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "pyrtc-view requires viewer dependencies. Install with: pip install pyRTC[viewer]"
            ) from _VIEWER_BACKEND_IMPORT_ERROR

    class _UnavailableQSizePolicy:
        Expanding = 0
        Fixed = 0

    class _UnavailableQKeySequence:
        ZoomIn = "Ctrl++"
        ZoomOut = "Ctrl+-"

    FigureCanvas = _QtUnavailableBase
    Figure = _QtUnavailableBase
    QAction = QApplication = QFrame = QGridLayout = QHBoxLayout = QInputDialog = QLabel = QMainWindow = QMenu = (  # type: ignore[assignment]
        QPushButton
    ) = QScrollArea = QToolButton = QVBoxLayout = QWidget = QTimer = _QtUnavailableBase
    QSizePolicy = _UnavailableQSizePolicy()
    QKeySequence = _UnavailableQKeySequence
    Qt = SimpleNamespace(
        AlignCenter=0,
        AlignVCenter=0,
        AlignHCenter=0,
        AlignLeft=0,
        RichText=0,
        ScrollBarAsNeeded=0,
        PreciseTimer=0,
        WidgetWithChildrenShortcut=0,
    )

from .viewer_helpers import StreamConnection, compute_window_size, normalize_geometry_value, resolve_grid


@dataclass(frozen=True)
class ViewerTheme:
    """Container for the colors used by a viewer theme preset."""

    name: str
    window_bg: str
    panel_bg: str
    panel_border: str
    text: str
    subtext: str
    accent: str
    button_bg: str
    button_fg: str
    axes_bg: str
    figure_bg: str
    stats_bg: str


THEMES = {
    "dark": ViewerTheme(
        name="dark",
        window_bg="#11161d",
        panel_bg="#1a2330",
        panel_border="#2e3b4e",
        text="#edf2f7",
        subtext="#a8b3c4",
        accent="#6ee7c8",
        button_bg="#223044",
        button_fg="#edf2f7",
        axes_bg="#0f1722",
        figure_bg="#0f1722",
        stats_bg="#152133",
    ),
    "light": ViewerTheme(
        name="light",
        window_bg="#eef3f8",
        panel_bg="#ffffff",
        panel_border="#c6d1dc",
        text="#162033",
        subtext="#536179",
        accent="#0f766e",
        button_bg="#dde7f1",
        button_fg="#162033",
        axes_bg="#f7fafc",
        figure_bg="#f7fafc",
        stats_bg="#eef4fa",
    ),
}


def _require_viewer_backend() -> None:
    if _VIEWER_BACKEND_IMPORT_ERROR is not None:
        raise ImportError(
            "pyrtc-view requires viewer dependencies. Install with: pip install pyRTC[viewer]"
        ) from _VIEWER_BACKEND_IMPORT_ERROR


class AddPlotPlaceholder(QFrame):
    """Empty grid cell that lets the user attach another SHM stream."""

    def __init__(self, add_callback):
        _require_viewer_backend()
        super().__init__()
        self.add_callback = add_callback
        self.theme = THEMES["dark"]
        self._build_ui()
        self.apply_theme(self.theme)

    def _build_ui(self):
        self.setFrameShape(QFrame.StyledPanel)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)
        layout.addStretch(1)
        self.add_button = QPushButton("+")
        self.add_button.setMinimumHeight(96)
        self.add_button.clicked.connect(self.add_callback)
        layout.addWidget(self.add_button)
        self.label = QLabel("Add SHM")
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)
        layout.addStretch(1)

    def apply_theme(self, theme: ViewerTheme):
        self.theme = theme
        self.setStyleSheet(
            f"QFrame {{ background: {theme.panel_bg}; border: 1px dashed {theme.panel_border}; border-radius: 12px; }}"
            f"QLabel {{ border: 0; color: {theme.subtext}; background: transparent; font-size: 13px; }}"
            f"QPushButton {{ background: {theme.button_bg}; color: {theme.button_fg}; border: 1px solid {theme.panel_border}; "
            f"border-radius: 12px; font-size: 28px; font-weight: 700; }}"
            f"QPushButton:hover {{ border-color: {theme.accent}; }}"
        )


class EdgeArrowButton(QToolButton):
    """Small edge-mounted button used to grow the mosaic layout."""

    def __init__(self, label: str, callback):
        _require_viewer_backend()
        super().__init__()
        self.setText(label)
        self.setCheckable(True)
        self.clicked.connect(self._handle_click)
        self._callback = callback
        self._reset_timer = QTimer(self)
        self._reset_timer.setSingleShot(True)
        self._reset_timer.timeout.connect(lambda: self.setChecked(False))

    def _handle_click(self):
        self.setChecked(True)
        self._callback()
        self._reset_timer.start(220)


class Stream2DWidget(QFrame):
    """Live panel for a single shared-memory stream.

    Each panel owns one ``StreamConnection`` and is responsible for rendering
    the latest frame, tracking basic stream status such as FPS and paused state,
    and exposing per-panel presentation controls like colorbars and scaling.
    """

    def __init__(
        self,
        connection: StreamConnection,
        remove_callback,
        static_vmin=None,
        static_vmax=None,
        show_colorbar=False,
        show_stats=True,
        show_range=True,
        log_scale=False,
        font_size=14,
    ):
        _require_viewer_backend()
        super().__init__()
        self.connection = connection
        self.remove_callback = remove_callback
        self.static_vmin = static_vmin
        self.static_vmax = static_vmax
        self.show_colorbar = show_colorbar
        self.show_stats = show_stats
        self.show_range = show_range
        self.log_scale = log_scale
        self.font_size = font_size
        self.theme = THEMES["dark"]
        self._last_stats_text = ""
        self._last_status_state = "paused"
        self._disposed = False
        self.colorbar = None
        self.colorbar_axes = None

        frame = self.connection.prime()
        self._frame_ratio = frame.shape[1] / max(frame.shape[0], 1)
        self._aspect = None if 0.1 <= self._frame_ratio <= 10 else "auto"

        self._build_ui(frame)
        self.apply_theme(self.theme)
        self.refresh(force_draw=True)

    def _build_ui(self, frame):
        self.setFrameShape(QFrame.StyledPanel)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(10, 10, 10, 10)
        outer_layout.setSpacing(8)

        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)

        self.title_label = QLabel(self.connection.display_name)
        self.title_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        header_layout.addWidget(self.title_label)

        self.settings_button = QToolButton()
        self.settings_button.setText("Settings")
        self.settings_button.setPopupMode(QToolButton.InstantPopup)
        self.settings_menu = QMenu(self)
        self.settings_button.setMenu(self.settings_menu)
        header_layout.addWidget(self.settings_button)

        outer_layout.addLayout(header_layout)

        self.figure = Figure(figsize=(4.0, 3.4))
        self.axes = self.figure.add_subplot(111)
        self.axes.set_anchor("C")
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        outer_layout.addWidget(self.canvas, stretch=1, alignment=Qt.AlignCenter)

        vmin, vmax = self._resolve_color_limits(frame)
        self.image = self.axes.imshow(
            frame,
            cmap="inferno",
            interpolation="nearest",
            aspect=self._aspect,
            origin="upper",
            vmin=vmin,
            vmax=vmax,
        )
        self.axes.set_xticks([])
        self.axes.set_yticks([])
        self._linear_norm = Normalize(vmin=vmin, vmax=vmax)
        self._sync_colorbar()
        self._update_figure_layout()

        stats_layout = QHBoxLayout()
        stats_layout.setContentsMargins(0, 2, 0, 0)
        stats_layout.setSpacing(8)
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setMinimumWidth(92)
        self.status_label.setVisible(self.show_stats)
        stats_layout.addWidget(self.status_label, alignment=Qt.AlignLeft)

        self.stats_label = QLabel("")
        self.stats_label.setAlignment(Qt.AlignCenter)
        self.stats_label.setTextFormat(Qt.RichText)
        self.stats_label.setMinimumWidth(340)
        self.stats_label.setMaximumWidth(340)
        self.stats_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.stats_label.setVisible(self.show_stats or self.show_range)
        stats_layout.addWidget(self.stats_label, alignment=Qt.AlignCenter)
        outer_layout.addLayout(stats_layout)

        self._build_settings_menu()

    def _build_settings_menu(self):
        self.settings_menu.clear()

        self.colorbar_action = QAction("Show Colorbar", self, checkable=True)
        self.colorbar_action.setChecked(self.show_colorbar)
        self.colorbar_action.toggled.connect(self._toggle_colorbar)
        self.settings_menu.addAction(self.colorbar_action)

        self.stats_action = QAction("Show FPS / Shape Stats", self, checkable=True)
        self.stats_action.setChecked(self.show_stats)
        self.stats_action.toggled.connect(self._toggle_stats)
        self.settings_menu.addAction(self.stats_action)

        self.range_action = QAction("Show Value Range", self, checkable=True)
        self.range_action.setChecked(self.show_range)
        self.range_action.toggled.connect(self._toggle_range)
        self.settings_menu.addAction(self.range_action)

        self.log_action = QAction("Use Log Scale", self, checkable=True)
        self.log_action.setChecked(self.log_scale)
        self.log_action.toggled.connect(self._toggle_log)
        self.settings_menu.addAction(self.log_action)

        self.settings_menu.addSeparator()
        remove_action = QAction("Close Plot", self)
        remove_action.triggered.connect(self.remove_callback)
        self.settings_menu.addAction(remove_action)

    def _resolve_color_limits(self, frame):
        frame_min = float(np.min(frame))
        frame_max = float(np.max(frame))
        if self.log_scale:
            frame_min = max(frame_min, frame_max / 1e3 if frame_max > 0 else 1e-3)
            frame_max = max(frame_max, 1e-2)
        if self.static_vmin is not None:
            frame_min = self.static_vmin
        if self.static_vmax is not None:
            frame_max = self.static_vmax
        if frame_min == frame_max:
            frame_max = frame_min + 1e-6
        return frame_min, frame_max

    def _sync_colorbar(self):
        if self.show_colorbar:
            if self.colorbar is None:
                self.colorbar_axes = self.figure.add_axes([0.84, 0.14, 0.028, 0.72])
                self.colorbar = self.figure.colorbar(self.image, cax=self.colorbar_axes)
                self._style_colorbar()
        elif self.colorbar is not None:
            self.colorbar.remove()
            self.colorbar = None
            self.colorbar_axes = None

    def _update_figure_layout(self):
        self.axes.set_position([0.11, 0.10, 0.66, 0.78])
        if self.colorbar_axes is not None:
            self.colorbar_axes.set_position([0.84, 0.14, 0.028, 0.72])

    def _style_colorbar(self):
        if self.colorbar is None:
            return
        self.colorbar.ax.set_facecolor(self.theme.figure_bg)
        self.colorbar.ax.yaxis.set_ticks_position("left")
        self.colorbar.ax.yaxis.set_label_position("left")
        self.colorbar.ax.yaxis.label.set_color(self.theme.text)
        self.colorbar.ax.tick_params(colors=self.theme.subtext, labelleft=True, labelright=False, pad=2)
        for tick_label in self.colorbar.ax.get_yticklabels():
            tick_label.set_color(self.theme.subtext)
        self.colorbar.outline.set_edgecolor(self.theme.panel_border)

    def _update_stats(self, frame, fps_text):
        paused = fps_text == "PAUSED"
        self._last_status_state = "paused" if paused else "running"
        self.status_label.setText(fps_text)
        self.status_label.setVisible(self.show_stats)
        self._apply_status_style()

        parts = []
        if self.show_range:
            parts.append(f"min={np.min(frame):.3g}")
            parts.append(f"max={np.max(frame):.3g}")
        stats_text = "&nbsp;&nbsp;|&nbsp;&nbsp;".join(parts)
        self.stats_label.setVisible(bool(stats_text) and (self.show_stats or self.show_range))
        if stats_text != self._last_stats_text:
            self.stats_label.setText(stats_text)
            self._last_stats_text = stats_text

    def _toggle_colorbar(self, checked):
        self.show_colorbar = checked
        self._sync_colorbar()
        self._update_figure_layout()
        if getattr(self, "colorbar_action", None) and self.colorbar_action.isChecked() != checked:
            self.colorbar_action.setChecked(checked)
        self.canvas.draw_idle()

    def _toggle_stats(self, checked):
        self.show_stats = checked
        self.status_label.setVisible(self.show_stats)
        self.stats_label.setVisible(bool(self._last_stats_text) and (self.show_stats or self.show_range))
        if getattr(self, "stats_action", None) and self.stats_action.isChecked() != checked:
            self.stats_action.setChecked(checked)
        self.refresh(force_draw=True)

    def _toggle_range(self, checked):
        self.show_range = checked
        self.stats_label.setVisible(bool(self._last_stats_text) and (self.show_stats or self.show_range))
        if getattr(self, "range_action", None) and self.range_action.isChecked() != checked:
            self.range_action.setChecked(checked)
        self.refresh(force_draw=True)

    def _toggle_log(self, checked):
        self.log_scale = checked
        if getattr(self, "log_action", None) and self.log_action.isChecked() != checked:
            self.log_action.setChecked(checked)
        self.refresh(force_draw=True)

    def set_font_size(self, font_size):
        self.font_size = font_size
        self.apply_theme(self.theme)

    def _apply_status_style(self):
        status_bg = self.theme.stats_bg
        status_fg = "#f87171" if self._last_status_state == "paused" else "#4ade80"
        self.status_label.setStyleSheet(
            f"padding: 5px 10px; border-radius: 999px; background: {status_bg}; color: {status_fg}; "
            f"font-size: {max(12, self.font_size)}px; font-weight: 700;"
        )

    def apply_theme(self, theme: ViewerTheme):
        self.theme = theme
        self.setStyleSheet(
            f"QFrame {{ background: {theme.panel_bg}; border: 1px solid {theme.panel_border}; "
            f"border-radius: 10px; }}"
            f"QLabel {{ border: 0; color: {theme.text}; background: transparent; }}"
            f"QToolButton {{ border: 0; border-radius: 6px; padding: 5px 8px; "
            f"background: {theme.button_bg}; color: {theme.button_fg}; }}"
            f"QToolButton:checked {{ background: {theme.accent}; color: {theme.axes_bg}; }}"
        )
        self.title_label.setStyleSheet(
            f"font-weight: 700; font-size: {self.font_size + 1}px; color: {theme.text};"
        )
        self.settings_button.setStyleSheet(f"font-size: {max(11, self.font_size)}px;")
        self._apply_status_style()
        self.stats_label.setStyleSheet(
            f"padding: 6px 8px; border-radius: 6px; background: {theme.stats_bg}; color: {theme.subtext}; "
            f"font-size: {max(12, self.font_size - 1)}px; font-family: monospace;"
        )

        self.figure.set_facecolor(theme.figure_bg)
        self.axes.set_facecolor(theme.axes_bg)
        self.axes.title.set_color(theme.text)
        for spine in self.axes.spines.values():
            spine.set_color(theme.panel_border)
        self._style_colorbar()
        self._update_figure_layout()
        self.canvas.draw_idle()

    def refresh(self, force_draw=False):
        if self._disposed:
            return False
        snapshot = self.connection.poll()
        status_changed = snapshot.get("status_changed", False)
        if not snapshot["changed"] and not status_changed and not force_draw:
            return False

        frame = snapshot["frame"]
        vmin, vmax = self._resolve_color_limits(frame)
        self._linear_norm = Normalize(vmin=vmin, vmax=vmax)
        if self.log_scale:
            self.image.set_norm(LogNorm(vmin=max(vmin, 1e-6), vmax=max(vmax, vmin + 1e-6)))
        else:
            self.image.set_norm(self._linear_norm)
        self._update_stats(frame, snapshot["fps_text"])
        if snapshot["changed"] or force_draw:
            self.image.set_data(frame)
            self.image.set_clim(vmin, vmax)
            if self.colorbar is not None:
                self.colorbar.update_normal(self.image)
                self._style_colorbar()
            self.canvas.draw_idle()
            return True
        return False

    def dispose(self):
        if self._disposed:
            return
        self._disposed = True
        try:
            if self.colorbar is not None:
                self.colorbar.remove()
                self.colorbar = None
                self.colorbar_axes = None
        except Exception:
            logging.exception("Viewer panel colorbar teardown failed for %s", self.connection.name)
        try:
            self.figure.clear()
        except Exception:
            logging.exception("Viewer panel figure teardown failed for %s", self.connection.name)
        try:
            self.canvas.close()
        except Exception:
            logging.exception("Viewer panel canvas teardown failed for %s", self.connection.name)
        try:
            self.connection.close()
        except Exception:
            logging.exception("Viewer panel SHM teardown failed for %s", self.connection.name)

    def closeEvent(self, event):
        self.dispose()
        super().closeEvent(event)


class MosaicViewerWindow(QMainWindow):
    """Top-level window that arranges multiple stream panels in a grid.

    The window manages global viewer state such as theme selection, refresh
    cadence, grid geometry, and bulk operations across all panels. It is the
    main application object behind ``pyrtc-view``.
    """

    def __init__(
        self,
        shm_names,
        fps,
        geometry,
        pixel_scale,
        static_vmin=None,
        static_vmax=None,
        theme_name="dark",
        show_colorbar=False,
        show_stats=True,
        show_range=True,
    ):
        _require_viewer_backend()
        super().__init__()
        self.fps = max(1, int(fps))
        self.theme_name = theme_name if theme_name in THEMES else "dark"
        self.rows, self.cols = resolve_grid(len(shm_names), geometry)
        self.cells = list(shm_names)
        self.font_size = 15
        while len(self.cells) < self.rows * self.cols:
            self.cells.append(None)

        connections = [StreamConnection(name) for name in shm_names]
        frames = [connection.cached_frame for connection in connections]
        width, height = compute_window_size(frames, self.rows, self.cols, pixel_scale)

        self.setWindowTitle(f"{' '.join(shm_names)} - PyRTC Viewer")
        self.resize(width, height)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        root_layout = QVBoxLayout(central_widget)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(12)

        toolbar_layout = QHBoxLayout()
        toolbar_layout.setContentsMargins(0, 0, 0, 0)
        self.summary_label = QLabel("Composite stream viewer")
        toolbar_layout.addWidget(self.summary_label)
        toolbar_layout.addStretch(1)

        self.settings_button = QToolButton()
        self.settings_button.setText("Settings")
        self.settings_button.setPopupMode(QToolButton.InstantPopup)
        self.settings_menu = QMenu(self)
        self.settings_button.setMenu(self.settings_menu)
        toolbar_layout.addWidget(self.settings_button)

        root_layout.addLayout(toolbar_layout)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        root_layout.addWidget(self.scroll_area, stretch=1)

        self.grid_host = QWidget()
        self.scroll_area.setWidget(self.grid_host)

        self.grid_frame = QWidget()
        self.grid_layout = QGridLayout(self.grid_frame)
        self.grid_layout.setContentsMargins(0, 0, 0, 0)
        self.grid_layout.setSpacing(12)

        host_layout = QVBoxLayout(self.grid_host)
        host_layout.setContentsMargins(0, 0, 0, 0)
        host_layout.setSpacing(8)

        center_layout = QHBoxLayout()
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(8)

        center_layout.addWidget(self.grid_frame, stretch=1)
        self.add_column_button = EdgeArrowButton("▶", self.add_column)
        self.add_column_button.setToolTip("Add column")
        center_layout.addWidget(self.add_column_button, alignment=Qt.AlignVCenter)
        host_layout.addLayout(center_layout)

        bottom_layout = QHBoxLayout()
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.addStretch(1)
        self.add_row_button = EdgeArrowButton("▼", self.add_row)
        self.add_row_button.setToolTip("Add row")
        bottom_layout.addWidget(self.add_row_button, alignment=Qt.AlignHCenter)
        bottom_layout.addStretch(1)
        host_layout.addLayout(bottom_layout)

        self.panels = {}
        self.placeholders = {}

        self._show_colorbars = show_colorbar
        self._show_stats = show_stats
        self._show_range = show_range
        self.static_vmin = static_vmin
        self.static_vmax = static_vmax
        self._registered_actions = []
        self._last_panel_errors = 0
        self._build_settings_menu()
        self.rebuild_grid()
        self.apply_theme(self.theme_name)

        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.PreciseTimer)
        self.timer.timeout.connect(self.refresh_panels)
        self.timer.start(max(1, 1000 // self.fps))

    def _pause_refresh(self):
        if not hasattr(self, "timer"):
            return False
        was_active = self.timer.isActive()
        if was_active:
            self.timer.stop()
        return was_active

    def _resume_refresh(self, was_active):
        if was_active and hasattr(self, "timer"):
            self.timer.start(max(1, 1000 // self.fps))

    def _clear_grid_widgets(self):
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            widget = item.widget()
            if widget is None:
                continue
            if isinstance(widget, Stream2DWidget):
                widget.dispose()
            widget.setParent(None)
            widget.deleteLater()

    def _build_settings_menu(self):
        self.settings_menu.clear()
        for action in self._registered_actions:
            self.removeAction(action)
        self._registered_actions = []

        self.theme_action = QAction(f"Theme: {self.theme_name.title()}", self)
        self.theme_action.setShortcut(QKeySequence("Ctrl+T"))
        self.theme_action.triggered.connect(self.toggle_theme)
        self.theme_action.setShortcutContext(Qt.WidgetWithChildrenShortcut)
        self.addAction(self.theme_action)
        self._registered_actions.append(self.theme_action)
        self.settings_menu.addAction(self.theme_action)

        self.increase_font_action = QAction("Increase Font Size", self)
        self.increase_font_action.setShortcut(QKeySequence.ZoomIn)
        self.increase_font_action.triggered.connect(lambda: self.set_font_size(min(22, self.font_size + 1)))
        self.increase_font_action.setShortcutContext(Qt.WidgetWithChildrenShortcut)
        self.addAction(self.increase_font_action)
        self._registered_actions.append(self.increase_font_action)
        self.settings_menu.addAction(self.increase_font_action)

        self.decrease_font_action = QAction("Decrease Font Size", self)
        self.decrease_font_action.setShortcut(QKeySequence.ZoomOut)
        self.decrease_font_action.triggered.connect(lambda: self.set_font_size(max(10, self.font_size - 1)))
        self.decrease_font_action.setShortcutContext(Qt.WidgetWithChildrenShortcut)
        self.addAction(self.decrease_font_action)
        self._registered_actions.append(self.decrease_font_action)
        self.settings_menu.addAction(self.decrease_font_action)

        font_menu = self.settings_menu.addMenu("Font Size Presets")
        for size in [11, 12, 13, 14, 16, 18]:
            action = QAction(f"{size} pt", self, checkable=True)
            action.setChecked(size == self.font_size)
            action.triggered.connect(lambda checked=False, chosen=size: self.set_font_size(chosen))
            font_menu.addAction(action)

        self.settings_menu.addSeparator()

        self.colorbar_all_action = QAction("Toggle Colorbars", self)
        self.colorbar_all_action.setShortcut(QKeySequence("Ctrl+Shift+C"))
        self.colorbar_all_action.triggered.connect(self.toggle_all_colorbars)
        self.colorbar_all_action.setShortcutContext(Qt.WidgetWithChildrenShortcut)
        self.addAction(self.colorbar_all_action)
        self._registered_actions.append(self.colorbar_all_action)
        self.settings_menu.addAction(self.colorbar_all_action)

        self.stats_all_action = QAction("Toggle Stats", self)
        self.stats_all_action.setShortcut(QKeySequence("Ctrl+Shift+S"))
        self.stats_all_action.triggered.connect(self.toggle_all_stats)
        self.stats_all_action.setShortcutContext(Qt.WidgetWithChildrenShortcut)
        self.addAction(self.stats_all_action)
        self._registered_actions.append(self.stats_all_action)
        self.settings_menu.addAction(self.stats_all_action)

        self.range_all_action = QAction("Toggle Value Range", self)
        self.range_all_action.setShortcut(QKeySequence("Ctrl+Shift+R"))
        self.range_all_action.triggered.connect(self.toggle_all_ranges)
        self.range_all_action.setShortcutContext(Qt.WidgetWithChildrenShortcut)
        self.addAction(self.range_all_action)
        self._registered_actions.append(self.range_all_action)
        self.settings_menu.addAction(self.range_all_action)

        self.settings_menu.addSeparator()

        self.add_row_action = QAction("Add Row", self)
        self.add_row_action.setShortcut(QKeySequence("Ctrl+Down"))
        self.add_row_action.triggered.connect(self.add_row)
        self.add_row_action.setShortcutContext(Qt.WidgetWithChildrenShortcut)
        self.addAction(self.add_row_action)
        self._registered_actions.append(self.add_row_action)
        self.settings_menu.addAction(self.add_row_action)

        self.add_column_action = QAction("Add Column", self)
        self.add_column_action.setShortcut(QKeySequence("Ctrl+Right"))
        self.add_column_action.triggered.connect(self.add_column)
        self.add_column_action.setShortcutContext(Qt.WidgetWithChildrenShortcut)
        self.addAction(self.add_column_action)
        self._registered_actions.append(self.add_column_action)
        self.settings_menu.addAction(self.add_column_action)

    def _set_summary(self, changed_panels=0):
        active = sum(cell is not None for cell in self.cells)
        geometry = normalize_geometry_value(self.rows * self.cols, f"{self.rows}x{self.cols}")
        summary = (
            f"Composite stream viewer  |  layout={geometry}  |  active={active}/{self.rows * self.cols}  |  updated={changed_panels}"
        )
        if self._last_panel_errors:
            summary += f"  |  panel_errors={self._last_panel_errors}"
        self.summary_label.setText(summary)

    def _apply_to_panels(self, action_name, panel_callback):
        error_count = 0
        for index, panel in self.panels.items():
            try:
                panel_callback(panel)
            except Exception:
                error_count += 1
                logging.exception("Viewer panel action '%s' failed for cell %s (%s)", action_name, index, panel.connection.name)
        self._last_panel_errors = error_count
        return error_count

    def rebuild_grid(self):
        refresh_was_active = self._pause_refresh()
        self._clear_grid_widgets()

        self.panels = {}
        self.placeholders = {}

        for index, name in enumerate(self.cells):
            row_index = index // self.cols
            col_index = index % self.cols
            if name is None:
                placeholder = AddPlotPlaceholder(lambda checked=False, idx=index: self.add_plot_at(idx))
                self.placeholders[index] = placeholder
                self.grid_layout.addWidget(placeholder, row_index, col_index)
            else:
                panel = Stream2DWidget(
                    StreamConnection(name),
                    remove_callback=lambda checked=False, idx=index: self.remove_plot_at(idx),
                    static_vmin=self.static_vmin,
                    static_vmax=self.static_vmax,
                    show_colorbar=self._show_colorbars,
                    show_stats=self._show_stats,
                    show_range=self._show_range,
                    font_size=self.font_size,
                )
                self.panels[index] = panel
                self.grid_layout.addWidget(panel, row_index, col_index)

        self._set_summary(0)
        self.apply_theme(self.theme_name)
        self._resume_refresh(refresh_was_active)

    def add_plot_at(self, index):
        shm_name, ok = QInputDialog.getText(self, "Add SHM", "Shared-memory stream name:")
        if not ok or not shm_name.strip():
            return
        self.cells[index] = shm_name.strip()
        self.rebuild_grid()

    def remove_plot_at(self, index):
        self.cells[index] = None
        self.rebuild_grid()

    def add_row(self):
        self.rows += 1
        self.cells.extend([None] * self.cols)
        self.rebuild_grid()

    def add_column(self):
        new_cells = []
        for row_index in range(self.rows):
            start = row_index * self.cols
            end = start + self.cols
            new_cells.extend(self.cells[start:end])
            new_cells.append(None)
        self.cols += 1
        self.cells = new_cells
        self.rebuild_grid()

    def set_font_size(self, font_size):
        self.font_size = font_size
        self._build_settings_menu()
        self.rebuild_grid()

    def toggle_theme(self):
        next_theme = "light" if self.theme_name == "dark" else "dark"
        self.apply_theme(next_theme)

    def apply_theme(self, theme_name):
        self.theme_name = theme_name if theme_name in THEMES else "dark"
        theme = THEMES[self.theme_name]
        self.theme_action.setText(f"Theme: {self.theme_name.title()}\tCtrl+T")
        self.centralWidget().setStyleSheet(
            f"QWidget {{ background: {theme.window_bg}; color: {theme.text}; }}"
            f"QPushButton, QComboBox, QToolButton {{ background: {theme.button_bg}; color: {theme.button_fg}; "
            f"border: 1px solid {theme.panel_border}; border-radius: 6px; padding: 6px 10px; }}"
            f"QToolButton:checked {{ background: {theme.accent}; color: {theme.axes_bg}; }}"
            f"QLabel {{ color: {theme.text}; }}"
            f"QMenu {{ background: {theme.panel_bg}; color: {theme.text}; border: 1px solid {theme.panel_border}; }}"
            f"QMenu::item:selected {{ background: {theme.button_bg}; }}"
        )
        self.summary_label.setStyleSheet(f"font-size: {self.font_size + 1}px; font-weight: 600;")
        self.settings_button.setStyleSheet(f"font-size: {self.font_size}px;")
        self.add_row_button.setStyleSheet(f"font-size: {self.font_size + 8}px;")
        self.add_column_button.setStyleSheet(f"font-size: {self.font_size + 8}px;")
        for panel in self.panels.values():
            panel.apply_theme(theme)
        for placeholder in self.placeholders.values():
            placeholder.apply_theme(theme)

    def refresh_panels(self):
        if not self.isVisible():
            return
        changed_panels = 0
        error_count = 0
        for index, panel in self.panels.items():
            try:
                if panel.refresh():
                    changed_panels += 1
            except Exception:
                error_count += 1
                logging.exception("Viewer panel refresh failed for cell %s (%s)", index, panel.connection.name)
        self._last_panel_errors = error_count
        self._set_summary(changed_panels)

    def toggle_all_colorbars(self):
        self._show_colorbars = not self._show_colorbars
        self._apply_to_panels("toggle_colorbar", lambda panel: panel._toggle_colorbar(self._show_colorbars))
        self._set_summary(0)

    def toggle_all_stats(self):
        self._show_stats = not self._show_stats
        self._apply_to_panels("toggle_stats", lambda panel: panel._toggle_stats(self._show_stats))
        self._set_summary(0)

    def toggle_all_ranges(self):
        self._show_range = not self._show_range
        self._apply_to_panels("toggle_range", lambda panel: panel._toggle_range(self._show_range))
        self._set_summary(0)

    def closeEvent(self, event):
        self.timer.stop()
        self._clear_grid_widgets()
        self.panels = {}
        self.placeholders = {}
        event.accept()


def launch_mosaic_viewer(argv, shm_names, fps, geometry, pixel_scale, static_vmin, static_vmax, theme_name):
    """Create the Qt application, size the window, and start the event loop."""

    _require_viewer_backend()
    app = QApplication(argv)
    window = MosaicViewerWindow(
        shm_names,
        fps,
        geometry,
        pixel_scale,
        static_vmin=static_vmin,
        static_vmax=static_vmax,
        theme_name=theme_name,
    )
    screen = app.primaryScreen()
    if screen is not None:
        available = screen.availableGeometry()
        max_width = int(available.width() * 0.78)
        max_height = int(available.height() * 0.78)
        window.resize(min(window.width(), max_width), min(window.height(), max_height))
    window.show()
    return app.exec_()