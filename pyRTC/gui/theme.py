"""Theme tokens and stylesheet helpers for the manager GUI."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GUITheme:
    name: str
    window_bg: str
    panel_bg: str
    panel_alt_bg: str
    border: str
    text: str
    subtext: str
    accent: str
    button_bg: str
    button_fg: str
    selection_bg: str
    running: str
    degraded: str
    failed: str
    stopped: str
    edge: str
    canvas_bg: str


THEMES = {
    "dark": GUITheme(
        name="dark",
        window_bg="#11161d",
        panel_bg="#1a2330",
        panel_alt_bg="#152133",
        border="#2e3b4e",
        text="#edf2f7",
        subtext="#a8b3c4",
        accent="#6ee7c8",
        button_bg="#223044",
        button_fg="#edf2f7",
        selection_bg="#203247",
        running="#16a34a",
        degraded="#f59e0b",
        failed="#ef4444",
        stopped="#64748b",
        edge="#7b8ba3",
        canvas_bg="#0f1722",
    ),
    "light": GUITheme(
        name="light",
        window_bg="#eef3f8",
        panel_bg="#ffffff",
        panel_alt_bg="#eef4fa",
        border="#c6d1dc",
        text="#162033",
        subtext="#536179",
        accent="#0f766e",
        button_bg="#dde7f1",
        button_fg="#162033",
        selection_bg="#d9e8f7",
        running="#15803d",
        degraded="#c2410c",
        failed="#dc2626",
        stopped="#64748b",
        edge="#7b8ba3",
        canvas_bg="#f7fafc",
    ),
}


def theme_names() -> tuple[str, ...]:
    return tuple(THEMES)


def get_theme(name: str | None) -> GUITheme:
    if not name:
        return THEMES["dark"]
    return THEMES.get(name, THEMES["dark"])


def build_main_window_stylesheet(theme: GUITheme) -> str:
    return f"""
    QMainWindow, QWidget {{
        background: {theme.window_bg};
        color: {theme.text};
    }}
    QFrame#Panel, QListWidget, QPlainTextEdit, QLineEdit, QComboBox, QScrollArea, QGraphicsView {{
        background: {theme.panel_bg};
        color: {theme.text};
        border: 1px solid {theme.border};
        border-radius: 10px;
    }}
    QLabel#SubtleText {{
        color: {theme.subtext};
    }}
    QPushButton, QToolButton, QComboBox {{
        background: {theme.button_bg};
        color: {theme.button_fg};
        border: 1px solid {theme.border};
        border-radius: 8px;
        padding: 6px 10px;
    }}
    QPushButton:hover, QToolButton:hover, QComboBox:hover {{
        border-color: {theme.accent};
    }}
    QListWidget::item:selected {{
        background: {theme.selection_bg};
        color: {theme.text};
    }}
    QDockWidget::title {{
        background: {theme.panel_alt_bg};
        color: {theme.text};
        padding: 6px 8px;
    }}
    QStatusBar {{
        background: {theme.panel_alt_bg};
        color: {theme.subtext};
    }}
    """