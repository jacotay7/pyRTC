"""Qt binding selection shared by the manager GUI and the stream viewer.

The GUI and viewer import Qt through `qtpy <https://github.com/spyder-ide/qtpy>`_
and support the Qt6 bindings only: PySide6 (installed by the ``gui`` and
``viewer`` extras) or PyQt6. Qt5 reached end of life in 2025 and is rejected.

Binding choice, in order:

1. ``QT_API`` (``pyside6`` or ``pyqt6``), if set, as qtpy documents it.
2. A Qt binding the process already imported (qtpy reuses it).
3. PySide6, then PyQt6, whichever imports first.

Nothing here changes ``os.environ``. When no usable binding is installed,
:func:`require_qt6` raises :class:`ImportError`, and the GUI modules replace
their Qt names with :func:`unavailable_qt_class` stand-ins, so they still
import (for tests and ``--help``) and only fail when a window is built.
"""

from __future__ import annotations

import importlib
import os
import sys

_QT_BINDINGS = ("PySide6", "PyQt6", "PyQt5", "PySide2")
_QT6_PREFERENCE = ("PySide6", "PyQt6")


def _preload_preferred_binding() -> None:
    """Import the preferred Qt6 binding so qtpy selects it over Qt5."""

    if os.environ.get("QT_API") or any(name in sys.modules for name in _QT_BINDINGS):
        return
    for name in _QT6_PREFERENCE:
        try:
            importlib.import_module(f"{name}.QtCore")
        except ImportError:
            continue
        return


def require_qt6() -> str:
    """Load qtpy on a Qt6 binding and return its name (``PySide6``/``PyQt6``).

    Raises :class:`ImportError` if qtpy or a Qt6 binding is missing, or if
    qtpy resolved to a Qt5 binding (for example through ``QT_API=pyqt5``).
    """

    _preload_preferred_binding()
    import qtpy

    if not qtpy.QT6:
        raise ImportError(
            f"pyrtc needs a Qt6 binding (PySide6 or PyQt6), but qtpy selected {qtpy.API_NAME}. "
            "Install PySide6, or set QT_API=pyside6 or QT_API=pyqt6."
        )
    return qtpy.API_NAME


class _UnavailableValue:
    """Placeholder for Qt enum values when Qt is unavailable.

    Attribute access and ``|`` return the placeholder itself, so class-level
    references such as ``Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop``
    still evaluate.
    """

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return self

    def __or__(self, other):
        return self

    __ror__ = __or__


_UNAVAILABLE_VALUE = _UnavailableValue()


class _UnavailableMeta(type):
    def __getattr__(cls, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _UNAVAILABLE_VALUE


def unavailable_qt_class(message: str, error: BaseException | None):
    """Return a stand-in for Qt classes when no Qt6 binding can be imported.

    The stand-in can be subclassed and its attributes (enums) read, but
    instantiating it, or a subclass, raises :class:`ImportError` with
    ``message``, chained to ``error``.
    """

    class QtUnavailable(metaclass=_UnavailableMeta):
        def __init__(self, *args, **kwargs):
            raise ImportError(message) from error

    return QtUnavailable
