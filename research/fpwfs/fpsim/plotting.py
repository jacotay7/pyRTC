"""Shared figure style and the plots/ folder (untracked, one folder per experiment).

``save(fig, "exp01_ideal_loop", "strehl_vs_rate", caption)`` writes the PNG and
keeps ``plots/<exp>/README.md`` as an index of every figure with its caption,
so the folder reads on its own.
"""

from __future__ import annotations

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.lines  # noqa: E402,F401
import matplotlib.ticker  # noqa: E402,F401

PLOTS = pathlib.Path(__file__).resolve().parent.parent / "plots"

# Reference palette (dataviz skill): fixed categorical order, never cycled.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
SEQ = "Blues"  # sequential, one hue
DIV = "RdBu_r"  # diverging, neutral midpoint

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "text.color": INK,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        "axes.prop_cycle": matplotlib.cycler(color=SERIES),
        "figure.dpi": 110,
        "savefig.dpi": 140,
        "savefig.bbox": "tight",
    }
)


def _root_index() -> None:
    """plots/README.md: one entry per experiment folder, in order."""
    lines = ["# FPWFS experiment plots", "",
             "One folder per experiment; each folder's README has every figure with a caption.", ""]
    for folder in sorted(p for p in PLOTS.iterdir() if p.is_dir()):
        index_path = folder / "index.json"
        if not index_path.exists():
            continue
        index = json.loads(index_path.read_text())
        summary = index.get("summary", "").split("\n\n")[0].replace("\n", " ")
        lines += [f"- [{index['title']}]({folder.name}/README.md): {summary}"]
    (PLOTS / "README.md").write_text("\n".join(lines) + "\n")


def save(fig, experiment: str, name: str, caption: str, title: str | None = None) -> pathlib.Path:
    """Save ``fig`` to plots/<experiment>/<name>.png and index its caption."""
    folder = PLOTS / experiment
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    index_path = folder / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {"title": experiment, "figs": {}}
    if title:
        index["title"] = title
    index["figs"][name] = caption.strip()
    index_path.write_text(json.dumps(index, indent=1))
    lines = [f"# {index['title']}", "", index.get("summary", ""), ""]
    for fig_name, cap in index["figs"].items():
        lines += [f"## {fig_name}", "", f"![{fig_name}]({fig_name}.png)", "", cap, ""]
    (folder / "README.md").write_text("\n".join(lines))
    _root_index()
    return path


def note(experiment: str, text: str, title: str | None = None) -> None:
    """Set the experiment folder's title / summary paragraph (shown first)."""
    folder = PLOTS / experiment
    folder.mkdir(parents=True, exist_ok=True)
    index_path = folder / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {"title": experiment, "figs": {}}
    if title:
        index["title"] = title
    index["summary"] = text.strip()
    index_path.write_text(json.dumps(index, indent=1))
    lines = [f"# {index['title']}", "", index["summary"], ""]
    for fig_name, cap in index["figs"].items():
        lines += [f"## {fig_name}", "", f"![{fig_name}]({fig_name}.png)", "", cap, ""]
    (folder / "README.md").write_text("\n".join(lines))
