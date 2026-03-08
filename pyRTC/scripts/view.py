import argparse
import math
import sys

import numpy as np
from matplotlib.colors import LogNorm

from pyRTC import ImageSHM, utils


def _is_float_token(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def _split_targets_and_limits(items):
    shms = list(items)
    vmin = None
    vmax = None

    if len(shms) >= 3 and _is_float_token(shms[-1]) and _is_float_token(shms[-2]):
        vmax = float(shms.pop())
        vmin = float(shms.pop())
    elif len(shms) >= 2 and _is_float_token(shms[-1]):
        vmax = float(shms.pop())

    if not shms:
        raise ValueError("At least one shared-memory stream name is required")

    return shms, vmin, vmax


def _normalize_frame(frame):
    frame = np.asarray(frame)
    if frame.ndim == 1:
        return frame.reshape(1, frame.size)
    if frame.ndim == 2:
        return frame
    return frame.reshape(frame.shape[0], -1)


def _read_shm_metadata(shm_name):
    metadata_shm = ImageSHM(shm_name + "_meta", (ImageSHM.METADATA_SIZE,), np.float64)
    metadata = metadata_shm.read_noblock()
    shm_dtype = utils.float_to_dtype(metadata[3])
    shm_dims = []
    index = 0
    while 4 + index < metadata.size and int(metadata[4 + index]) > 0:
        shm_dims.append(int(metadata[4 + index]))
        index += 1
    if not shm_dims:
        shm_dims = [1]
    return metadata_shm, tuple(shm_dims), shm_dtype


def _resolve_grid(num_plots: int, geometry: str):
    geometry = geometry.lower()
    if geometry == "row":
        return 1, num_plots
    if geometry == "column":
        return num_plots, 1
    if geometry == "square":
        cols = int(math.ceil(math.sqrt(num_plots)))
        rows = int(math.ceil(num_plots / cols))
        return rows, cols
    if "x" in geometry:
        parts = geometry.split("x", 1)
        try:
            rows = int(parts[0])
            cols = int(parts[1])
        except ValueError as exc:
            raise ValueError(f"Invalid geometry string: {geometry}") from exc
        if rows < 1 or cols < 1:
            raise ValueError(f"Invalid geometry string: {geometry}")
        if rows * cols < num_plots:
            raise ValueError(
                f"Geometry {geometry} does not have enough cells for {num_plots} SHMs"
            )
        return rows, cols
    raise ValueError(
        "Geometry must be one of: square, row, column, or an explicit grid like 2x3"
    )


def _compute_window_size(frames, rows, cols, pixel_scale):
    max_height = max(frame.shape[0] for frame in frames)
    max_width = max(frame.shape[1] for frame in frames)
    plot_width = cols * max_width * pixel_scale
    plot_height = rows * max_height * pixel_scale
    width = int(min(1600, max(260, plot_width + cols * 120)))
    height = int(min(1200, max(220, plot_height + rows * 120 + 60)))
    return width, height


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="View a pyRTC shared-memory stream in real time.")
    parser.add_argument(
        "items",
        nargs="+",
        help=(
            "One or more SHM names to display. For backward compatibility you can still append "
            "optional vmin and vmax values at the end, for example: pyrtc-view signal2D -1 1"
        ),
    )
    parser.add_argument("--fps", type=int, default=30, help="Refresh rate in frames per second")
    parser.add_argument("--affinity", type=int, default=0, help="CPU affinity index")
    parser.add_argument(
        "--geometry",
        default="square",
        help="Layout for multiple SHMs: square, row, column, or an explicit grid like 2x3",
    )
    parser.add_argument(
        "--pixel-scale",
        type=float,
        default=12.0,
        help="Approximate window pixels per SHM pixel when auto-sizing the viewer window",
    )
    return parser


def main(argv=None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    try:
        from PyQt5.QtCore import QTimer
        from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure
    except ImportError as exc:
        raise SystemExit(
            "pyrtc-view requires viewer dependencies. Install with: pip install pyRTC[viewer]"
        ) from exc

    class RealTimeView(QMainWindow):
        def __init__(self, shm_names, fps, geometry, pixel_scale, static_vmin=None, static_vmax=None):
            super().__init__()

            self.old_count = 0
            self.old_time = 0
            self.static_vmin = static_vmin
            self.static_vmax = static_vmax
            self.log = False

            self.streams = []
            frames = []
            for shm_name in shm_names:
                metadata_shm, shm_shape, shm_dtype = _read_shm_metadata(shm_name)
                shm = ImageSHM(shm_name, shm_shape, shm_dtype)
                frame = _normalize_frame(shm.read_noblock())
                self.streams.append(
                    {
                        "name": shm_name,
                        "metadata": metadata_shm,
                        "shm": shm,
                        "frame_shape": frame.shape,
                        "vmin": static_vmin,
                        "vmax": static_vmax,
                    }
                )
                frames.append(frame)

            rows, cols = _resolve_grid(len(self.streams), geometry)

            self.setWindowTitle(f"{' '.join(shm_names)} - PyRTC Viewer")
            window_width, window_height = _compute_window_size(frames, rows, cols, pixel_scale)
            self.setGeometry(100, 100, window_width, window_height)

            central_widget = QWidget(self)
            self.setCentralWidget(central_widget)

            self.figure = Figure(
                figsize=(max(3.0, window_width / 100.0), max(2.5, window_height / 100.0)),
                tight_layout=True,
            )
            subplot_axes = self.figure.subplots(rows, cols, squeeze=False)
            self.plot_entries = []

            for index, stream in enumerate(self.streams):
                row_index = index // cols
                col_index = index % cols
                axes = subplot_axes[row_index][col_index]
                frame = frames[index]

                aspect = None
                aspect_cap = 10
                frame_ratio = frame.shape[1] / max(frame.shape[0], 1)
                if frame_ratio < 1 / aspect_cap or frame_ratio > aspect_cap:
                    aspect = "auto"

                vmin, vmax = self.updateVminVmax(stream, frame)
                image = axes.imshow(
                    frame,
                    cmap="inferno",
                    interpolation="nearest",
                    aspect=aspect,
                    origin="upper",
                    vmin=vmin,
                    vmax=vmax,
                )
                fps_text = axes.text(
                    0.5,
                    1.02,
                    "PAUSED",
                    fontsize=10,
                    transform=axes.transAxes,
                    ha="center",
                    va="bottom",
                    color="g",
                )
                axes.set_title(stream["name"])
                linear_norm = image.norm
                colorbar = self.figure.colorbar(image, ax=axes, fraction=0.046, pad=0.04)
                self.plot_entries.append(
                    {
                        "stream": stream,
                        "axes": axes,
                        "image": image,
                        "fpsText": fps_text,
                        "LinearNorm": linear_norm,
                        "colorbar": colorbar,
                    }
                )

            for empty_index in range(len(self.streams), rows * cols):
                row_index = empty_index // cols
                col_index = empty_index % cols
                subplot_axes[row_index][col_index].axis("off")

            self.logButton = QPushButton("Toggle Log Colorbar")
            self.logButton.clicked.connect(self.toggleLog)

            self.canvas = FigureCanvas(self.figure)
            central_layout = QVBoxLayout(central_widget)
            central_layout.addWidget(self.canvas)
            central_layout.addWidget(self.logButton)

            self.update_view()

            self.timer = QTimer(self)
            self.timer.timeout.connect(self.update_view)
            self.timer.start(1000 // fps)

        def update_view(self):
            should_draw = False
            for entry in self.plot_entries:
                frame = _normalize_frame(entry["stream"]["shm"].read_noblock())
                metadata = entry["stream"]["metadata"].read_noblock()
                old_count = entry["stream"].get("old_count", 0)
                old_time = entry["stream"].get("old_time", 0)
                new_count = metadata[0]
                new_time = metadata[1]
                if new_time > old_time:
                    speed_fps = np.round((new_count - old_count) / (new_time - old_time), 2)
                    speed_fps = str(speed_fps) + "FPS"
                else:
                    speed_fps = "PAUSED"

                entry["stream"]["old_count"] = new_count
                entry["stream"]["old_time"] = new_time
                if isinstance(frame, np.ndarray):
                    vmin, vmax = self.updateVminVmax(entry["stream"], frame)
                    entry["fpsText"].set_text(str(speed_fps))
                    entry["image"].set_data(frame)
                    entry["image"].set_clim(vmin, vmax)
                    entry["colorbar"].update_normal(entry["image"])
                    should_draw = True

            if should_draw:
                self.canvas.draw_idle()

        def updateVminVmax(self, stream, frame):
            vmin, vmax = np.min(frame), np.max(frame)
            if self.log:
                vmin, vmax = max(vmin, vmax / 1e3), max(1e-2, vmax)
            if self.static_vmax is not None:
                vmax = self.static_vmax
            if self.static_vmin is not None:
                vmin = self.static_vmin
            stream["vmin"] = vmin
            stream["vmax"] = vmax
            return vmin, vmax

        def toggleLog(self):
            self.log = not self.log
            for entry in self.plot_entries:
                if self.log:
                    entry["image"].set_norm(LogNorm(vmin=1e-2, vmax=1))
                else:
                    entry["image"].set_norm(entry["LinearNorm"])
            self.update_view()

        def closeEvent(self, event):
            for stream in self.streams:
                stream["shm"].close()
                stream["metadata"].close()
            event.accept()

    try:
        shm_names, static_vmin, static_vmax = _split_targets_and_limits(args.items)
        _resolve_grid(len(shm_names), args.geometry)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    utils.set_affinity(args.affinity)

    app = QApplication(sys.argv)
    view = RealTimeView(
        shm_names,
        args.fps,
        args.geometry,
        args.pixel_scale,
        static_vmin=static_vmin,
        static_vmax=static_vmax,
    )
    view.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
