import argparse
import sys

import numpy as np
from matplotlib.colors import LogNorm

from pyRTC import ImageSHM, utils


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="View a pyRTC shared-memory stream in real time.")
    parser.add_argument("shm", type=str, help="Shared-memory stream name to display")
    parser.add_argument("vmin", type=float, nargs="?", default=None, help="Static minimum colorbar value")
    parser.add_argument("vmax", type=float, nargs="?", default=None, help="Static maximum colorbar value")
    parser.add_argument("--fps", type=int, default=30, help="Refresh rate in frames per second")
    parser.add_argument("--affinity", type=int, default=0, help="CPU affinity index")
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
        def __init__(self, shm_name, fps, static_vmin=None, static_vmax=None):
            super().__init__()

            self.old_count = 0
            self.old_time = 0
            self.static_vmin = static_vmin
            self.static_vmax = static_vmax
            self.vmax = static_vmax
            self.vmin = static_vmin
            self.log = False

            self.metadata = ImageSHM(shm_name + "_meta", (ImageSHM.METADATA_SIZE,), np.float64)
            metadata = self.metadata.read_noblock()
            shm_width, shm_height = int(metadata[4]), int(metadata[5])

            shm_height = max(1, shm_height)
            shm_width = max(1, shm_width)

            shm_dtype = utils.float_to_dtype(metadata[3])
            self.shm = ImageSHM(shm_name, (shm_width, shm_height), shm_dtype)

            self.setWindowTitle(f"{shm_name} - PyRTC Viewer")
            self.setGeometry(100, 100, 800, 600)

            central_widget = QWidget(self)
            self.setCentralWidget(central_widget)

            self.figure = Figure(figsize=(8, 8), tight_layout=True)
            self.axes = self.figure.add_subplot(111)

            frame = self.shm.read_noblock()

            aspect = None
            aspect_cap = 10
            if shm_width / shm_height < 1 / aspect_cap or shm_width / shm_height > aspect_cap:
                aspect = "auto"

            self.updateVminVmax(frame)
            self.im = self.axes.imshow(
                frame,
                cmap="inferno",
                interpolation="nearest",
                aspect=aspect,
                origin="upper",
                vmin=self.vmin,
                vmax=self.vmax,
            )

            self.fpsText = self.axes.text(
                frame.shape[1] // 2,
                int(1.15 * frame.shape[0]),
                "PAUSED",
                fontsize=18,
                ha="center",
                va="bottom",
                color="g",
            )

            self.LinearNorm = self.im.norm
            self.cbar = self.figure.colorbar(self.im, ax=self.axes)

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
            frame = self.shm.read_noblock()
            metadata = self.metadata.read_noblock()
            new_count = metadata[0]
            new_time = metadata[1]
            if new_time > self.old_time:
                speed_fps = np.round((new_count - self.old_count) / (new_time - self.old_time), 2)
                speed_fps = str(speed_fps) + "FPS"
            else:
                speed_fps = "PAUSED"

            self.old_count = new_count
            self.old_time = new_time
            if isinstance(frame, np.ndarray):
                self.updateVminVmax(frame)

                self.fpsText.set_text(str(speed_fps))
                self.im.set_data(frame)

                self.im.set_clim(self.vmin, self.vmax)

                self.cbar.update_normal(self.im)

                self.canvas.draw()

        def updateVminVmax(self, frame):
            vmin, vmax = np.min(frame), np.max(frame)
            if self.log:
                vmin, vmax = max(vmin, vmax / 1e3), max(1e-2, vmax)
            self.vmin, self.vmax = vmin, vmax
            if self.static_vmax is not None:
                self.vmax = self.static_vmax
            if self.static_vmin is not None:
                self.vmin = self.static_vmin

        def toggleLog(self):
            self.log = not self.log
            if self.log:
                self.im.set_norm(LogNorm(vmin=1e-2, vmax=1))
            else:
                self.im.set_norm(self.LinearNorm)

        def closeEvent(self, event):
            self.shm.close()
            event.accept()

    utils.set_affinity(args.affinity)

    app = QApplication(sys.argv)
    view = RealTimeView(args.shm, args.fps, static_vmin=args.vmin, static_vmax=args.vmax)
    view.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
