import sys
import numpy as np
from multiprocessing import shared_memory
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from pyRTC.Pipeline import ImageSHM
from pyRTC.utils import *
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import matplotlib
import os

def read_shared_memory(shm_arr):
    return np.copy(shm_arr)

class RealTimeView(QMainWindow):
    def __init__(self, shm_name, fps):
        super().__init__()

        self.old_count = 0
        self.old_time = 0
        self.metadata = ImageSHM(shm_name+"_meta", (ImageSHM.METADATA_SIZE,), np.float64)
        metadata = self.metadata.read_noblock()
        shm_width, shm_height = int(metadata[4]),  int(metadata[5])

        shm_height = max(1,shm_height)
        shm_width = max(1,shm_width)

        shm_dtype = float_to_dtype(metadata[3])
        self.shm = ImageSHM(shm_name, (shm_width, shm_height), shm_dtype)
        
        self.setWindowTitle(f'{shm_name} - PyRTC Viewer')
        self.setGeometry(100, 100, 800, 600)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        # Create Matplotlib Figure and Axes
        self.figure = Figure(figsize=(8, 8), tight_layout=True)
        self.axes = self.figure.add_subplot(111)

        frame = self.shm.read_noblock_safe()

        aspect = None
        ASPECTCAP = 10
        if shm_width/shm_height < 1/ASPECTCAP or shm_width/shm_height > ASPECTCAP:
            aspect = "auto"

        self.im = self.axes.imshow(frame, cmap='inferno', interpolation='nearest', aspect=aspect,
                                origin='upper',vmin = np.min(frame), vmax = np.max(frame))
        
        self.fpsText = self.axes.text(frame.shape[1]//2,int(1.15*frame.shape[0]), 'PAUSED', fontsize=18, ha='center', va='bottom', color = 'g')

        self.LinearNorm = self.im.norm
        self.cbar = self.figure.colorbar(self.im, ax=self.axes)

        self.log = False
        self.logButton = QPushButton('Toggle Log Colorbar')
        self.logButton.clicked.connect(self.toggleLog)

        # plt.colorbar()
        # Create Matplotlib canvas
        self.canvas = FigureCanvas(self.figure)
        central_layout = QVBoxLayout(central_widget)
        central_layout.addWidget(self.canvas)
        central_layout.addWidget(self.logButton)

        self.update_view()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_view)
        self.timer.start(1000 //fps)

    def update_view(self):
        # try:
        frame = self.shm.read_noblock()
        metadata = self.metadata.read_noblock()
        new_count = metadata[0]
        new_time = metadata[1]
        if new_time > self.old_time:
            speed_fps = np.round((new_count - self.old_count)/(new_time- self.old_time),2)
            speed_fps = str(speed_fps) + "FPS"
        else:
            speed_fps = 'PAUSED'

        self.old_count = new_count
        self.old_time = new_time
        if isinstance(frame,np.ndarray):
            vmin, vmax = np.min(frame), np.max(frame)
            self.fpsText.set_text(str(speed_fps))
            # print(vmin,vmax)
            self.im.set_data(frame)
            if self.log:
                vmin, vmax = max(vmin,vmax/1e4), max(1e-2, vmax)
            self.im.set_clim(vmin, vmax)
                
            if isinstance(self.cbar,matplotlib.colorbar.Colorbar):
                try:
                    self.cbar.remove()
                    self.cbar = self.figure.colorbar(self.im, ax=self.axes)
                except:
                    pass
            self.canvas.draw()


    def toggleLog(self):
        self.log = not self.log
        if self.log:
            self.im.set_norm(LogNorm())
        else:
            self.im.set_norm(self.LinearNorm)
        return

    def closeEvent(self, event):
        # Code to execute when the window is closed
        self.shm.close()
        event.accept()

if __name__ == '__main__':
    pid = os.getpid()
    if sys.platform != 'darwin':
        os.sched_setaffinity(pid, {0,})

    app = QApplication(sys.argv)
    shm_name = sys.argv[1]

    view = RealTimeView(shm_name, 30)

    view.show()
    sys.exit(app.exec_())
