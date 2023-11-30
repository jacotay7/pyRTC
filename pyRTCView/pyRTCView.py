import sys
import numpy as np
from multiprocessing import shared_memory
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from pyRTC.Pipeline import ImageSHM

def read_shared_memory(shm_arr):
    return np.copy(shm_arr)

class RealTimeView(QMainWindow):
    def __init__(self,  shm_name, shm_width, shm_height, shm_dtype, fps):
        super().__init__()

        self.shm = ImageSHM(shm_name, (shm_width, shm_height), shm_dtype)

        self.setWindowTitle('PyRTC Viewer')
        self.setGeometry(100, 100, 800, 600)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        # Create Matplotlib Figure and Axes
        self.figure = Figure(figsize=(8, 8), tight_layout=True)
        self.axes = self.figure.add_subplot(111)
        frame = self.shm.read()
        im = self.axes.imshow(frame, cmap='inferno', interpolation='nearest', 
                                origin='upper',vmin = np.min(frame), vmax = np.max(frame))
        self.cbar = self.figure.colorbar(im)
        # Create Matplotlib canvas
        self.canvas = FigureCanvas(self.figure)
        central_layout = QVBoxLayout(central_widget)
        central_layout.addWidget(self.canvas)

        self.update_view()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_view)
        self.timer.start(1000 //fps)

    def update_view(self):
        try:
            frame = self.shm.read()
            
            # Clear the previous plot
            self.axes.clear()
            # Plot the image using imshow
            im = self.axes.imshow(frame, cmap='inferno', interpolation='nearest', 
                                    origin='upper',vmin = np.min(frame), vmax = np.max(frame))
            self.cbar.set_ticks(np.linspace(np.min(frame), np.max(frame), 10))
            # Refresh the canvas
            self.canvas.draw()
        except:
            print("Unable to View Image")


    def closeEvent(self, event):
        # Code to execute when the window is closed
        self.shm.close()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    shm_name, shm_width, shm_height = sys.argv[1:]
    shm_width = int(shm_width)
    shm_height = int(shm_height)
    view = RealTimeView(shm_name, shm_width, shm_height,np.float64, 30)

    view.show()
    sys.exit(app.exec_())
