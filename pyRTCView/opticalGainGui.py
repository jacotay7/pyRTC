import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QPushButton, QWidget, QLineEdit, QLabel, QHBoxLayout

from pyRTC.Pipeline import *

class PlotCanvas(FigureCanvas):
    def __init__(self, ogShm, parent=None):
        fig, self.ax = plt.subplots()
        super().__init__(fig)
        self.setParent(parent)
        self.ogShm = ogShm
        self.plot()

    def plot(self):
        vec = self.ogShm.read_noblock()
        t = np.arange(vec.size)
        s = np.squeeze(vec)
        self.ax.clear()
        self.ax.plot(t, s)
        self.ax.set_title('pyRTC Optical Gain', size = 18)
        self.ax.set_xlabel('Mode #', size = 16)
        self.ax.set_ylabel('OG', size = 16)
        self.draw()

class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.title = 'pyRTC Optical Gain GUI'

        self.ogShm, self.ogDims, self.ogDtype  = initExistingShm("og")

        self.left = 100
        self.top = 100
        self.width = 800
        self.height = 600
        self.initUI()
        self.factor = 1

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.plot_canvas = PlotCanvas( self.ogShm, parent = self)
        layout.addWidget(self.plot_canvas)

        button_layout = QHBoxLayout()

        self.plus_button = QPushButton('+', self)
        self.plus_button.clicked.connect(self.increase_factor)
        button_layout.addWidget(self.plus_button)

        self.minus_button = QPushButton('-', self)
        self.minus_button.clicked.connect(self.decrease_factor)
        button_layout.addWidget(self.minus_button)

        self.zero_all_button = QPushButton('ZERO GAINS', self)
        self.zero_all_button.clicked.connect(self.zero_gains)
        button_layout.addWidget(self.zero_all_button)

        layout.addLayout(button_layout)

        input_layout = QHBoxLayout()

        self.label1 = QLabel('Mode #:', self)
        input_layout.addWidget(self.label1)
        self.textbox1 = QLineEdit(self)
        input_layout.addWidget(self.textbox1)

        self.label2 = QLabel('Step Size:', self)
        input_layout.addWidget(self.label2)
        self.textbox2 = QLineEdit(self)
        input_layout.addWidget(self.textbox2)

        layout.addLayout(input_layout)

    def increase_factor(self):
        curOg = self.ogShm.read_noblock()
        try:
            mode = int(self.textbox1.text())
            step = float(self.textbox2.text())
            curOg[mode] += step
            self.ogShm.write(curOg)
        except Exception as e:
            print(e)
            
        self.plot_canvas.plot()

    def decrease_factor(self):
        curOg = self.ogShm.read_noblock()
        try:
            mode = int(self.textbox1.text())
            step = float(self.textbox2.text())
            curOg[mode] -= step
            self.ogShm.write(curOg)
        except:
            pass
        self.plot_canvas.plot()

    def zero_gains(self):
        self.ogShm.write(0*self.ogShm.read_noblock())
        self.plot_canvas.plot()
        return

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec_())
