from pyRTC.Pipeline import *
from pyRTC.utils import *

class Modulator:

    def __init__(self, conf) -> None:

        self.alive = True
        self.running = False

        return

    def __del__(self):
        self.stop()
        self.alive=False
        return

    def start(self):
        self.running = True
        return

    def stop(self):
        self.running = False
        return     