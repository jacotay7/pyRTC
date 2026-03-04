from pyRTC.pyRTCComponent import pyRTCComponent
from pyRTC.utils import setFromConfig

class Modulator(pyRTCComponent):
    """
    A placeholder class for any modulator specific logic. See hardware/PIModulator for an 
    implementation.
    """
    def __init__(self, conf) -> None:
        self.name = setFromConfig(conf, "name", 'modulator')
        return
 