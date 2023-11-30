"""
Loop Superclass
"""
from pyRTC.Pipeline import ImageSHM
import numpy as np
import matplotlib.pyplot as plt

class Loop:

    def __init__(self, wfs, wfc) -> None:
        self.wfs = wfs
        self.wfc = wfc

        self.IM = np.zeros((self.wfc.M2C.shape[1], self.wfs.read().size))
        return
    
    def __del__(self):
        return

    def start(self):
        return
    
    def stop(self):
        return

    def computeIM(self, pokeAmp, N = 100):

        self.wfc.flatten()
        for i in range(self.IM.shape[0]):

            
            correction = np.zeros_like(self.wfc.read())
            #Plus amplitude
            correction[i] = pokeAmp
            tmp_plus = np.zeros_like(self.IM[i])
            for n in range(N):
                self.wfc.write(correction)
                self.wfs.expose()
                self.wfs.computeSignal()
                tmp_plus += self.wfs.read()
            tmp_plus /= N

            #Minus amplitude
            self.wfc.flatten()
            correction[i] = -pokeAmp
            tmp_minus = np.zeros_like(self.IM[i])
            for n in range(N):
                self.wfc.write(correction)
                self.wfs.expose()
                self.wfs.computeSignal()
                tmp_minus += self.wfs.read()
            tmp_minus /= N

            self.IM[i] = (tmp_plus-tmp_minus)/(2*pokeAmp)
            self.wfc.flatten()
        return
    
    def plotIM(self, row=None):
        if not (row is None):
            row2D = self.wfs.signal2D(self.IM[row])
            plt.imshow(row2D, cmap = 'inferno')
            plt.colorbar()
            plt.show()
        else:
            plt.imshow(self.IM, cmap = 'inferno', aspect='auto')
            plt.show()

