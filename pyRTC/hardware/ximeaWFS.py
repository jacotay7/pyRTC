from pyRTC.WavefrontSensor import *
from ximea import xiapi

class XIMEA_WFS(WavefrontSensor):

    def __init__(self, exposure = None, roi = None, binning = None, gain = None, bitDepth = None):
        
        self.cam = xiapi.Camera()
        self.cam.open_device()

        if not (bitDepth is None):
            self.setBitDepth(bitDepth)
        if not (binning is None):
            self.setBinning(binning)
        if not (exposure is None):
            self.setExposure(exposure)
        if not (roi is None):
            self.setRoi(roi)
        if not (gain is None):
            self.setGain(gain)


        self.img = xiapi.Image()
       
        super().__init__((self.cam.get_width(),self.cam.get_height()))
        self.cam.start_acquisition()
        return

    def setRoi(self, roi):
        super().setRoi(roi)

        self.cam.set_param('width', self.roiWidth)
        self.cam.set_param('height', self.roiHeight)
        self.cam.set_param('offsetX', self.roiLeft)
        self.cam.set_param('offsetY', self.roiTop)
        return
    
    def setExposure(self, exposure):
        super().setExposure(exposure)
        self.cam.set_param('exposure', self.exposure)
        return
    
    def setBinning(self, binning):
        super().setBinning(binning)
        if self.binning == 2:
            self.cam.set_param('downsampling', "XI_DWN_2x2")
        return
    
    def setGain(self, gain):
        super().setGain(gain)
        self.cam.set_param('gain', self.gain)
        return
    
    def setBitDepth(self, bitDepth):
        super().setBitDepth(bitDepth)
        if self.bitDepth > 8:
            self.cam.set_param('imgdataformat', "XI_MONO16")
        return

    def expose(self):
        
        self.cam.get_image(self.img)
        self.data = np.ndarray((self.img.width,self.img.height), 
                               buffer= self.img.get_image_data_raw(), 
                               dtype=np.uint16)
        super().expose()

        return

    def __del__(self):
        super().__del__()
        self.cam.stop_acquisition()
        self.cam.close_device()
        
        return