from pyRTC.WavefrontSensor import *
from ximea import xiapi
from pyRTC.Pipeline import *
from pyRTC.utils import *
import argparse
import sys
import os 

class XIMEA_WFS(WavefrontSensor):

    def __init__(self, conf):
        super().__init__(conf)
        self.cam = xiapi.Camera()
        # self.cam.open_device()
        self.cam.open_device_by("XI_OPEN_BY_SN", conf["serial"])

        if "bitDepth" in conf:
            self.setBitDepth(conf["bitDepth"])
        if "binning" in conf:
            self.setBinning(conf["binning"])
        if "exposure" in conf:
            self.setExposure(conf["exposure"])
        if "top" in conf and "left" in conf and "width" in conf and "height" in conf:
            roi=[conf["width"],conf["height"],conf["left"],conf["top"]]
            self.setRoi(roi)
        if "gain" in conf:
            self.setGain(conf["gain"])


        self.img = xiapi.Image()
       

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
        time.sleep(1e-1)
        self.cam.stop_acquisition()
        self.cam.close_device()
        
        return
    
if __name__ == "__main__":

    # Create argument parser
    parser = argparse.ArgumentParser(description="Read a config file from the command line.")

    # Add command-line argument for the config file
    parser.add_argument("-c", "--config", required=True, help="Path to the config file")
    parser.add_argument("-p", "--port", required=True, help="Port for communication")

    # Parse command-line arguments
    args = parser.parse_args()

    conf = read_yaml_file(args.config)

    pid = os.getpid()
    set_affinity((conf["wfs"]["affinity"])%os.cpu_count()) 
    decrease_nice(pid)

    confWFS = conf["wfs"]
    wfs = XIMEA_WFS(conf=confWFS)

    wfs.start()
    
    l = Listener(wfs, port= int(args.port))
    while l.running:
        l.listen()
        time.sleep(1e-3)