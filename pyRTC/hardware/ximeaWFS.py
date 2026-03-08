import time

from pyRTC.logging_utils import get_logger
from pyRTC.Pipeline import launchComponent
from pyRTC.WavefrontSensor import WavefrontSensor
from ximea import xiapi


logger = get_logger(__name__)

class XIMEA_WFS(WavefrontSensor):

    def __init__(self, conf):
        try:
            super().__init__(conf)
            self.cam = xiapi.Camera()
            self.cam.open_device_by("XI_OPEN_BY_SN", conf["serial"])

            self.downsampledImage = None
            if "bitDepth" in conf:
                self.setBitDepth(conf["bitDepth"])
            if "binning" in conf:
                self.setBinning(conf["binning"])
            if "exposure" in conf:
                self.setExposure(conf["exposure"])
            if "top" in conf and "left" in conf and "width" in conf and "height" in conf:
                roi = [conf["width"], conf["height"], conf["left"], conf["top"]]
                self.setRoi(roi)
            if "gain" in conf:
                self.setGain(conf["gain"])

            self.img = xiapi.Image()
            self.cam.start_acquisition()
            self.logger.info("Initialized XIMEA wavefront sensor serial=%s", conf["serial"])
        except Exception:
            logger.exception("Failed to initialize XIMEA wavefront sensor")
            raise

        return

    def setRoi(self, roi):
        try:
            super().setRoi(roi)
            self.cam.set_param('width', self.roiWidth)
            self.cam.set_param('height', self.roiHeight)
            self.cam.set_param('offsetX', self.roiLeft)
            self.cam.set_param('offsetY', self.roiTop)
            self.logger.info("Applied XIMEA ROI %s", roi)
        except Exception:
            self.logger.exception("Failed to apply XIMEA ROI %s", roi)
            raise
        return
    
    def setExposure(self, exposure):
        try:
            super().setExposure(exposure)
            self.cam.set_param('exposure', self.exposure)
            self.logger.info("Applied XIMEA exposure=%s", self.exposure)
        except Exception:
            self.logger.exception("Failed to apply XIMEA exposure=%s", exposure)
            raise
        return
    
    def setBinning(self, binning):
        try:
            super().setBinning(binning)
            if self.binning == 2:
                self.cam.set_param('downsampling', "XI_DWN_2x2")
            self.logger.info("Applied XIMEA binning=%s", self.binning)
        except Exception:
            self.logger.exception("Failed to apply XIMEA binning=%s", binning)
            raise
        return
    
    def setGain(self, gain):
        try:
            super().setGain(gain)
            self.cam.set_param('gain', self.gain)
            self.logger.info("Applied XIMEA gain=%s", self.gain)
        except Exception:
            self.logger.exception("Failed to apply XIMEA gain=%s", gain)
            raise
        return
    
    def setBitDepth(self, bitDepth):
        try:
            super().setBitDepth(bitDepth)
            if self.bitDepth > 8:
                self.cam.set_param('imgdataformat', "XI_MONO16")
            self.logger.info("Applied XIMEA bitDepth=%s", self.bitDepth)
        except Exception:
            self.logger.exception("Failed to apply XIMEA bitDepth=%s", bitDepth)
            raise
        return

    def expose(self):
        
        self.cam.get_image(self.img)
        
        # self.data = np.ndarray((self.img.width,self.img.height), 
        #                        buffer= self.img.get_image_data_raw(), 
        #                        dtype=np.uint16)
        # if self.binning > 2:
        #     # /2 is adjusted for on-chip binning
        #     self.data = downsample_uint16_image_jit(self.img.get_image_data_numpy(), self.binning//2)
        # else:
        self.data = self.img.get_image_data_numpy()
        super().expose()

        return

    def __del__(self):
        component_logger = getattr(self, "logger", logger)
        try:
            super().__del__()
        finally:
            cam = getattr(self, "cam", None)
            if cam is not None:
                try:
                    time.sleep(1e-1)
                    cam.stop_acquisition()
                    cam.close_device()
                    component_logger.info("Closed XIMEA wavefront sensor")
                except Exception:
                    component_logger.exception("Failed while closing XIMEA wavefront sensor")
        
        return
    
if __name__ == "__main__":

    launchComponent(XIMEA_WFS, "wfs", start = True)