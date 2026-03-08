"""FLIR/Spinnaker science-camera adapter.

The implementation in this module connects a Spinnaker-compatible camera to the
pyRTC ``ScienceCamera`` abstraction. It applies runtime configuration such as
ROI, exposure, gain, gamma, and pixel format through the vendor API, then
publishes frames into the normal pyRTC science-camera pipeline.
"""

import numpy as np

from pyRTC.logging_utils import get_logger
from pyRTC.Pipeline import launchComponent
from pyRTC.ScienceCamera import ScienceCamera
from rotpy.camera import CameraList
from rotpy.system import SpinSystem


logger = get_logger(__name__)

class spinCam(ScienceCamera):
    """Science-camera wrapper for cameras exposed through ``rotpy``.

    This adapter is intended for hardware deployments that use the FLIR
    Spinnaker stack. It owns camera startup and shutdown, mirrors pyRTC camera
    settings into the device node map, and converts acquired frames into the
    numpy arrays expected by downstream pyRTC consumers.
    """

    def __init__(self, conf):
        try:
            super().__init__(conf)

            system = SpinSystem()
            cameras = CameraList.create_from_system(system, update_cams=True, update_interfaces=True)

            self.index = conf["index"]
            self.camera = cameras.create_camera_by_index(self.index)
            self.camera.init_cam()
            self.dtype = np.uint16

            if "bitDepth" in conf:
                self.setBitDepth(conf["bitDepth"])
            self.camera.camera_nodes.ExposureAuto.set_node_value_from_str('Off', verify=True)
            self.camera.camera_nodes.GainAuto.set_node_value_from_str('Off', verify=True)
            if "binning" in conf:
                self.setBinning(conf["binning"])
            if "exposure" in conf:
                self.setExposure(conf["exposure"])
            if "top" in conf and "left" in conf and "width" in conf and "height" in conf:
                roi = [conf["width"], conf["height"], conf["left"], conf["top"]]
                self.setRoi(roi)
            if "gain" in conf:
                self.setGain(conf["gain"])
            if "gamma" in conf:
                self.setGamma(conf["gamma"])

            self.camera.begin_acquisition()
            self.logger.info("Initialized Spinnaker science camera index=%s", self.index)
        except Exception:
            logger.exception("Failed to initialize Spinnaker science camera")
            raise

        return

    def setRoi(self, roi):
        try:
            super().setRoi(roi)
            self.camera.camera_nodes.OffsetX.set_node_value(0)
            self.camera.camera_nodes.OffsetY.set_node_value(0)
            self.camera.camera_nodes.Height.set_node_value(self.roiHeight)
            self.camera.camera_nodes.Width.set_node_value(self.roiWidth)
            self.camera.camera_nodes.OffsetX.set_node_value(self.roiLeft)
            self.camera.camera_nodes.OffsetY.set_node_value(self.roiTop)
            self.logger.info("Applied Spinnaker ROI %s", roi)
        except Exception:
            self.logger.exception("Failed to apply Spinnaker ROI %s", roi)
            raise
        return
    
    def setExposure(self, exposure):
        try:
            super().setExposure(exposure)
            self.camera.camera_nodes.ExposureTime.set_node_value(exposure, verify=True)
            self.logger.info("Applied Spinnaker exposure=%s", exposure)
        except Exception:
            self.logger.exception("Failed to apply Spinnaker exposure=%s", exposure)
            raise
        return
    
    def setBinning(self, binning):
        try:
            super().setBinning(binning)
            self.logger.info("Applied Spinnaker binning=%s", self.binning)
        except Exception:
            self.logger.exception("Failed to apply Spinnaker binning=%s", binning)
            raise
        return
    
    def setGain(self, gain):
        try:
            super().setGain(gain)
            self.camera.camera_nodes.Gain.set_node_value(self.gain)
            self.logger.info("Applied Spinnaker gain=%s", self.gain)
        except Exception:
            self.logger.exception("Failed to apply Spinnaker gain=%s", gain)
            raise
        return

    def setGamma(self, gamma):
        try:
            super().setGamma(gamma)
            self.gamma = np.clip(self.gamma, 0.5, 3.9)
            self.camera.camera_nodes.Gamma.set_node_value(self.gamma)
            self.logger.info("Applied Spinnaker gamma=%s", self.gamma)
        except Exception:
            self.logger.exception("Failed to apply Spinnaker gamma=%s", gamma)
            raise
        return

    def setBitDepth(self, bitDepth):
        try:
            super().setBitDepth(bitDepth)
            if self.bitDepth == 8:
                self.camera.camera_nodes.PixelFormat.set_node_value_from_str('Mono8', verify=True)
            elif self.bitDepth == 16:
                self.camera.camera_nodes.PixelFormat.set_node_value_from_str('Mono16', verify=True)
            self.logger.info("Applied Spinnaker bitDepth=%s", self.bitDepth)
        except Exception:
            self.logger.exception("Failed to apply Spinnaker bitDepth=%s", bitDepth)
            raise

        return

    def expose(self):
        
        self.img = self.camera.get_next_image(timeout=5)
        self.data = np.ndarray(self.imageShape, 
                               buffer= self.img.get_image_data(), 
                               dtype=np.uint16)
        super().expose()

        return

    def __del__(self):
        component_logger = getattr(self, "logger", logger)
        try:
            super().__del__()
        finally:
            camera = getattr(self, "camera", None)
            if camera is not None:
                try:
                    camera.end_acquisition()
                    camera.deinit_cam()
                    camera.release()
                    component_logger.info("Closed Spinnaker science camera")
                except Exception:
                    component_logger.exception("Failed while closing Spinnaker science camera")
        
        return

if __name__ == "__main__":

    launchComponent(spinCam, "psf", start = True)
        
