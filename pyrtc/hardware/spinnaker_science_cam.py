"""FLIR/Spinnaker science-camera adapter.

The implementation in this module connects a Spinnaker-compatible camera to the
pyrtc ``ScienceCamera`` abstraction. It applies runtime configuration such as
ROI, exposure, gain, gamma, and pixel format through the vendor API, then
publishes frames into the normal pyrtc science-camera pipeline.
"""

import numpy as np

from pyrtc.logging_utils import get_logger
from pyrtc.manager import launch_component
from pyrtc.science_camera import ScienceCamera
from rotpy.camera import CameraList
from rotpy.system import SpinSystem


logger = get_logger(__name__)

class SpinCam(ScienceCamera):
    """Science-camera wrapper for cameras exposed through ``rotpy``.

    This adapter is intended for hardware deployments that use the FLIR
    Spinnaker stack. It owns camera startup and shutdown, mirrors pyrtc camera
    settings into the device node map, and converts acquired frames into the
    numpy arrays expected by downstream pyrtc consumers.
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

            if "bit_depth" in conf:
                self.set_bit_depth(conf["bit_depth"])
            self.camera.camera_nodes.ExposureAuto.set_node_value_from_str('Off', verify=True)
            self.camera.camera_nodes.GainAuto.set_node_value_from_str('Off', verify=True)
            if "binning" in conf:
                self.set_binning(conf["binning"])
            if "exposure" in conf:
                self.set_exposure(conf["exposure"])
            if "top" in conf and "left" in conf and "width" in conf and "height" in conf:
                roi = [conf["width"], conf["height"], conf["left"], conf["top"]]
                self.set_roi(roi)
            if "gain" in conf:
                self.set_gain(conf["gain"])
            if "gamma" in conf:
                self.set_gamma(conf["gamma"])

            self.camera.begin_acquisition()
            self.logger.info("Initialized Spinnaker science camera index=%s", self.index)
        except Exception:
            logger.exception("Failed to initialize Spinnaker science camera")
            raise

        return

    def set_roi(self, roi):
        try:
            super().set_roi(roi)
            self.camera.camera_nodes.OffsetX.set_node_value(0)
            self.camera.camera_nodes.OffsetY.set_node_value(0)
            self.camera.camera_nodes.Height.set_node_value(self.roi_height)
            self.camera.camera_nodes.Width.set_node_value(self.roi_width)
            self.camera.camera_nodes.OffsetX.set_node_value(self.roi_left)
            self.camera.camera_nodes.OffsetY.set_node_value(self.roi_top)
            self.logger.info("Applied Spinnaker ROI %s", roi)
        except Exception:
            self.logger.exception("Failed to apply Spinnaker ROI %s", roi)
            raise
        return
    
    def set_exposure(self, exposure):
        try:
            super().set_exposure(exposure)
            self.camera.camera_nodes.ExposureTime.set_node_value(exposure, verify=True)
            self.logger.info("Applied Spinnaker exposure=%s", exposure)
        except Exception:
            self.logger.exception("Failed to apply Spinnaker exposure=%s", exposure)
            raise
        return
    
    def set_binning(self, binning):
        try:
            super().set_binning(binning)
            self.logger.info("Applied Spinnaker binning=%s", self.binning)
        except Exception:
            self.logger.exception("Failed to apply Spinnaker binning=%s", binning)
            raise
        return
    
    def set_gain(self, gain):
        try:
            super().set_gain(gain)
            self.camera.camera_nodes.Gain.set_node_value(self.gain)
            self.logger.info("Applied Spinnaker gain=%s", self.gain)
        except Exception:
            self.logger.exception("Failed to apply Spinnaker gain=%s", gain)
            raise
        return

    def set_gamma(self, gamma):
        try:
            super().set_gamma(gamma)
            self.gamma = np.clip(self.gamma, 0.5, 3.9)
            self.camera.camera_nodes.Gamma.set_node_value(self.gamma)
            self.logger.info("Applied Spinnaker gamma=%s", self.gamma)
        except Exception:
            self.logger.exception("Failed to apply Spinnaker gamma=%s", gamma)
            raise
        return

    def set_bit_depth(self, bit_depth):
        try:
            super().set_bit_depth(bit_depth)
            if self.bit_depth == 8:
                self.camera.camera_nodes.PixelFormat.set_node_value_from_str('Mono8', verify=True)
            elif self.bit_depth == 16:
                self.camera.camera_nodes.PixelFormat.set_node_value_from_str('Mono16', verify=True)
            self.logger.info("Applied Spinnaker bit_depth=%s", self.bit_depth)
        except Exception:
            self.logger.exception("Failed to apply Spinnaker bit_depth=%s", bit_depth)
            raise

        return

    def expose(self):
        
        self.img = self.camera.get_next_image(timeout=5)
        self.data = np.ndarray(self.image_shape, 
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

    launch_component(SpinCam, "psf", start = True)
        
