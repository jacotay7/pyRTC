"""XIMEA wavefront-sensor camera adapter.

This module provides the camera-facing portion of a Shack-Hartmann-style
wavefront sensor built on XIMEA hardware. It maps pyrtc configuration and frame
acquisition semantics onto the ``xiapi`` SDK so the rest of the pipeline can
interact with the device through the standard ``WavefrontSensor`` interface.
"""

import time

from pyrtc.logging_utils import get_logger
from pyrtc.manager import launch_component
from pyrtc.wavefront_sensor import WavefrontSensor
from ximea import xiapi


logger = get_logger(__name__)


class XIMEAWFS(WavefrontSensor):
    """Wavefront-sensor adapter for a XIMEA camera.

    The class handles device connection, runtime camera configuration, and
    frame capture for XIMEA-backed sensors. It intentionally focuses on the raw
    image transport layer; slope extraction and other wavefront-sensing logic
    remain in the normal pyrtc processing components.
    """

    def __init__(self, conf):
        try:
            super().__init__(conf)
            self.cam = xiapi.Camera()
            self.cam.open_device_by("XI_OPEN_BY_SN", conf["serial"])

            self.downsampled_image = None
            if "bit_depth" in conf:
                self.set_bit_depth(conf["bit_depth"])
            if "binning" in conf:
                self.set_binning(conf["binning"])
            if "exposure" in conf:
                self.set_exposure(conf["exposure"])
            if "top" in conf and "left" in conf and "width" in conf and "height" in conf:
                roi = [conf["width"], conf["height"], conf["left"], conf["top"]]
                self.set_roi(roi)
            if "gain" in conf:
                self.set_gain(conf["gain"])

            self.img = xiapi.Image()
            self.cam.start_acquisition()
            self.logger.info("Initialized XIMEA wavefront sensor serial=%s", conf["serial"])
        except Exception:
            logger.exception("Failed to initialize XIMEA wavefront sensor")
            raise

        return

    def set_roi(self, roi):
        try:
            super().set_roi(roi)
            self.cam.set_param("width", self.roi_width)
            self.cam.set_param("height", self.roi_height)
            self.cam.set_param("offset_x", self.roi_left)
            self.cam.set_param("offset_y", self.roi_top)
            self.logger.info("Applied XIMEA ROI %s", roi)
        except Exception:
            self.logger.exception("Failed to apply XIMEA ROI %s", roi)
            raise
        return

    def set_exposure(self, exposure):
        try:
            super().set_exposure(exposure)
            self.cam.set_param("exposure", self.exposure)
            self.logger.info("Applied XIMEA exposure=%s", self.exposure)
        except Exception:
            self.logger.exception("Failed to apply XIMEA exposure=%s", exposure)
            raise
        return

    def set_binning(self, binning):
        try:
            super().set_binning(binning)
            if self.binning == 2:
                self.cam.set_param("downsampling", "XI_DWN_2x2")
            self.logger.info("Applied XIMEA binning=%s", self.binning)
        except Exception:
            self.logger.exception("Failed to apply XIMEA binning=%s", binning)
            raise
        return

    def set_gain(self, gain):
        try:
            super().set_gain(gain)
            self.cam.set_param("gain", self.gain)
            self.logger.info("Applied XIMEA gain=%s", self.gain)
        except Exception:
            self.logger.exception("Failed to apply XIMEA gain=%s", gain)
            raise
        return

    def set_bit_depth(self, bit_depth):
        try:
            super().set_bit_depth(bit_depth)
            if self.bit_depth > 8:
                self.cam.set_param("imgdataformat", "XI_MONO16")
            self.logger.info("Applied XIMEA bit_depth=%s", self.bit_depth)
        except Exception:
            self.logger.exception("Failed to apply XIMEA bit_depth=%s", bit_depth)
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
    launch_component(XIMEAWFS, "wfs", start=True)
