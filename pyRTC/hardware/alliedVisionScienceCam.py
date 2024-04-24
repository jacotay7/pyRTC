from pyRTC.ScienceCamera import *
from pyRTC.Pipeline import *
from pyRTC.utils import *

import argparse
import os 
import sys
if sys.platform == "win32":
    os.environ["VIMBA_X_HOME"] = "C:\Program Files\Allied Vision\Vimba X"
import vmbpy


class alliedVisionScienceCam(ScienceCamera):

    def __init__(self, conf):
        super().__init__(conf)

        self.index = setFromConfig(conf, "index", 0)
        self.dtype = np.uint16

        self.vmb = vmbpy.VmbSystem.get_instance()
        with self.vmb:
            self.cam = self.vmb.get_all_cameras()[self.index]

            # self.setup_camera()

            if "bitDepth" in conf:
                self.setBitDepth(conf["bitDepth"])
            # if "binning" in conf:
            #     self.setBinning(conf["binning"])
            if "exposure" in conf:
                self.setExposure(conf["exposure"])
            if "top" in conf and "left" in conf and "width" in conf and "height" in conf:
                roi=[conf["width"],conf["height"],conf["left"],conf["top"]]
                self.setRoi(roi)
            # if "gain" in conf:
            #     self.setGain(conf["gain"])

        return

    def setup_camera(self):
        with self.cam as cam:
            try:
                stream = cam.get_streams()[0]
                stream.GVSPAdjustPacketSize.run()
                while not stream.GVSPAdjustPacketSize.is_done():
                    pass

            except (AttributeError, vmbpy.VmbFeatureError):
                pass

    def setRoi(self, roi):
        # ROI should be a tuple (offsetX, offsetY, width, height)
        super().setRoi(roi)
        with self.cam as cam:
            cam.Width.set(self.roiWidth)
            cam.Height.set(self.roiHeight)
            cam.OffsetX.set(self.roiLeft)
            cam.OffsetY.set(self.roiTop)

    def setExposure(self, exposure):
        # Exposure time in microseconds
        super().setExposure(exposure)
        # with self.cam as cam:
        #     cam.ExposureTime.set(exposure)
    
    def setBinning(self, binning):
        super().setBinning(binning)
        return
    
    def setGain(self, gain):
        # Gain value
        super().setGain(gain)
        # self.cam.Gain.set(gain)
    
    def setBitDepth(self, bitDepth):
        # Bit depth, typically one of the pixel format options
        super().setBitDepth(bitDepth)
        with self.cam as cam:
            if self.bitDepth == 16:
                cam.set_pixel_format(vmbpy.PixelFormat.Mono16)
            elif self.bitDepth == 14:
                cam.set_pixel_format(vmbpy.PixelFormat.Mono14) 
            elif self.bitDepth == 8:
                cam.set_pixel_format(vmbpy.PixelFormat.Mono8) 
                # Ensure correct PixelFormat value is used

    def expose(self):
        # Capture an image
        super().expose()
        with self.vmb:
            with self.cam as cam:
                self.data = np.squeeze(cam.get_frame().as_numpy_ndarray())

    def __del__(self):
        super().__del__()

        
        return

if __name__ == "__main__":

    # Create argument parser
    parser = argparse.ArgumentParser(description="Read a config file from the command line.")

    # Add command-line argument for the config file
    parser.add_argument("-c", "--config", required=True, help="Path to the config file")
    parser.add_argument("-p", "--port", required=True, help="Port for communication")
    parser.add_argument("-h", "--host", required=False, help="Host IP for communication")

    # Parse command-line arguments
    args = parser.parse_args()

    host = '127.0.0.1'
    if args.host is not None:
        host = args.host

    conf = read_yaml_file(args.config)

    pid = os.getpid()
    set_affinity((conf["psf"]["affinity"])%os.cpu_count()) 
    decrease_nice(pid)

    psf = alliedVisionScienceCam(conf=conf["psf"])
    psf.start()

    l = Listener(psf, port = int(args.port), host= host)
    while l.running:
        l.listen()
        time.sleep(1e-3)
        
