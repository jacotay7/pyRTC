from pyRTC.ScienceCamera import *
from pyRTC.Pipeline import *
from pyRTC.utils import *
from rotpy.system import SpinSystem
from rotpy.camera import CameraList
import argparse
import sys
import os 

class spinCam(ScienceCamera):

    def __init__(self, conf):
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
            roi=[conf["width"],conf["height"],conf["left"],conf["top"]]
            self.setRoi(roi)
        if "gain" in conf:
            self.setGain(conf["gain"])
        
        self.camera.begin_acquisition()

        return

    def setRoi(self, roi):
        super().setRoi(roi)

        self.camera.camera_nodes.OffsetX.set_node_value(0)
        self.camera.camera_nodes.OffsetY.set_node_value(0)
        self.camera.camera_nodes.Height.set_node_value(self.roiHeight)
        self.camera.camera_nodes.Width.set_node_value(self.roiWidth)
        self.camera.camera_nodes.OffsetX.set_node_value(self.roiLeft)
        self.camera.camera_nodes.OffsetY.set_node_value(self.roiTop)
        return
    
    def setExposure(self, exposure):
        super().setExposure(exposure)
        self.camera.camera_nodes.ExposureTime.set_node_value(exposure, verify=True)
        return
    
    def setBinning(self, binning):
        super().setBinning(binning)
        # if self.binning == 2:
        #     self.cam.set_param('downsampling', "XI_DWN_2x2")
        return
    
    def setGain(self, gain):
        super().setGain(gain)
        self.camera.camera_nodes.Gain.set_node_value(self.gain)
        return
    
    def setBitDepth(self, bitDepth):
        super().setBitDepth(bitDepth)
        if self.bitDepth == 8:
            self.camera.camera_nodes.PixelFormat.set_node_value_from_str('Mono8', verify=True)
        elif self.bitDepth == 16:
            self.camera.camera_nodes.PixelFormat.set_node_value_from_str('Mono16', verify=True)

        return

    def expose(self):
        
        self.img = self.camera.get_next_image(timeout=5)
        self.data = np.ndarray(self.imageShape, 
                               buffer= self.img.get_image_data(), 
                               dtype=np.uint16)
        super().expose()

        return

    def __del__(self):
        super().__del__()
        self.camera.end_acquisition()
        self.camera.deinit_cam()
        self.camera.release()
        
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
    os.sched_setaffinity(pid, {conf["psf"]["affinity"],})
    decrease_nice(pid)

    psf = spinCam(conf=conf["psf"])
    psf.start()

    l = Listener(psf, port = int(args.port))
    while l.running:
        l.listen()
        time.sleep(1e-3)
        
