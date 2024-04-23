from pyRTC.WavefrontSensor import *
from pyRTC.Pipeline import *
from pyRTC.utils import *
import argparse
import os
import sys
#I had to remove some functions from the FliSdk_V2.py to get it to work.
sys.path.append("/opt/FirstLightImaging/FliSdk/Examples/Python/Polling_Grab_One_Camera")
import FliSdk_V2
import ctypes

CLBKFUNC = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.POINTER(ctypes.c_byte), ctypes.c_void_p)


class FliCBlueOneWFS(WavefrontSensor):

    def __init__(self, conf):
        super().__init__(conf)
        self.newImage = False
        self.context = FliSdk_V2.Init()
        listOfGrabbers = FliSdk_V2.DetectGrabbers(self.context)

        print(listOfGrabbers)

        if len(listOfGrabbers) == 0:
            print("No grabber detected, exit.")
            return

        listOfCameras = FliSdk_V2.DetectCameras(self.context)

        if len(listOfCameras) == 0 or listOfCameras[0] == '':
            print("No camera detected, exit.")
            return
        
        print(listOfCameras)
        ok = FliSdk_V2.SetCamera(self.context, listOfCameras[conf["index"]])
        if not ok:
            print("Error Opening Camera.")
            return
        
        FliSdk_V2.SetMode(self.context, FliSdk_V2.Mode.Full)
        ok = FliSdk_V2.Update(self.context)
        if not ok:
            print("Error while updating SDK.")
            return 
        # if "bitDepth" in conf:
        #     self.setBitDepth(conf["bitDepth"])
        # if "binning" in conf:
        #     self.setBinning(conf["binning"])
        if "exposure" in conf:
            self.setExposure(conf["exposure"])
        # if "top" in conf and "left" in conf and "width" in conf and "height" in conf:
        #     roi=[conf["width"],conf["height"],conf["left"],conf["top"]]
        #     self.setRoi(roi)
        # if "gain" in conf:
        #     self.setGain(conf["gain"])
        self.clbk_func = CLBKFUNC(self.imageCallback)
        UserContext = 4
        FliSdk_V2.EnableRingBuffer(self.context, True)
        callbackContext = FliSdk_V2.AddCallBackNewImage(self.context, self.clbk_func, 0, False, UserContext)
        FliSdk_V2.Start(self.context)
        return


    
    def imageCallback(self, image, context = None):
        self.postImage()

        return
    
    def postImage(self):
        self.newImage = True
        return

    def setRoi(self, roi):
        super().setRoi(roi)

        return
    
    def setExposure(self, exposure):
        super().setExposure(exposure)

        if FliSdk_V2.IsSerialCamera(self.context):
            res, fps = FliSdk_V2.FliSerialCamera.GetFps(self.context)
        elif FliSdk_V2.IsCblueSfnc(self.context):
            res, fps = FliSdk_V2.FliCblueSfnc.GetAcquisitionFrameRate(self.context)
        print("Original camera FPS: " + str(fps))


        fps = float(1e6/float(exposure)) # for exposure in microseconds
        print("Setting FPS: " + str(fps))
        if FliSdk_V2.IsSerialCamera(self.context):
            FliSdk_V2.FliSerialCamera.SetFps(self.context, fps)
        elif FliSdk_V2.IsCblueSfnc(self.context):
            FliSdk_V2.FliCblueSfnc.SetAcquisitionFrameRate(self.context, fps)
        
        if FliSdk_V2.IsSerialCamera(self.context):
            res, fps = FliSdk_V2.FliSerialCamera.GetFps(self.context)
        elif FliSdk_V2.IsCblueSfnc(self.context):
            res, fps = FliSdk_V2.FliCblueSfnc.GetAcquisitionFrameRate(self.context)
        print("New Camera FPS (should be close to desired value): " + str(fps))
        

        exp = FliSdk_V2.FliCblueSfnc.GetExposureTime(self.context)
        print(exp)
        return
    
    def setBinning(self, binning):
        super().setBinning(binning)
        return
    
    def setGain(self, gain):
        super().setGain(gain)
        return
    
    def setBitDepth(self, bitDepth):
        super().setBitDepth(bitDepth)
        return

    def expose(self):
        super().expose()
        # while not FliSdk_V2.IsGrabNFinished(self.context):
        while not self.newImage:
            time.sleep(1e-5)
        self.newImage = False
        self.data =  FliSdk_V2.GetRawImageAsNumpyArray(self.context, -1)
        # img = FliSdk_V2.GetRawImageAsNumpyArray(self.context, -1)
        # img[img < 0 ] = 0
        # self.data = img.astype(np.uint16)
        # self.data = FliSdk_V2.GetProcessedImageGrayscale16bNumpyArray(self.context, -1)
        return

    def __del__(self):
        super().__del__()
        FliSdk_V2.Stop(self.context)
        FliSdk_V2.Exit(self.context)
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
    wfs = FliCBlueOneWFS(conf=confWFS)

    wfs.start()
    
    l = Listener(wfs, port= int(args.port))
    while l.running:
        l.listen()
        time.sleep(1e-3)