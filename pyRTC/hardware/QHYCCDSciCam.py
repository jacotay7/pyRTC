from pyRTC.ScienceCamera import *
from pyRTC.Pipeline import *
from pyRTC.utils import *

import argparse
import os 

import time
import ctypes

"""
To get the QHY shared object to import properly, I needed to preload the opencv libraries. 
It seems to not come compiled agaisnt opencv, and python doesn't like that.
The best way to do this is to add them to your conda environment. This is the only way I could
get it to work in interactive python:

//Load your desired conda environment
conda activate <ENVIRONMENT_NAME>

//Create necessary files if not existant
cd $CONDA_PREFIX
mkdir -p ./etc/conda/activate.d
mkdir -p ./etc/conda/deactivate.d
touch ./etc/conda/activate.d/env_vars.sh
touch ./etc/conda/deactivate.d/env_vars.sh

//Open the activation env vars file
nano ./etc/conda/activate.d/env_vars.sh

//Write the following:
#!/bin/sh

export LD_PRELOAD='/usr/lib/x86_64-linux-gnu/libopencv_core.so:/usr/lib/x86_64-linux-gnu/libopencv_imgproc.so'

//Open the deactivation env vars  file
nano ./etc/conda/deactivate.d/env_vars.sh

//Write the following
#!/bin/sh

unset LD_PRELOAD
"""

def checkError(ret, funcName = "DEFAULT"):

    if ret < 0:
        print(f"QHYCCD: Error running {funcName}")
    return




"""
@brief CONTROL_ID enum define

List of function could be control
"""
class CONTROL_ID:
    CONTROL_BRIGHTNESS = ctypes.c_short(0) # image brightness
    CONTROL_CONTRAST = ctypes.c_short(1)   # image contrast
    CONTROL_WBR  = ctypes.c_short(2)       # red of white balance
    CONTROL_WBB = ctypes.c_short(3)        # blue of white balance
    CONTROL_WBG = ctypes.c_short(4)        # the green of white balance
    CONTROL_GAMMA = ctypes.c_short(5)      # screen gamma
    CONTROL_GAIN = ctypes.c_short(6)       # camera gain
    CONTROL_OFFSET = ctypes.c_short(7)     # camera offset
    CONTROL_EXPOSURE = ctypes.c_short(8)   # expose time (us)
    CONTROL_SPEED = ctypes.c_short(9)      # transfer speed
    CONTROL_TRANSFERBIT = ctypes.c_short(10)  # image depth bits
    CONTROL_CHANNELS = ctypes.c_short(11)     # image channels
    CONTROL_USBTRAFFIC = ctypes.c_short(12)   # hblank
    CONTROL_ROWNOISERE = ctypes.c_short(13)   # row denoise
    CONTROL_CURTEMP = ctypes.c_short(14)      # current cmos or ccd temprature
    CONTROL_CURPWM = ctypes.c_short(15)       # current cool pwm
    CONTROL_MANULPWM = ctypes.c_short(16)     # set the cool pwm
    CONTROL_CFWPORT = ctypes.c_short(17)      # control camera color filter wheel port
    CONTROL_COOLER = ctypes.c_short(18)       # check if camera has cooler
    CONTROL_ST4PORT = ctypes.c_short(19)      # check if camera has st4port
    CAM_COLOR = ctypes.c_short(20)
    CAM_BIN1X1MODE = ctypes.c_short(21)       # check if camera has bin1x1 mode
    CAM_BIN2X2MODE = ctypes.c_short(22)       # check if camera has bin2x2 mode
    CAM_BIN3X3MODE = ctypes.c_short(23)       # check if camera has bin3x3 mode
    CAM_BIN4X4MODE = ctypes.c_short(24)       # check if camera has bin4x4 mode
    CAM_MECHANICALSHUTTER = ctypes.c_short(25)# mechanical shutter
    CAM_TRIGER_INTERFACE = ctypes.c_short(26) # triger
    CAM_TECOVERPROTECT_INTERFACE = ctypes.c_short(27)  # tec overprotect
    CAM_SINGNALCLAMP_INTERFACE = ctypes.c_short(28)    # singnal clamp
    CAM_FINETONE_INTERFACE = ctypes.c_short(29)        # fine tone
    CAM_SHUTTERMOTORHEATING_INTERFACE = ctypes.c_short(30)  # shutter motor heating
    CAM_CALIBRATEFPN_INTERFACE = ctypes.c_short(31)         # calibrated frame
    CAM_CHIPTEMPERATURESENSOR_INTERFACE = ctypes.c_short(32)# chip temperaure sensor
    CAM_USBREADOUTSLOWEST_INTERFACE = ctypes.c_short(33)    # usb readout slowest

    CAM_8BITS = ctypes.c_short(34)                          # 8bit depth
    CAM_16BITS = ctypes.c_short(35)                         # 16bit depth
    CAM_GPS = ctypes.c_short(36)                            # check if camera has gps

    CAM_IGNOREOVERSCAN_INTERFACE = ctypes.c_short(37)       # ignore overscan area

    QHYCCD_3A_AUTOBALANCE = ctypes.c_short(38)
    QHYCCD_3A_AUTOEXPOSURE = ctypes.c_short(39)
    QHYCCD_3A_AUTOFOCUS = ctypes.c_short(40)
    CONTROL_AMPV = ctypes.c_short(41)                       # ccd or cmos ampv
    CONTROL_VCAM = ctypes.c_short(42)                       # Virtual Camera on off
    CAM_VIEW_MODE = ctypes.c_short(43)

    CONTROL_CFWSLOTSNUM = ctypes.c_short(44)         # check CFW slots number
    IS_EXPOSING_DONE = ctypes.c_short(45)
    ScreenStretchB = ctypes.c_short(46)
    ScreenStretchW = ctypes.c_short(47)
    CONTROL_DDR = ctypes.c_short(48)
    CAM_LIGHT_PERFORMANCE_MODE = ctypes.c_short(49)

    CAM_QHY5II_GUIDE_MODE = ctypes.c_short(50)
    DDR_BUFFER_CAPACITY = ctypes.c_short(51)
    DDR_BUFFER_READ_THRESHOLD = ctypes.c_short(52)
    DefaultGain = ctypes.c_short(53)
    DefaultOffset = ctypes.c_short(54)
    OutputDataActualBits = ctypes.c_short(55)
    OutputDataAlignment = ctypes.c_short(56)

    CAM_SINGLEFRAMEMODE = ctypes.c_short(57)
    CAM_LIVEVIDEOMODE = ctypes.c_short(58)
    CAM_IS_COLOR = ctypes.c_short(59)
    hasHardwareFrameCounter = ctypes.c_short(60)
    CONTROL_MAX_ID = ctypes.c_short(71)
    CAM_HUMIDITY = ctypes.c_short(72)
    #check if camera has	 humidity sensor 

class ERR:
    QHYCCD_READ_DIRECTLY = 0x2001
    QHYCCD_DELAY_200MS   = 0x2000
    QHYCCD_SUCCESS       = 0
    QHYCCD_ERROR         = 0xFFFFFFFF

class QHYCCD(ScienceCamera):

    def __init__(self, conf):
        super().__init__(conf)

        self.powerShm = ImageSHM("power", (1,), dtype=np.float64)
        self.noiseThrehold = 2
        self.dtype = np.uint8

        ctypes.cdll.LoadLibrary("/usr/lib/x86_64-linux-gnu/libopencv_core.so")
        ctypes.cdll.LoadLibrary("/usr/lib/x86_64-linux-gnu/libopencv_imgproc.so")
        self.qhyccd = ctypes.cdll.LoadLibrary("/usr/local/lib/libqhyccd.so")
        self.qhyccd.GetQHYCCDParam.restype = ctypes.c_double
        self.qhyccd.OpenQHYCCD.restype = ctypes.POINTER(ctypes.c_uint32)
        checkError(self.qhyccd.InitQHYCCDResource(),
                    "InitQHYCCDResource")
        checkError(self.qhyccd.ScanQHYCCD(),
                    "ScanQHYCCD")

        type_char_array_32 = ctypes.c_char * 32
        id = type_char_array_32()
        checkError(self.qhyccd.GetQHYCCDId(ctypes.c_int(0), id),
                    "GetQHYCCDId")
        print("Camera ID:", id.value)
        self.cam = self.qhyccd.OpenQHYCCD(id)
        checkError(self.qhyccd.SetQHYCCDStreamMode(self.cam, 1),
                    "SetQHYCCDStreamMode")  # Set to live mode
        checkError(self.qhyccd.InitQHYCCD(self.cam),
                    "InitQHYCCD")

        chipw = ctypes.c_double()
        chiph = ctypes.c_double()
        w = ctypes.c_uint()
        h = ctypes.c_uint()
        pixelw = ctypes.c_double()
        pixelh = ctypes.c_double() 
        self.bpp = ctypes.c_uint()
        self.channels = ctypes.c_uint32(1)
        checkError(self.qhyccd.GetQHYCCDChipInfo(self.cam, 
                                            ctypes.byref(chipw), 
                                            ctypes.byref(chiph), 
                                            ctypes.byref(w), 
                                            ctypes.byref(h),
                                            ctypes.byref(pixelw), 
                                            ctypes.byref(pixelh), 
                                            ctypes.byref(self.bpp)),
                                            "GetQHYCCDChipInfo")
        if "gain" in conf:
            self.setGain(conf["gain"])
        if "exposure" in conf:
            self.setExposure(conf["exposure"])
        if "bitDepth" in conf:
            self.setBitDepth(conf["bitDepth"])
        if "binning" in conf:
            self.setBinning(conf["binning"])
        self.usbtraffic = setFromConfig(conf, "usbtraffic", 20)
        checkError(self.qhyccd.SetQHYCCDParam(self.cam, 
                                    CONTROL_ID.CONTROL_SPEED,
                                ctypes.c_double(2)),
                                "SetQHYCCDParam CONTROL_SPEED")

        self.setUSBTraffic(self.usbtraffic)
        checkError(self.qhyccd.SetQHYCCDParam(self.cam, 
                                         CONTROL_ID.CONTROL_OFFSET, 
                                         ctypes.c_double(0)),
                                        "SetQHYCCDParam CONTROL OFFSET")
        if "top" in conf and "left" in conf and "width" in conf and "height" in conf:
            roi=[conf["width"],conf["height"],conf["left"],conf["top"]]
            self.setRoi(roi)
        else:
            self.roi_w = w
            self.roi_h = h
        

        self.imgdata = (ctypes.c_uint8 * self.roi_w.value * self.roi_h.value)()
        checkError(self.qhyccd.BeginQHYCCDLive(self.cam),
                            "BeginQHYCCDLive")

        return

    def setRoi(self, roi):
        super().setRoi(roi)
        self.roi_w = ctypes.c_uint32(roi[0])
        self.roi_h = ctypes.c_uint32(roi[1])
        roi_x = ctypes.c_uint32(roi[2])
        roi_y = ctypes.c_uint32(roi[3])
        self.qhyccd.SetQHYCCDResolution(self.cam, 
                                   roi_x, 
                                   roi_y, 
                                   self.roi_w, 
                                   self.roi_h)
        
        return
    
    def setExposure(self, exposure):
        super().setExposure(exposure)
        checkError(self.qhyccd.SetQHYCCDParam(self.cam, 
                                         CONTROL_ID.CONTROL_EXPOSURE, 
                                         ctypes.c_double(exposure)),
                                        "SetQHYCCDParam CONTROL EXPOSURE")# unit: us

        return
    
    def setBinning(self, binning):
        super().setBinning(binning)
        if binning == 1:
            checkError(self.qhyccd.SetQHYCCDBinMode(self.cam, 
                                            ctypes.c_uint32(1),
                                            ctypes.c_uint32(1)),
                                            "SetQHYCCDBinMode")
        return
    
    def setGain(self, gain):
        super().setGain(gain)
        checkError(self.qhyccd.SetQHYCCDParam(self.cam, 
                              CONTROL_ID.CONTROL_GAIN, 
                              ctypes.c_double(gain)),
                            "SetQHYCCDParam CONTROL GAIN")
        return
    
    def setBitDepth(self, bitDepth):
        super().setBitDepth(bitDepth)

        return

    def setUSBTraffic(self, usbtraffic):
        checkError(self.qhyccd.SetQHYCCDParam(self.cam, 
                                         CONTROL_ID.CONTROL_USBTRAFFIC, 
                                         ctypes.c_double(usbtraffic)),
                                        "SetQHYCCDParam USB TRAFFIC") 

    def expose(self):
        ret = self.qhyccd.GetQHYCCDLiveFrame(self.cam, 
                                    ctypes.byref(self.roi_w), 
                                    ctypes.byref(self.roi_h), 
                                    ctypes.byref(self.bpp), 
                                    ctypes.byref(self.channels), 
                                    self.imgdata)
        if ret == 0:
            self.data = np.asarray(self.imgdata)#np.ndarray(self.imageShape, 
                                #buffer=self.imgdata, 
                               # dtype=self.dtype)
            mask = self.data>self.noiseThrehold
            if np.sum(mask) == 0:
                val = np.array([0])
            else:
                val = np.mean(self.data[mask])
            self.powerShm.write(val.astype(np.float64).reshape((1,)))
            super().expose()

        return
    
    def readLong(self):
        x = self.psfLong.read()
        mask = x>self.noiseThrehold
        if np.sum(mask) == 0:
            val = 0*np.mean(self.data)
        else:
            val = np.mean(self.data[mask])
        return val

    def __del__(self):
        super().__del__()
        checkError(self.qhyccd.StopQHYCCDLive(self.cam), "StopQHYCCDLive")
        
        return

if __name__ == "__main__":

    launchComponent(QHYCCD, "psf", start = True)
        
