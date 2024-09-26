from pyRTC import WavefrontSensor
import numpy as np
from pyRTC.Pipeline import *

conf = {"name": "test", "width": 100, "functions": ["expose"]}

def test_init_wfs():
    
    wfs = WavefrontSensor(conf)

    assert(wfs.width == 100)
    assert(wfs.height == 1)
    assert(wfs.name == "test")
    assert((wfs.dark == np.zeros((100,1)).astype(np.int32)).all())
    assert(type(wfs.image) == ImageSHM)
    assert(type(wfs.imageRaw) == ImageSHM)
    assert(type(wfs.darkFile) == str )

    return

def test_cam_properties():
    
    wfs = WavefrontSensor(conf)
    wfs.setRoi([1,2,3,4])
    assert(wfs.roiWidth == 1)
    assert(wfs.roiHeight == 2)
    assert(wfs.roiLeft == 3)
    assert(wfs.roiTop == 4)

    wfs.setExposure(5.2)
    assert(wfs.exposure == 5.2)

    wfs.setBinning(2)
    assert(wfs.binning == 2)

    wfs.setGain(3.1)
    assert(wfs.gain == 3.1)

    return

def test_expose():
    #initialize and turn on WFS
    wfs = WavefrontSensor(conf)
    wfs.start()

    #Read an image, should be zeros
    img = wfs.read()
    assert(np.sum(img) == 0)
    assert(img.dtype == np.int32)
    
    #Set a dark of all ones
    wfs.setDark(np.ones_like(wfs.dark))
    #Read a new image
    img = wfs.read()
    img = wfs.read()
    #Should be all negative one
    assert((img == -1).all())
    assert(img.dtype == np.int32)

    #Take a new dark
    wfs.darkCount = 1
    wfs.takeDark()
    img = wfs.read()
    img = wfs.read()
    #Should be all 0s now
    assert((img == 0).all())
    assert(img.dtype == np.int32)
    
    #Save dark to file
    wfs.setDark(2*np.ones_like(wfs.dark))
    wfs.saveDark("dark_test.npy")
    #Load it from file
    wfs.loadDark("dark_test.npy")
    #read a new image
    img = wfs.read()
    img = wfs.read()
    #Should be all -2s now
    assert((img == -2).all())
    assert(img.dtype == np.int32)
    #Stop WFS
    wfs.stop()


    return


