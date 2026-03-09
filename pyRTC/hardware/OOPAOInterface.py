"""Bridge between pyRTC components and an OOPAO optical simulation.

This module adapts the OOPAO telescope, atmosphere, deformable-mirror, pyramid
sensor, and PSF-camera objects into the pyRTC component interfaces. It is used
for simulation-backed development and validation where the control stack should
behave as if it were driving real hardware.
"""

import argparse
import os
import time

import numpy as np

from pyRTC.logging_utils import get_logger
from pyRTC.Pipeline import Listener
from pyRTC.ScienceCamera import ScienceCamera
from pyRTC.WavefrontCorrector import WavefrontCorrector
from pyRTC.WavefrontSensor import WavefrontSensor
from pyRTC.utils import decrease_nice, read_yaml_file, set_affinity

from OOPAO.Atmosphere import Atmosphere
from OOPAO.DeformableMirror import DeformableMirror
from OOPAO.Pyramid import Pyramid
from OOPAO.Source import Source
from OOPAO.Telescope import Telescope


logger = get_logger(__name__)

class _OOPAOWFSensor(WavefrontSensor):
    """Wavefront-sensor wrapper around an OOPAO pyramid sensor.

    The wrapper advances the simulated atmosphere when required, propagates the
    guide star through the telescope and deformable mirror, and exposes the
    resulting detector frame through the standard pyRTC ``WavefrontSensor`` API.
    """

    def __init__(self, wfsConf, tel, ngs, atm, dm, wfs) -> None:
        
        self.tel = tel
        self.ngs = ngs
        self.atm = atm
        self.dm = dm
        self.wfs = wfs
        
        super().__init__(wfsConf)

    def _propagate_source(self):
        if self.tel.isPaired:
            # Atmosphere.update() rebuilds the source/telescope state from
            # scratch, so only the DM and WFS need to be applied afterwards.
            self.atm.update()
            self.ngs * self.dm * self.wfs
            return

        # Without atmosphere, OOPAO does not reset the source state for us.
        # Rebuilding the source/telescope path each frame prevents the DM OPD
        # from accumulating across repeated exposures of a static command.
        self.ngs ** self.tel
        self.ngs * self.dm * self.wfs
        
    def expose(self):
        self._propagate_source()

        #Generate a new exposure
        self.data = self.wfs.cam.frame.astype(np.uint16)
        super().expose()

        return

    def addAtmosphere(self):
        self.tel+self.atm

    def removeAtmosphere(self):
        self.tel-self.atm

class _OOPAOWFCorrector(WavefrontCorrector):
    """Wavefront-corrector wrapper for an OOPAO deformable mirror.

    This adapter maps pyRTC command vectors onto the OOPAO deformable-mirror
    coefficient array so the simulated optical train responds to control-loop
    updates exactly where a physical mirror would in a deployed system.
    """

    def __init__(self, correctorConf, tel, dm) -> None:
    
        self.tel = tel
        self.dm = dm
        
        self.dm.coefs = 0
        super().__init__(correctorConf)

        #Set-up additional pyRTC parameters from simulation
        numActuators = self.dm.validAct.size
        self.setLayout(self.dm.validAct.reshape(int(np.sqrt(numActuators)),
                                                            int(np.sqrt(numActuators))))

    def readM2C(self, filename=''):
        self.setM2C(None)
    
    def sendToHardware(self):
        
        super().sendToHardware()

        self.dm.coefs = self.currentShape.astype(np.float64)

    def setFlat(self, flat):
        super().setFlat(flat)
        self.dm.flat = flat 


class _OOPAOScienceCamera(ScienceCamera):
    """Science-camera wrapper around the OOPAO PSF path.

    The class reuses the current atmosphere and deformable-mirror state to
    synthesize a PSF image that can be consumed by pyRTC exactly like a hardware
    science camera. It is intentionally simulation-facing and does not attempt
    to hide OOPAO-specific PSF generation details.
    """

    def __init__(self, scienceConf, tel, src, atm, dm) -> None:
        self.tel = tel
        self.src = src
        self.atm = atm
        self.dm = dm
        self._atmosphere_enabled = False
        super().__init__(scienceConf)
        self._reference_psf = self._render_reference_psf()
        self._reference_peak = float(np.max(self._reference_psf)) if self._reference_psf.size else 1.0
        if self._reference_peak <= 0:
            self._reference_peak = 1.0
        self.setModelPSF(self._scale_psf_to_detector(self._reference_psf).astype(self.psfLongDtype))

    def _compute_psf(self, opd_no_pupil):
        self.src ** self.tel
        self.src.OPD_no_pupil = opd_no_pupil
        self.tel.computePSF(zeroPaddingFactor=5)
        return np.array(self.tel.PSF, dtype=np.float64, copy=True)

    def _render_reference_psf(self):
        zero_opd = np.zeros(self.tel.pupil.shape, dtype=np.float64)
        reference = self._compute_psf(zero_opd)
        return np.nan_to_num(reference, nan=0.0, posinf=0.0, neginf=0.0)

    def _scale_psf_to_detector(self, psf):
        psf = np.nan_to_num(psf, nan=0.0, posinf=0.0, neginf=0.0)
        if psf.size == 0:
            return np.zeros(self.imageShape, dtype=np.float64)

        scaled = psf / self._reference_peak
        scaled *= np.iinfo(self.imageRawDType).max
        return np.clip(scaled, 0, np.iinfo(self.imageRawDType).max)

    def _current_opd_no_pupil(self):
        base_opd = np.zeros(self.tel.pupil.shape, dtype=np.float64)
        if self._atmosphere_enabled:
            if getattr(self.atm, "OPD_no_pupil", None) is not None:
                base_opd = np.array(self.atm.OPD_no_pupil, dtype=np.float64, copy=True)
            elif getattr(self.atm, "OPD", None) is not None:
                base_opd = np.array(self.atm.OPD, dtype=np.float64, copy=True)

        dm_opd = getattr(self.dm, "OPD", None)
        if dm_opd is None:
            return base_opd

        dm_opd = np.asarray(dm_opd)
        if dm_opd.ndim == 2:
            return base_opd + dm_opd.astype(np.float64, copy=False)

        # Interaction-matrix calibration can temporarily drive the DM with a cube.
        # The science camera only renders one frame at a time, so keep the most
        # recent 2D command if the PSF path is left running during calibration.
        return base_opd + dm_opd[..., -1].astype(np.float64, copy=False)

    def _render_psf_frame(self):
        psf = self._compute_psf(self._current_opd_no_pupil())
        return self._scale_psf_to_detector(psf).astype(self.imageRawDType)
        
    def expose(self):
        self.data = self._render_psf_frame()
        
        super().expose()

        return

    def integrate(self):
        super().integrate()
        if np.max(self.model) > 0:
            self.computeStrehl(median_filter_size=1, gaussian_sigma=0)
        return
    
    def addAtmosphere(self):
        self._atmosphere_enabled = True

    def removeAtmosphere(self):
        self._atmosphere_enabled = False

class OOPAOInterface():
    """Assembles a complete pyRTC-compatible OOPAO simulation stack.

    ``OOPAOInterface`` creates the simulated telescope, atmosphere, guide star,
    deformable mirror, pyramid sensor, and science camera, then wraps the key
    pieces in pyRTC component adapters. The resulting objects can be launched or
    driven through the same orchestration code used for physical hardware,
    making the class useful for algorithm development, documentation examples,
    and end-to-end synthetic tests.
    """

    def __init__(self, conf, param=None) -> None:

        if param is None:
            self.param = _initializeDummyParameterFile()
        else:
            self.param = param

        wfsConf = conf["wfs"]
        correctorConf = conf["wfc"]
        scienceConf = conf["psf"]

        if param is None:
            param = self.param
        else:
            self.param = param

        #Create our Telescope Simulatation
        self.tel = Telescope(resolution     = param['resolution'],
                        diameter            = param['diameter'],
                        samplingTime        = param['samplingTime'],
                        centralObstruction  = param['centralObstruction'])
        
        #A second copy of the telescope so that the PSF camera is fighting with the
        #Wavefront Sensor
        self.tel_psf = Telescope(resolution     = param['resolution'],
                diameter            = param['diameter'],
                samplingTime        = param['samplingTime'],
                centralObstruction  = param['centralObstruction'])
        
        self.src = Source(optBand   = param["sourceBand"], 
                          magnitude = param['magnitude'])
        self.src*self.tel_psf

        #Create a guide star
        self.ngs = Source(optBand = param['opticalBand'], magnitude = param['magnitude'])
        self.ngs*self.tel

        self.atm = Atmosphere(telescope     = self.tel,\
               r0            = param['r0'],\
               L0            = param['L0'],\
               windSpeed     = param['windSpeed'],\
               fractionalR0  = param['fractionnalR0'],\
               windDirection = param['windDirection'],\
               altitude      = param['altitude'])

        self.dm=DeformableMirror(telescope          = self.tel,
                                        nSubap         = param['nSubaperture'], 
                                        mechCoupling   = param['mechanicalCoupling'])

        # create the Pyramid WFS Object
        self.wfs = Pyramid(nSubap         = param['nSubaperture'],
                    telescope             = self.tel,
                    modulation            = param['modulation'],
                    lightRatio            = param['lightThreshold'],
                    n_pix_separation      = param['n_pix_separation'],
                    psfCentering          = param['psfCentering'],
                    postProcessing        = param['postProcessing'])
        
        #Initialize the atmosphere
        self.atm.initializeAtmosphere(self.tel)

        self.wfsInterface = _OOPAOWFSensor(wfsConf, self.tel, self.ngs, self.atm, self.dm, self.wfs)
        self.dmInterface  = _OOPAOWFCorrector(correctorConf, self.tel, self.dm)
        self.psfInterface = _OOPAOScienceCamera(scienceConf, self.tel_psf, self.src, self.atm, self.dm)

        #Add the atmosphere to the system
        self.addAtmosphere()

    def addAtmosphere(self):
        self.psfInterface.addAtmosphere()
        self.wfsInterface.addAtmosphere()

    def removeAtmosphere(self):
        self.psfInterface.removeAtmosphere()
        self.wfsInterface.removeAtmosphere()

    def restartSimulation(self):
        del self.wfsInterface
        del self.dmInterface
        del self.psfInterface

        self.__init__(param=self.param)

    def get_hardware(self):
        return self.wfsInterface, self.dmInterface, self.psfInterface


def _initializeDummyParameterFile():
    """Return a small default OOPAO parameter dictionary for local simulation."""

    from OOPAO.tools.tools import createFolder

    # initialize the dictionaries
    param = dict()
    
    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ATMOSPHERE PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    param['r0'                   ] = 0.3                                            # value of r0 in the visible in [m]
    param['L0'                   ] = 30                                             # value of L0 in the visible in [m]
    param['fractionnalR0'        ] = [0.45,0.1,0.1,0.25,0.1]                        # Cn2 profile
    param['windSpeed'            ] = [10,12,11,15,20]                               # wind speed of the different layers in [m.s-1]
    param['windDirection'        ] = [0,72,144,216,288]                             # wind direction of the different layers in [degrees]
    param['altitude'             ] = [0, 1000,5000,10000,12000 ]                    # altitude of the different layers in [m]
                    
    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% M1 PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    param['diameter'             ] = 8                                              # diameter in [m]
    param['nSubaperture'         ] = 10                                             # number of PWFS subaperture along the telescope diameter
    param['nPixelPerSubap'       ] = 4                                              # sampling of the PWFS subapertures
    param['resolution'           ] = param['nSubaperture']*param['nPixelPerSubap']  # resolution of the telescope driven by the PWFS
    param['sizeSubaperture'      ] = param['diameter']/param['nSubaperture']        # size of a sub-aperture projected in the M1 space
    param['samplingTime'         ] = 1/1000                                         # loop sampling time in [s]
    param['centralObstruction'   ] = 0.112                                          # central obstruction in percentage of the diameter
    param['nMissingSegments'     ] = 0                                              # number of missing segments on the M1 pupil
    param['m1_reflectivity'      ] = 1                                              # reflectivity of the 798 segments
          
    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% NGS PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    param['magnitude'            ] = 8                                              # magnitude of the guide star
    param['opticalBand'          ] = 'I'                                            # optical band of the guide star
    param['sourceBand'          ] = 'K'

    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% DM PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    param['nActuator'            ] = param['nSubaperture']+1                        # number of actuators 
    param['mechanicalCoupling'   ] = 0.45
    param['isM4'                 ] = False                                          # tag for the deformable mirror class
    param['dm_coordinates'       ] = None                                           # tag for the eformable mirror class
    
    # mis-registrations                                                             
    param['shiftX'               ] = 0                                              # shift X of the DM in pixel size units ( tel.D/tel.resolution ) 
    param['shiftY'               ] = 0                                              # shift Y of the DM in pixel size units ( tel.D/tel.resolution )
    param['rotationAngle'        ] = 0                                              # rotation angle of the DM in [degrees]
    param['anamorphosisAngle'    ] = 0                                              # anamorphosis angle of the DM in [degrees]
    param['radialScaling'        ] = 0                                              # radial scaling in percentage of diameter
    param['tangentialScaling'    ] = 0                                              # tangential scaling in percentage of diameter
    
    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% WFS PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    param['modulation'            ] = 5                                             # modulation radius in ratio of wavelength over telescope diameter
    param['n_pix_separation'      ] = 4                                             # separation ratio between the PWFS pupils
    param['psfCentering'          ] = False                                         # centering of the FFT and of the PWFS mask on the 4 central pixels
    param['lightThreshold'        ] = 0.1                                           # light threshold to select the valid pixels
    param['postProcessing'        ] = 'slopesMaps'                                  # post-processing of the PWFS signals 
    
    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% OUTPUT DATA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # name of the system
    param['name'] = 'VLT_' +  param['opticalBand'] +'_band_'+ str(param['nSubaperture'])+'x'+ str(param['nSubaperture'])  
    
    # location of the calibration data
    param['pathInput'            ] = 'data_calibration/' 
    
    # location of the output data
    param['pathOutput'            ] = 'data_cl/'
    

    logger.info('Reading/Writting calibration data from %s', param['pathInput'])
    logger.info('Writting output data in %s', param['pathOutput'])

    createFolder(param['pathOutput'])
    
    return param

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

    sim = OOPAOInterface(conf=conf)
    
    listener = Listener(sim, port= int(args.port))
    while listener.running:
        listener.listen()
        time.sleep(1e-3)