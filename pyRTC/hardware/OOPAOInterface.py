from pyRTC.WavefrontCorrector import *
from pyRTC.WavefrontSensor import *
from pyRTC.SlopesProcess import *
from pyRTC.ScienceCamera import *
from pyRTC.Pipeline import *
from pyRTC.utils import *

from OOPAO.Atmosphere import Atmosphere
from OOPAO.DeformableMirror import DeformableMirror
from OOPAO.MisRegistration import MisRegistration
from OOPAO.Pyramid import Pyramid
from OOPAO.Source import Source
from OOPAO.Telescope import Telescope
from OOPAO.calibration.ao_calibration import ao_calibration
from OOPAO.calibration.compute_KL_modal_basis import compute_M2C
from OOPAO.tools.displayTools import displayMap

class _OOPAOWFSensor(WavefrontSensor):

    def __init__(self, wfsConf, param) -> None:
        
        self.param = param
        self.wfs = None

        super().__init__(wfsConf)
        
    def expose(self):
        
        self.data = self.wfs.cam.frame.astype(np.uint16)
        super().expose()

        return

class _OOPAOWFCorrector(WavefrontCorrector):

    def __init__(self, correctorConf, param) -> None:
    
        self.param = param
        self.dm = None
    
        super().__init__(correctorConf)

    def readM2C(self, filename=''):
        self.setM2C(None)
    
    def sendToHardware(self, flagInd=0):
        self.M2C = np.float32(self.M2C)
        super().sendToHardware(flagInd)

        self.dm.coefs = self.currentShape.astype(np.float64)

    def setFlat(self, flat):
        super().setFlat(flat)
        self.dm.flat = flat 
        
        
class _OOPAOSlopesProcess(SlopesProcess):

    def __init__(self, conf, param) -> None:
    
        self.param = param
        self.wfs = None
        self.atm = None
        self.ngs = None
        self.tel = None
        self.dm = None
        self.src = None
    
        super().__init__(conf)
        
        self.total_OPD = [None]
        self.residual_OPD = [None]
        self.strehl_history = [None]
        
    def passOOPAOVars(self, wfsIn, atmIn, ngsIn, telIn, dmIn, srcIn):
        self.wfs = wfsIn
        self.atm = atmIn
        self.ngs = ngsIn
        self.tel = telIn
        self.dm = dmIn
        self.src = srcIn
    
    def computeSignal(self):
        
        self.total_OPD.append(np.std(self.tel.OPD[np.where(self.tel.pupil>0)])*1e9)
        
        super().computeSignal()
            
        self.atm.update()
        self.ngs*self.tel*self.dm*self.wfs
        self.src*self.tel
        
        self.residual_OPD.append(np.std(self.tel.OPD[np.where(self.tel.pupil>0)])*1e9)
        self.strehl_history.append(np.exp(-np.var(self.tel.src.phase[np.where(self.tel.pupil==1)])))
        
    def reset(self):
        self.total_OPD = [None]
        self.residual_OPD = [None]
        self.strehl_history = [None]
        

class _OOPAOScienceCamera(ScienceCamera):

    def __init__(self, scienceConf, param) -> None:
    
        self.param = param
        self.tel = None
        
        super().__init__(scienceConf)
        
    def expose(self):
        
        self.tel.computePSF(zeroPaddingFactor=5, N_crop=136)
        self.data = (255.*self.tel.PSF_norma_zoom).astype(np.uint16)
        
        # if self.data is None:
        #     self.data = np.ones_like(self.dark).astype(np.uint16)
        # else:
        #     self.data = self.data.astype(np.uint16)

        super().expose()

        return

class OOPAOInterface():

    def __init__(self, conf, param=None) -> None:

        if param is None:
            self.param = _initializeDummyParameterFile()
        else:
            self.param = param

        wfsConf = conf["wfs"]
        correctorConf = conf["wfc"]
        scienceConf = conf["psf"]

        self.tel = None
        self.dm = None
        self.wfs = None
        self.science = None
        self.ngs = None
        self.atm = None
        self.stepCounter = 0
        self.ao_calib = None
        self.M2C = None

        self.wfsInterface = _OOPAOWFSensor(wfsConf, param)
        self.dmInterface  = _OOPAOWFCorrector(correctorConf, param)
        self.psfInterface = _OOPAOScienceCamera(scienceConf, param)
        self.slopesInterface = _OOPAOSlopesProcess(conf, param)
        
        self.initializeSimulation()

    def addAtmosphere(self):
        self.tel+self.atm

    def removeAtmosphere(self):
        self.tel-self.atm

    def initializeSimulation(self, param=None):
        
        if param is None:
            param = self.param
        else:
            self.param = param

        self.tel = Telescope(resolution     = param['resolution'],
                        diameter            = param['diameter'],
                        samplingTime        = param['samplingTime'],
                        centralObstruction  = param['centralObstruction'])
       
        self.ngs = Source(optBand = param['opticalBand'], magnitude = param['magnitude'])
        self.ngs*self.tel

        self.atm = Atmosphere(telescope     = self.tel,\
               r0            = param['r0'],\
               L0            = param['L0'],\
               windSpeed     = param['windSpeed'],\
               fractionalR0  = param['fractionnalR0'],\
               windDirection = param['windDirection'],\
               altitude      = param['altitude'])
        
        self.atm.initializeAtmosphere(self.tel)
        self.tel+self.atm
        self.dm=DeformableMirror(telescope          = self.tel,
                                     nSubap         = param['nSubaperture'], 
                                     mechCoupling   = param['mechanicalCoupling'])

        # make sure tel and atm are separated to initialize the PWFS
        self.tel-self.atm

        # create the Pyramid Object
        self.wfs = Pyramid(nSubap         = param['nSubaperture'],
                    telescope             = self.tel,
                    modulation            = param['modulation'],
                    lightRatio            = param['lightThreshold'],
                    n_pix_separation      = param['n_pix_separation'],
                    psfCentering          = param['psfCentering'],
                    postProcessing        = param['postProcessing'])
        
        # generate M2C
        self.M2C =  compute_M2C(param             = param,
                                telescope         = self.tel,
                                atmosphere        = self.atm,
                                deformableMirror  = self.dm,
                                nameFolder        = None,
                                nameFile          = None,
                                remove_piston     = True)
        
        # calibrate to get IM
        # self.ao_calib =  ao_calibration(param            = param,
        #                                 ngs              = self.ngs,
        #                                 tel              = self.tel,
        #                                 atm              = self.atm,
        #                                 dm               = self.dm,
        #                                 wfs              = self.wfs,
        #                                 nameFolderIntMat = None,
        #                                 nameIntMat       = None,
        #                                 nameFolderBasis  = None,
        #                                 nameBasis        = None,
        #                                 nMeasurements    = 100)
        
        self.src = Source(optBand   = 'K', magnitude = param['magnitude'])
        self.src*self.tel
        
        # combine telescope with atmosphere
        self.tel+self.atm

        # initialize DM commands
        self.dm.coefs=0
        self.ngs*self.tel*self.dm*self.wfs

        self.dmInterface.dm = self.dm
        self.wfsInterface.wfs = self.wfs
        self.psfInterface.tel = self.tel
        
        self.slopesInterface.passOOPAOVars(self.wfs, self.atm, self.ngs, self.tel, self.dm, self.src)
        
        return self.dmInterface, self.wfsInterface, self.psfInterface, self.slopesInterface

    def restartSimulation(self):
        # reset atmosphere
        self.atm.initializeAtmosphere(self.tel)
        self.tel+self.atm

        # reset DM
        self.dm.coefs=0
        self.ngs*self.tel*self.dm*self.wfs

        # reset count
        self.stepCounter = 0
        
        self.slopesInterface.reset()


def _initializeDummyParameterFile():
    from OOPAO.tools.tools import createFolder

    # initialize the dictionaries
    param = dict()
    
    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ATMOSPHERE PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    param['r0'                   ] = 0.3                                            # value of r0 in the visibile in [m]
    param['L0'                   ] = 30                                             # value of L0 in the visibile in [m]
    param['fractionnalR0'        ] = [0.45,0.1,0.1,0.25,0.1]                        # Cn2 profile
    param['windSpeed'            ] = [10,12,11,15,20]                               # wind speed of the different layers in [m.s-1]
    param['windDirection'        ] = [0,72,144,216,288]                             # wind direction of the different layers in [degrees]
    param['altitude'             ] = [0, 1000,5000,10000,12000 ]                    # altitude of the different layers in [m]
                    
    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% M1 PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    param['diameter'             ] = 8                                              # diameter in [m]
    param['nSubaperture'         ] = 20                                             # number of PWFS subaperture along the telescope diameter
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
    
    
    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% LOOP PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    param['nLoop'                 ] = 5000                                          # number of iteration                             
    param['photonNoise'           ] = True                                          # Photon Noise enable  
    param['readoutNoise'          ] = 0                                             # Readout Noise value
    param['gainCL'                ] = 0.5                                           # integrator gain
    param['nModes'                ] = 300                                           # number of KL modes controlled 
    param['nPhotonPerSubaperture' ] = 1000                                          # number of photons per subaperture (update of ngs.magnitude)
    param['getProjector'          ] = True                                          # modal projector too get modal coefficients of the turbulence and residual phase

    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% OUTPUT DATA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # name of the system
    param['name'] = 'VLT_' +  param['opticalBand'] +'_band_'+ str(param['nSubaperture'])+'x'+ str(param['nSubaperture'])  
    
    # location of the calibration data
    param['pathInput'            ] = 'data_calibration/' 
    
    # location of the output data
    param['pathOutput'            ] = 'data_cl/'
    

    print('Reading/Writting calibration data from ' + param['pathInput'])
    print('Writting output data in ' + param['pathOutput'])

    createFolder(param['pathOutput'])
    
    return param