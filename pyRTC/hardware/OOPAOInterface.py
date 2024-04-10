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
from OOPAO.ShackHartmann import ShackHartmann
from OOPAO.Source import Source
from OOPAO.Telescope import Telescope
from OOPAO.calibration.ao_calibration import ao_calibration
from OOPAO.calibration.compute_KL_modal_basis import compute_M2C
from OOPAO.tools.displayTools import displayMap

class _OOPAOWFSensor(WavefrontSensor):

    def __init__(self, wfsConf, tel, ngs, atm, dm, wfs) -> None:
        
        self.tel = tel
        self.ngs = ngs
        self.atm = atm
        self.dm = dm
        self.wfs = wfs
        
        super().__init__(wfsConf)
        
    def expose(self):

        #Advance the atmosphere
        if self.tel.isPaired:
            self.atm.update()

        #Propagate the ngs to the wfs and apply the DM state
        self.ngs*self.tel*self.dm*self.wfs

        #Generate a new exposure
        self.data = self.wfs.cam.frame.astype(np.uint16)
        super().expose()

        return

    def addAtmosphere(self):
        self.tel+self.atm

    def removeAtmosphere(self):
        self.tel-self.atm

class _OOPAOWFCorrector(WavefrontCorrector):

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

    def __init__(self, scienceConf, tel, src, atm, dm) -> None:
        self.tel = tel
        self.src = src
        self.atm = atm
        self.dm = dm
        self.old_opd = tel.OPD.copy()
        super().__init__(scienceConf)
        
    # def expose(self):

    #     #We need to manually add the current atmosphere OPD to the telescope
    #     new_atm = not (self.atm.OPD == self.old_opd).all()
    #     if self.tel.isPaired and new_atm :
    #         self.old_opd = self.atm.OPD.copy()
    #     elif not self.tel.isPaired: 
    #         self.old_opd = np.zeros_like(self.old_opd)
        
    #     if self.dm.OPD.shape == self.old_opd.shape:
    #         self.tel.OPD = self.old_opd + self.dm.OPD
        
    #     # #Add current dm state to the telescope
    #     self.src*self.tel#*self.dm
    #     #Compute PSF
    #     self.tel.computePSF(zeroPaddingFactor=5)
    #     #Check that we still have the right source coupled
    #     self.data = (255.*self.tel.PSF_norma_zoom).astype(np.uint16)

    #     super().expose()

    #     return
    

    def expose(self):

        #We need to manually add the current atmosphere OPD to the telescope
        new_atm = not (self.atm.OPD == self.old_opd).all()
        if self.tel.isPaired and new_atm :
            self.old_opd = self.atm.OPD.copy()
        elif not self.tel.isPaired: 
            self.old_opd = np.zeros_like(self.old_opd)
        
        if self.dm.OPD.shape == self.old_opd.shape:
            self.tel.OPD = self.old_opd + self.dm.OPD
        
        # #Add current dm state to the telescope
        self.src*self.tel#*self.dm
        #Compute PSF
        self.tel.computePSF(zeroPaddingFactor=5, N_crop=None)
        #Check that we still have the right source coupled
        self.data = (255.*self.tel.PSF_norma).astype(np.uint16)
        
        super().expose()

        return
    
    def addAtmosphere(self):
        self.tel+self.atm

    def removeAtmosphere(self):
        self.tel-self.atm

class OOPAOInterface():

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

        # print('RESOLUTION', param['resolution'])
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

        if conf["slopes"]["type"].lower() == "pywfs":

            # create the Pyramid WFS Object
            self.wfs = Pyramid(nSubap         = param['nSubaperture'],
                        telescope             = self.tel,
                        modulation            = param['modulation'],
                        lightRatio            = param['lightThreshold'],
                        n_pix_separation      = param['n_pix_separation'],
                        psfCentering          = param['psfCentering'],
                        postProcessing        = param['postProcessing'])
            
        else:
            self.wfs = ShackHartmann(nSubap          = param['nSubaperture'],        # number of subaperture
                        telescope             = self.tel,                  # telescope object
                        lightRatio            = 0.5,                  # flux threshold to select valid sub-subaperture
                        binning_factor        = 1,                    # binning factor
                        is_geometric          = False,                # Flag to use a geometric shack-hartmann (direct gradient measurement)
                        shannon_sampling      = True)                 # Flag to use a shannon sampling for the shack-hartmann spots

        
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
    

    print('Reading/Writting calibration data from ' + param['pathInput'])
    print('Writting output data in ' + param['pathOutput'])

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
    
    l = Listener(sim, port= int(args.port))
    while l.running:
        l.listen()
        time.sleep(1e-3)