from pyRTC.utils import *
from pyRTC.Pipeline import *
from pyRTC.Optimizer import *

import numpy as np

class NCPAOptimizer(Optimizer):

    def __init__(self, conf, loop, slopes) -> None:
        
        self.loop = loop
        self.slopes = slopes
        self.wfcShm, self.wfcDims, self.wfcDtype = initExistingShm("wfc")
        self.strehlShm, _, _ = initExistingShm("strehl")
        self.startMode = setFromConfig(conf, "startMode", 0)
        self.endMode = setFromConfig(conf, "endMode", 20)
        self.correctionMag = setFromConfig(conf, "correctionMag", 2e-3)
        self.numReads = setFromConfig(conf, "numReads", 5)
        self.isCL = False
        self.origRefSlopes = None
        self.validSubAps = None
        self.IM = None

        super().__init__(conf)

    def objective(self, trial):

        self.applyTrial(trial)

        result = np.empty(self.numReads)
        for i in range(self.numReads):
            result[i] = self.strehlShm.read()
        return np.mean(result)
    
    def applyTrial(self, trial):
        modalCoefs = np.zeros(self.wfcDims, dtype=self.wfcDtype)
        for i in range(self.startMode,self.endMode):
            modalCoefs[i] = np.float32(trial.suggest_float(f'{i}', 
                                                           -self.correctionMag,
                                                            self.correctionMag))
        #In CL, we need to update reference slopes
        if self.isCL:
            #Compute the slope adjustment
            refSlopesAdjust = np.zeros_like(self.origRefSlopes)
            refSlopesAdjust[self.validSubAps] = self.IM@modalCoefs
            #Adjust reference slopes
            refSlopes = self.origRefSlopes + refSlopesAdjust
            #save to file so process can load
            np.save(self.newRefSlopesFile, refSlopes)
            #Load new reference slopes
            self.slopes.setProperty("refSlopesFile", self.newRefSlopesFile)
            self.slopes.run("loadRefSlopes")
            #Reset where we get our reference slopes from. This is to ensure
            #that for subsequent optimizations we have the same initial starting point
            self.slopes.setProperty("refSlopesFile", self.refSlopesFile)
        # In OL, just write directly to mirror shape
        else:
            self.wfcShm.write(modalCoefs)
        return super().applyTrial(trial)

    def applyOptimum(self, overwrite=False):
        super().applyOptimum()
        modalCoefs = np.zeros(self.wfcDims, dtype=self.wfcDtype)
        for k in self.study.best_params.keys():
            modalCoefs[int(k)] = self.study.best_params[k]

        if self.isCL:
            #Compute the slope adjustment
            refSlopesAdjust = np.zeros_like(self.origRefSlopes)
            refSlopesAdjust[self.validSubAps] = self.IM@modalCoefs
            #Adjust reference slopes
            refSlopes = self.origRefSlopes + refSlopesAdjust
            if overwrite:
                #Overwrite original ref Slopes
                np.save(self.refSlopesFile, refSlopes)
                self.slopes.setProperty("refSlopesFile", self.refSlopesFile)
            else:
                #save to tmp file so process can load
                np.save(self.newRefSlopesFile, refSlopes)
                #Load new reference slopes
                self.slopes.setProperty("refSlopesFile", self.newRefSlopesFile)

            self.slopes.run("loadRefSlopes")
            #Reset where we get our reference slopes from. This is to ensure
            #that for subsequent optimizations we have the same initial starting point
            self.slopes.setProperty("refSlopesFile", self.refSlopesFile)
        # In OL, just write directly to mirror shape
        else:
            self.wfcShm.write(modalCoefs)

        return 
    
    def optimize(self):
        #Load the original reference slopes
        self.refSlopesFile = self.slopes.getProperty("refSlopesFile")
        #Are we in closed loop or open loop
        self.isCL = self.loop.getProperty("running")
        if self.isCL:
            #Load current valid sub aperture map
            self.validSubAps = np.load(self.slopes.getProperty("validSubApsFile"))
            #Compute current IM
            self.IM =np.load(self.loop.getProperty("IMFile"))

            self.origRefSlopes = np.load(self.refSlopesFile)
            #Set-up the slopes to read from new slopes file on load
            self.newRefSlopesFile = get_tmp_filepath(self.refSlopesFile)
    
        #Run the maximization
        super().optimize()

        #Reset the slope reference (for subsequent optimizations)
        self.slopes.setProperty("refSlopesFile", self.refSlopesFile)
        return 

if __name__ == "__main__":

    #Prevents camera output from messing with communication
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')

    # Create argument parser
    parser = argparse.ArgumentParser(description="Read a config file from the command line.")

    # Add command-line argument for the config file
    parser.add_argument("-c", "--config", required=True, help="Path to the config file")
    parser.add_argument("-p", "--port", required=True, help="Port for communication")

    # Parse command-line arguments
    args = parser.parse_args()

    conf = read_yaml_file(args.config)["optimizer"]

    pid = os.getpid()
    set_affinity((conf["affinity"])%os.cpu_count()) 
    decrease_nice(pid)

    component = NCPAOptimizer(conf=conf)
    component.start()

    # Go back to communicating with the main program through stdout
    sys.stdout = original_stdout

    l = Listener(component, port = int(args.port))
    while l.running:
        l.listen()
        time.sleep(1e-3)