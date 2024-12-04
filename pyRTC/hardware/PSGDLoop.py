import os 
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ['NUMBA_NUM_THREADS'] = '1'

from pyRTC.Loop import *
from pyRTC.Pipeline import *
from pyRTC.utils import *

class PSGDLoop(Loop):

    def __init__(self, conf) -> None:

        signal = ImageSHM("signal", (1,), np.uint8)
        signal = ImageSHM("signal2D", (1,), np.uint8)
        #Initialize the pyRTC super class
        super().__init__(conf)

        self.psfShm, psfShape, psfDType = initExistingShm("power")
        self.psfLongShm, psfLongShape, psfLongDType = initExistingShm("psfLong")
        self.wfcShm, self.wfcShape, self.wfcDType = initExistingShm("wfc")

        self.useLong = False
        self.perturb = True
        self.prevMeasurement = 0
        self.amp = self.conf["amp"]
        self.rate = self.conf["rate"]
        self.norm = setFromConfig(self.conf, "norm", 1.0)
        self.droppedModeList = None

        self.currentShape = np.zeros_like(self.wfcShm.read_noblock())
        self.dofs = self.currentShape.size
        #Compute a random poke for the system
        poke = np.zeros(self.dofs)
        self.poke = poke.reshape(self.currentShape.shape).astype(self.currentShape.dtype)

        tmp = 0

        for i in range(100):
            tmp += np.max(self.psfShm.read())
        self.norm = tmp/100

        self.gradientDamp = setFromConfig(self.conf, 'gradientDamp', -1)

        # SPSA Parameters
        self.theta = self.currentShape.copy()
        self.k = 1  # Iteration counter
        self.a = setFromConfig(self.conf, 'a', 0.1)
        self.c = setFromConfig(self.conf, 'c', 0.1)
        self.A = setFromConfig(self.conf, 'A', 10)
        self.alpha = setFromConfig(self.conf, 'alpha', 0.602)
        self.gamma = setFromConfig(self.conf, 'gamma', 0.101)
        self.spsa_state = 0  # State variable for SPSA steps
        self.delta = None
        self.y_plus = None
        self.y_minus = None

        return
    
    def psgd(self):

        if self.droppedModeList is not None and len(self.droppedModeList) > 0:
            self.currentShape[self.droppedModeList] *= 0
        #Adjust the environmant
        self.wfcShm.write(self.currentShape)
        #Adjust loss function for GD
        if self.useLong:
            self.currentMeasurement = np.max(self.psfLongShm.read())/self.norm
        else:
            self.currentMeasurement = np.max(self.psfShm.read())/self.norm

        self.currentShape = self.wfcShm.read()
        #At the beginning of the cycle
        if self.perturb:

            #Make a self.prevMeasurement measurement of intensity
            self.prevMeasurement = self.currentMeasurement
            #Compute a random poke for the system
            poke = np.random.uniform(-self.amp,
                                     self.amp,
                                     self.dofs)
            self.poke = poke.reshape(self.currentShape.shape).astype(self.currentShape.dtype)
            #Apply the poke
            self.currentShape += self.poke

        else:

            #Compute any change made from the poke to the system
            delta = (self.currentMeasurement - self.prevMeasurement)
            """
            Remove phase adjustment of the random parameter and 
            Add a movement in the right direction of the poke scaled 
            by the improvement delta
            """
            deltaShape = (self.poke*self.rate*delta).reshape(self.currentShape.shape)
            if self.gradientDamp > 0:
                deltaShape = np.clip(deltaShape, -1*self.gradientDamp, self.gradientDamp)
            self.currentShape -= self.poke
            self.currentShape += deltaShape

        self.perturb = not self.perturb

        return
    
    def spsa(self):
        """
        Implement the SPSA algorithm with state management across function calls.
        """
        if self.spsa_state == 0:
            # Compute ak and ck
            self.ak = self.a / ((self.k + self.A) ** self.alpha)
            self.ck = self.c / (self.k ** self.gamma)

            # Generate perturbation vector delta of +1 or -1
            self.delta = np.random.choice([-1, 1], size=self.dofs)
            self.deltaShape = self.delta.reshape(self.currentShape.shape).astype(self.currentShape.dtype)

            # Compute theta_plus and theta_minus
            self.theta_plus = self.currentShape + self.ck * self.deltaShape
            self.theta_minus = self.currentShape - self.ck * self.deltaShape

            # Write theta_plus to the system
            self.wfcShm.write(self.theta_plus)

            # Advance state
            self.spsa_state = 1

        elif self.spsa_state == 1:
            # Read y_plus
            if self.useLong:
                self.y_plus = np.max(self.psfLongShm.read()) / self.norm
            else:
                self.y_plus = np.max(self.psfShm.read()) / self.norm

            # Write theta_minus to the system
            self.wfcShm.write(self.theta_minus)

            # Advance state
            self.spsa_state = 2

        elif self.spsa_state == 2:
            # Read y_minus
            if self.useLong:
                self.y_minus = np.max(self.psfLongShm.read()) / self.norm
            else:
                self.y_minus = np.max(self.psfShm.read()) / self.norm

            # Estimate gradient
            ghat = ((self.y_plus - self.y_minus) / (2 * self.ck)) * self.delta

            # Update currentShape
            self.currentShape -= self.ak * ghat.reshape(self.currentShape.shape)

            # Optionally, clip currentShape to allowed values
            # For example, if there are limits on the control parameters
            # self.currentShape = np.clip(self.currentShape, min_value, max_value)

            # Write updated theta to the system
            self.wfcShm.write(self.currentShape)

            # Increment k
            self.k += 1

            # Reset state
            self.spsa_state = 0

            # Optionally, measure current performance
            if self.useLong:
                current_performance = np.max(self.psfLongShm.read()) / self.norm
            else:
                current_performance = np.max(self.psfShm.read()) / self.norm

            # Optionally, print or log progress
            if self.k % 100 == 0 or self.k == 1:
                print(f"Iteration {self.k}, Performance: {current_performance:.4f}")

        return
    def flatten(self):
        self.currentShape *= 0
        return super().flatten()

    def __del__(self):
        super().__del__()
        return
    

if __name__ == "__main__":

    launchComponent(PSGDLoop, "loop", start = False)