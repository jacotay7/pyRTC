import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pyRTC.Pipeline import * 
from pyRTC.utils import *
import torch


class pyRTCEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, conf):
        super(pyRTCEnv, self).__init__()

        confLoop = conf["loop"]
        self.device = setFromConfig(confLoop, 'device', 'cuda:0')

        # DM
        confDM = conf["wfc"]

        #------------------------------Shared Memory Objects (AO System Components) ---------------------------#

        # Wavefront Corrector
        self.wfc2DShm, self.wfc2DDims, self.wfc2DDtype = initExistingShm("wfc2D")
        self.wfcShm, self.wfcDims, self.wfcDtype = initExistingShm("wfc")
        
        self.numModes = self.wfcDims[0]
        self.numActiveModes = self.numModes
        
        # Precomputed Slopes
        self.signalShm, self.signalDims, self.signalDtype = initExistingShm("signal")

        # Strehl Tracker
        self.strehlShm, self.strehlDims, self.strehlDtype = initExistingShm("strehl")


        #------------------------------System Parameters & Interaction----------------------------------#

        # Loop Parameters
        self.gain  =  setFromConfig(confLoop, "gain", 0.1)
        self.linearActionNorm = setFromConfig(confLoop, "linearActionNorm", 1)
        self.actionNorm = setFromConfig(confLoop, "actionNorm", 1000)
        self.leakyGain =  setFromConfig(confLoop, "leakyGain", 0)
        self.actionScale = setFromConfig(confLoop, "actionScale", 0.01)
        self.commandMagnitudeWeight = setFromConfig(confLoop, "commandMagnitudeWeight", 1.0)

        # Calibration Data
        self.IMFile = setFromConfig(confLoop, "IMFile", "")
        self.IM = torch.tensor(np.load(self.IMFile), dtype=torch.float32, device=self.device)

        self.M2C_file = setFromConfig(confDM, "m2cFile", "")
        self.M2C_CL = torch.tensor(np.load(self.M2C_file), dtype=torch.float32, device=self.device)
        self.C2M = torch.pinverse(self.M2C_CL)

        #Filter (F) projects a shape onto the basis
        self.F = self.M2C_CL@self.C2M


        self.CM = torch.zeros_like(self.IM.T, device=self.device)
        self.CM[:self.numActiveModes,:] = torch.pinverse(self.IM[:,:self.numActiveModes], 
                                                         rcond=0)
        self.CM[self.numActiveModes:,:] = 0
        self.gCM = self.gain*self.CM

        self.numActuators = setFromConfig(confDM, "numActuators", 97)

        #Actuators across the dm (approximation of squaring the circle)
        self.nActuator = int(np.round(np.sqrt(self.numActuators/(np.pi/4)),0))
       
        if self.numActuators == 97:
            xx, yy = np.meshgrid(np.arange(11), np.arange(11))
            layout = np.sqrt((xx - 5)**2 + (yy-5)**2) < 5.5
        
        nonzero_indices = torch.nonzero(torch.tensor(layout), as_tuple=True)

        # This will give you two tensors: xvalid and yvalid

        print("TORCH TENSOR THE VALID PIXELS")
        self.xvalid = nonzero_indices[0].clone().detach()
        self.yvalid = nonzero_indices[1].clone().detach()
        

        #-----------------Internal Env Variables------------------------#

        self.nLoop = setFromConfig(confLoop, "nLoop", 10000)
        self.SR           = []

        self.SE_PSF       = []
        self.LE_PSF       = []

    

        #This will set the episode length
        self.current_step = 0
        self.readtimeout = 1 #wait 1s before auto returning a read


        self.linearCorrection = None
        self.curWfc2D = torch.zeros_like(torch.tensor(self.wfc2DShm.read_noblock()), dtype=torch.float32, device=self.device)

        self.signalObs = torch.zeros_like(torch.tensor(self.signalShm.read_noblock()), dtype=torch.float32, device=self.device)
        self.lastAction = None

        self.defaultAction = torch.zeros(self.numActuators, dtype = torch.float32, device=self.device)
        self.control = torch.zeros_like(torch.tensor(self.wfcShm.read_noblock()), device=self.device)

        self.lastShape = self.curWfc2D.cpu().numpy()

    def _get_obs(self, blocking=True):

        if blocking:
            self.lastShape = self.wfc2DShm.read_timeout(self.readtimeout)
            # make a blocking read of the wfc to make sure the action has been set
            self.curWfc2D = torch.tensor(self.lastShape, device=self.device)
            # read the slopes 2D signal
            self.signalObs = torch.tensor(self.signalShm.read_timeout(self.readtimeout), device=self.device)
        else:
            self.lastShape = self.wfc2DShm.read_noblock()
            # make a blocking read of the wfc to make sure the action has been set
            self.curWfc2D = torch.tensor(self.lastShape , device=self.device)
            # read the slopes 2D signal
            self.signalObs = torch.tensor(self.signalShm.read_noblock(), device=self.device)

        #Normalize linearAction amplitude to be in correct range
        self.linearCorrection = -(self.M2C_CL@self.CM@self.signalObs)
        obs = self.vec_to_img(self.linearCorrection)
        
        return obs

    def applyAction(self, action):

        self.lastAction = action

        #Make the action into a modal vector
        modal_action = self.C2M@action

        #Leak the control vector
        self.control *= (1-self.leakyGain)
        self.control[:self.numActiveModes] += modal_action[:self.numActiveModes]

        self.lastControl = self.control.cpu().numpy()

        #Send to mirror
        self.wfcShm.write(self.lastControl )

    def comp_reward(self, verbose=False):

        # reward = self.strehlShm.read()[0]
        # reward -=  self.loop.tipTiltShm.read_noblock()/10
        reward = -torch.mean(self.signalObs**2)
        reward = np.float64(reward) 
        return reward #- self.commandMagnitudeWeight*np.mean(np.abs(self.lastShape))

    def step(self, i, action):

        self.current_step += 1

        action = self.img_to_vec(action) * self.actionScale

        #Force the Shms to block until updates
        # self.wfc2DShm.markSeen()
        # self.signalShm.markSeen()

        #Send the command to the mirror
        self.applyAction(action)
        
        #Get a new observation of the current state
        observation = self._get_obs(blocking=True)

        # Compute reward
        reward = self.comp_reward()

        truncated = False

        #Terminated condition -> divergent slopes
        terminated = False #or truncated

        return observation, reward, 0, False, dict()


    def reset(self, seed=None, options=None):

        self.current_step = 0

        self.control *= 0

        #Flatten the mirror
        self.applyAction(self.defaultAction)

        # Sleep to make sure the blocking order is reset
        time.sleep(0.05)

        #Build the observation
        observation = self._get_obs(blocking=False)

        #Flatten the mirror
        self.applyAction(self.defaultAction)

        # Get info dict
        info = self._get_info()

        return observation

    def reset_soft(self):
        return self.reset()

        return observation, info
    def _get_info(self):
        return {
            'TimeLimit.truncated': False
        }
    def render(self):
        if self.render_mode == None:
            return
    def close(self):
        return
    
    def sample_noise(self, sigma):

        noise_cpu = sigma * np.random.normal(0,1 , size = (int(self.numActuators),))
        noise = torch.tensor(noise_cpu, dtype=torch.float32, device=self.device)
        return self.vec_to_img(self.F @ noise)


    def vec_to_img(self, action_vec):
        valid_actus = torch.zeros((self.nActuator , self.nActuator), dtype = action_vec.dtype, device=self.device)
        valid_actus[self.xvalid, self.yvalid] = action_vec

        return valid_actus
    

    def img_to_vec(self, action):
        assert len(action.shape) == 2
        
        return action[self.xvalid, self.yvalid]
    
    def calculate_strehl_AVG(self):

        return 0
