from pyRTC.Loop import *
from pyRTC.Pipeline import *
from pyRTC.utils import *
import argparse
import os 
import time
import torch
import gymnasium.spaces as spaces
import gymnasium
from rtgym import RealTimeGymInterface


class LoopGymInterface(RealTimeGymInterface):

    def __init__(self, loop):
        self.loop = loop
        self.initialized = False



    def get_action_space(self):
        numActuators = self.loop.confWFC['numModes']

        high_cmd = np.ones(numActuators) * np.inf
        low_cmd = np.ones(numActuators) * -np.inf

        return spaces.Box(low=low_cmd, high=high_cmd, dtype=np.float32)

    def get_observation_space(self):

        high_wfs = np.ones((self.loop.slope_width, self.loop.slope_height)) * np.inf
        low_wfs = np.ones((self.loop.slope_width, self.loop.slope_height)) * -np.inf

        slope_obs = spaces.Box(low=low_wfs, high=high_wfs, dtype=np.float32)

        cmd_obs = self.get_action_space()

        return spaces.Tuple([cmd_obs, slope_obs])


    def get_default_action(self):
        return np.zeros(self.loop.confWFC['numModes'], dtype=np.float32)


    def send_control(self, control):

        control[self.loop.numActiveModes:] = 0
        self.loop.wfcShm.write(control)



    def reset(self, seed=None, options=None):
        if not self.initialized:
            self.initialized = True

        # Flatten the mirror
        self.loop.wfcShm.write(self.get_default_action())

        # Read actuator positions
        act_pos = self.loop.wfcShm.read()
            
        # Read current slope map
        slopes = self.loop.slopesShm.read()

        return [np.array(act_pos, dtype='float32'),
                np.array(slopes, dtype='float32')], {}


    def get_obs_rew_terminated_info(self):
        
        # Read actuator positions
        act_pos = self.loop.wfcShm.read()
            
        # Read current slope map
        slopes = self.loop.slopesShm.read()

        obs =  [np.array(act_pos, dtype='float32'),
                np.array(slopes, dtype='float32')]
        
        reward = np.exp(-np.var(slopes), dtype='float32')

        terminated = reward < np.exp(-9)

        info = {}

        return obs, reward, terminated, info

    def wait(self):
        self.send_control(self.get_default_action())