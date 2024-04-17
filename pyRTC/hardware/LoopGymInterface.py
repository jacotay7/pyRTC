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
        self.control = self.get_default_action()
        self.flat2D = np.zeros((self.loop.wfc2D_width, self.loop.wfc2D_height))


    def merge_slope_cmd(self, slope, cmd):

        if slope.shape[1] >= cmd.shape[1]:

            pad_size = slope.shape[1] - cmd.shape[1]

            padded_cmd = np.pad(cmd, ((0, 0), (0, pad_size)), mode='constant', constant_values=0)

            return np.concatenate((slope, padded_cmd), axis=0)
        
        else:
            pad_size = cmd.shape[1] - slope.shape[1]

            padded_slopes = np.pad(slope, ((0, 0), (0, pad_size)), mode='constant', constant_values=0)

            return np.concatenate((padded_slopes, cmd), axis=0)



    def get_action_space(self):
        numActuators = self.loop.confWFC['numModes']

        return spaces.Box(low=-1, high=1, shape=(numActuators, ) , dtype=np.float32)



    def get_observation_space(self):

        if self.loop.slope_height >= self.loop.wfc2D_height:

            pad_size = 0
    
        else:
            pad_size = self.loop.wfc2D_height - self.loop.slope_height


        return spaces.Box(low=-1, high=1, shape=(self.loop.slope_width + self.loop.wfc2D_width, self.loop.slope_height + pad_size), dtype=np.float32)



    def get_default_action(self):
        return np.zeros(self.loop.confWFC['numModes'], dtype=np.float32)


    def send_control(self, control):

        control[self.loop.numActiveModes:] = 0

        self.control = control
        
        self.loop.wfcShm.write(control)



    def reset(self, seed=None, options=None):
        if not self.initialized:
            self.initialized = True

        # Flatten the mirror
        flat_cmd = self.get_default_action()

        self.send_control(flat_cmd)

        # Get flat dm map
        dm = self.flat2D
            
        # Read current slope map
        slopes = self.loop.slopesShm.read_noblock()

        obs = self.merge_slope_cmd(slopes, dm)

    
        return np.array(obs), {}


    def get_obs_rew_terminated_info(self):
        
        # Read actuator positions
        act_pos = self.control

        # padded_act_pos = np.pad(act_pos, ((0, self.loop.slope_height - act_pos.shape[0]), (0, 0)), mode='constant', constant_values=0)
            
        # Read current slope map
        slopes = self.loop.slopesShm.read()

        dm = self.loop.wfc2DShm.read()

        # obs =  [np.array(act_pos, dtype='float32'),
        #         np.array(slopes, dtype='float32')]

        obs = self.merge_slope_cmd(slopes, dm)
        
        reward = np.exp(-np.var(slopes), dtype='float32')

        terminated = reward < np.exp(-9)

        info = {}

        return np.array(obs), reward, terminated, info

    def wait(self):
        self.send_control(self.get_default_action())