from pyRTC.utils import *
from pyRTC.Pipeline import *
from pyRTC.Optimizer import *

import time
import numpy as np
import datetime
from tqdm import tqdm

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback



# class NCPAEnv(gym.Env):
#     """Custom Environment that follows gym interface"""

#     def __init__(self, loop, slopes, psf):
#         super(NCPAEnv, self).__init__()
        
#         self.num_correction_modes = 20

#         self.psf = psf
#         self.slopes = slopes
#         self.loop = loop
#         self.bufferSize = 5

#         # Define action and observation space
#         # They must be gym.spaces objects
#         # Example when using discrete actions:
#         self.action_space = spaces.Box(-1,1, shape = (self.num_correction_modes,), dtype=float)
#         # Example for using image as input (observation space):
#         self.strehl_observation_space = spaces.Box(low=0, high=2, shape=(self.bufferSize,), dtype=float)
#         self.previous_pertub_space = spaces.Box(low=-1, high=1, shape=(self.bufferSize, self.num_correction_modes), dtype=float)

#         self.observation_space = spaces.Dict(
#             {
#                 "strehl": self.strehl_observation_space,
#                 "previousActions": self.previous_pertub_space
#             }
#         )

#         self.default_action = np.zeros_like(self.action_space.sample())

#         self.normFactor = 650
#         self.NCPAMAG = 2

#         #A dummy vector of size numModes
#         self.commandVec = np.zeros(self.loop.IM.shape[1], dtype = self.default_action.dtype)

#         self.resetNCPA()

#         self.current_step = 0

#         self.epochLength = 32

#         # Initialize obs vector
#         self.strehlObs = np.zeros_like(self.strehl_observation_space.sample())
#         self.perturbObs = np.zeros_like(self.previous_pertub_space.sample())
#         self.previousAction = np.zeros_like(self.default_action)


#     def step(self, action):
#         #Update epoch progress
#         self.current_step  += 1

#         self.applyAction(action)

#         obs = self.getObs()

#         self.previousAction = action

#         #reward is total stehl improvement over the buffer
#         a,b = obs["strehl"][-1], obs["strehl"][-2]
#         reward = (a-b)*np.exp(3*max(a,b))#np.sum(obs["strehl"][1:] -  obs["strehl"][:-1])

#         #Terminated condition -> divergent slopes
#         terminated = self.current_step >= self.epochLength
#         truncated = False# check if the episode is done

#         info = {} # additional data, not required
        
#         return obs, reward, terminated, truncated, info

#     def resetNCPA(self):

#         #Reset ref slopes
#         self.slopes.loadRefSlopes()

#         #Ask for an action
#         self.ncpa_action = self.action_space.sample()
#         #Rescale and add to dummy, same as adding one big step in an action direction
#         self.commandVec[:self.ncpa_action.size] = self.NCPAMAG*self.ncpa_action/self.normFactor
#         #Change the reference slopes
#         self.slopes.refSlopes[self.slopes.validSubAps] += self.loop.IM@self.commandVec
#         #Save perturbed slopes as original
#         self.origRef = self.slopes.refSlopes[self.slopes.validSubAps]

#         return

#     def getObs(self):

#         self.strehlObs[:-1] = self.strehlObs[1:]
#         self.perturbObs[:-1] = self.perturbObs[1:]

#         self.perturbObs[-1] = self.previousAction

#         #Burn one image (its asynch imaging)
#         self.psf.computeStrehl()
#         self.strehlObs[-1]  = self.psf.computeStrehl() # calculate reward

#         return {"strehl": self.strehlObs, "previousActions": self.perturbObs}

#     def reset(self, seed=None, options=None):
        
#         #Define a new NCPA
#         self.resetNCPA()
#         time.sleep(1)

#         #Reload the strehl observation buffer
#         self.psf.computeStrehl()
#         self.strehlObs = self.psf.computeStrehl()*np.ones_like(self.strehlObs)
#         #Reset previous action buffer 
#         self.perturbObs = np.zeros_like(self.perturbObs)

#         #Reset Epoch
#         self.current_step = 0

#         obs = {"strehl": self.strehlObs, "previousActions": self.perturbObs}

#         return obs, dict()

#     def applyAction(self, action):
#         #Update the reference slopes based on the action
#         self.commandVec[:action.size] = action/self.normFactor
#         self.slopes.refSlopes[self.slopes.validSubAps] = self.origRef + self.loop.IM@self.commandVec
        
#         return
    
#     def close(self):
#         self.reset()
#         # Perform any necessary cleanup
#         return


    # def __init__(self, conf, loop, slopes, psf, modelName = None):
        
    #     self.modelName = modelName
    #     folder = "/home/whetstone/RLAO/pyRTC/models/"
    #     time_str = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    #     self.loop = loop
    #     self.env = NCPAEnv(loop, slopes, psf)

    #     self.logdir = setFromConfig(conf["optimizer"], "logdir", "ncpa_optimizer_logdir")

    #     if self.modelName is not None:
    #         self.model = PPO.load(folder + self.modelName, env = self.env)
    #     else:
    #         self.model = PPO("MultiInputPolicy", 
    #                         self.env,
    #                         n_steps = self.env.epochLength,
    #                         batch_size = self.env.epochLength,
    #                         verbose=1,
    #                         tensorboard_log=self.logdir)

            
    #         self.modelName = "ncpa_rl_model_"+time_str

    #     # Use deterministic actions for evaluation
    #     self.callback = EvalCallback(self.env, best_model_save_path=folder,
    #                                 log_path=folder, eval_freq=1, verbose=1, n_eval_episodes = 3,
    #                                 deterministic=False, render=False)

    #     # self.callback = SaveOnBestTrainingRewardCallback(1,
    #     #                                                  folder,
    #     #                                                  self.modelName)
  
    #     self.numEpochs = 100
    #     self.learningTimesteps = self.env.epochLength*self.numEpochs

    #     return
    
    # def learn(self):
    #     self.loop.start()
    #     time.sleep(1)
    #     self.model.learn(total_timesteps=self.learningTimesteps,
    #              progress_bar=False,
    #              callback =  self.callback,
    #              reset_num_timesteps=False,
    #              tb_log_name=datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
    #     self.loop.stop()



class NCAPOptimizer(Optimizer):

    def __init__(self, conf) -> None:
        
        self.wfcShm, self.wfcDims, self.wfcDtype = initExistingShm("wfc")
        self.strehlShm, _, _ = initExistingShm("strehl")
        self.startMode = setFromConfig(conf, "startMode", 0)
        self.endMode = setFromConfig(conf, "endMode", 20)
        self.correctionMag = setFromConfig(conf, "correctionMag", 2e-3)
        self.numReads = setFromConfig(conf, "numReads", 5)

        super().__init__(conf)

    def objective(self, trial):

        numModesCorrect = self.endMode - self.startMode
        modalCoefs = np.zeros(self.wfcDims, dtype=self.wfcDtype)
        for i in range(self.startMode,numModesCorrect):
            modalCoefs[i] = np.float32(trial.suggest_float(f'{i}', 
                                                           -self.correctionMag,
                                                            self.correctionMag))

        self.wfcShm.write(modalCoefs)

        result = np.empty(self.numReads)
        for i in range(self.numReads):
            result[i] = self.strehlShm.read()
        return np.mean(result)
    
    def applyOptimum(self):
        super().applyOptimum()
        modalCoefs = np.zeros(self.wfcDims, dtype=self.wfcDtype)
        for k in self.study.best_params.keys():
            modalCoefs[int(k)] = self.study.best_params[k]
        self.wfcShm.write(modalCoefs)
        return 