from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from copy import deepcopy
from gym.spaces.box import Box
import inspect

from utils.helpers import Experience            # NOTE: here state0 is always "None"
from utils.helpers import preprocessAtari, rgb2gray, rgb2y, scale
from core.env import Env

class OpenSim(Env):  # low dimensional observations
    """ Class to setup the OpenSim-RL environment (https://github.com/praveen-palanisamy/pytorch-rl.git) Where the agent has to learn to run! Continuous (18 dim) action space."""
    def __init__(self, args, env_ind=0):
        super(OpenSim, self).__init__(args, env_ind)

        assert self.env_type == "opensim"
        try: from osim.env import RunEnv 
        except ImportError as e: self.logger.warning("WARNING: opensim not found")

        self.env = RunEnv(visualize= True)
        #self.env.seed(self.seed)    # NOTE: so each env would be different

        # action space setup
        self.actions     = range(self.action_dim)
        self.logger.warning("Action Space: %s", self.env.action_space)

        # state space setup
        self.logger.warning("State  Space: %s", self.state_shape)

        # continuous space
        #if args.agent_type == "a3c":
        self.enable_continuous = True #args.enable_continuous

    def _preprocessState(self, state):    # NOTE: here no preprecessing is needed
        return state
    
    @property
    def action_dim(self):
        return self.env.action_space.shape[0]

    @property
    def state_shape(self):
        return self.env.observation_space.shape[0]

    def render(self):
        #if self.mode == 2:
        #    frame = self.env.render(mode='rgb_array')
        #    frame_name = self.img_dir + "frame_%04d.jpg" % self.frame_ind
        #    self.imsave(frame_name, frame)
        #    self.logger.warning("Saved  Frame    @ Step: " + str(self.frame_ind) + " To: " + frame_name)
        #    self.frame_ind += 1
        #    return frame
        #else:
        #    return self.env.render()
        return


    def visual(self):
        pass

    def sample_random_action(self):
        return self.env.action_space.sample()

    def reset(self):
        self._reset_experience()
        self.exp_state1 = self.env.reset()
        return self._get_experience()

    def step(self, action):
        self.exp_action = action
        if self.enable_continuous:
            self.exp_state1, self.exp_reward, self.exp_terminal1, _ = self.env.step(self.exp_action)
        return self._get_experience()
