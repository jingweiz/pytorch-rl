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

class AtariRamEnv(Env):  # atari games w/ ram states as input
    def __init__(self, args, env_ind=0):
        super(AtariRamEnv, self).__init__(args, env_ind)

        assert self.env_type == "atari-ram"
        try: import gym
        except ImportError as e: self.logger.warning("WARNING: gym not found")

        self.env = gym.make(self.game)
        self.env.seed(self.seed)    # NOTE: so each env would be different

        # action space setup
        self.actions     = range(self.action_dim)
        self.logger.warning("Action Space: %s", self.actions)

        # state space setup
        self.logger.warning("State  Space: %s", self.state_shape)

    def _preprocessState(self, state):    # NOTE: here the input is [0, 255], so we normalize
        return state/255.                 # TODO: check again the range, also syntax w/ python3

    @property
    def state_shape(self):
        return self.env.observation_space.shape[0]

    def render(self):
        return self.env.render()

    def visual(self):   # TODO: try to grab also the pixel-level outputs and visualize
        pass

    def sample_random_action(self):
        return self.env.action_space.sample()

    def reset(self):
        self._reset_experience()
        self.exp_state1 = self.env.reset()
        return self._get_experience()

    def step(self, action_index):
        self.exp_action = action_index
        self.exp_state1, self.exp_reward, self.exp_terminal1, _ = self.env.step(self.actions[self.exp_action])
        return self._get_experience()
