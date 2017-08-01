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

class GymEnv(Env):  # low dimensional observations
    def __init__(self, args, env_ind=0):
        super(GymEnv, self).__init__(args, env_ind)

        assert self.env_type == "gym"
        try: import gym
        except ImportError as e: self.logger.warning("WARNING: gym not found")

        self.env = gym.make(self.game)
        self.env.seed(self.seed)    # NOTE: so each env would be different

        # action space setup
        self.actions     = range(self.action_dim)
        self.logger.warning("Action Space: %s", self.actions)

        # state space setup
        self.logger.warning("State  Space: %s", self.state_shape)

        # continuous space
        if args.agent_type == "a3c":
            self.enable_continuous = args.enable_continuous
        else:
            self.enable_continuous = False

    def _preprocessState(self, state):    # NOTE: here no preprecessing is needed
        return state

    @property
    def state_shape(self):
        return self.env.observation_space.shape[0]

    def render(self):
        if self.mode == 2:
            frame = self.env.render(mode='rgb_array')
            frame_name = self.img_dir + "frame_%04d.jpg" % self.frame_ind
            self.imsave(frame_name, frame)
            self.logger.warning("Saved  Frame    @ Step: " + str(self.frame_ind) + " To: " + frame_name)
            self.frame_ind += 1
            return frame
        else:
            return self.env.render()


    def visual(self):
        pass

    def sample_random_action(self):
        return self.env.action_space.sample()

    def reset(self):
        self._reset_experience()
        self.exp_state1 = self.env.reset()
        return self._get_experience()

    def step(self, action_index):
        self.exp_action = action_index
        if self.enable_continuous:
            self.exp_state1, self.exp_reward, self.exp_terminal1, _ = self.env.step(self.exp_action)
        else:
            self.exp_state1, self.exp_reward, self.exp_terminal1, _ = self.env.step(self.actions[self.exp_action])
        return self._get_experience()
