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

class AtariEnv(Env):  # pixel-level inputs
    def __init__(self, args, env_ind=0):
        super(AtariEnv, self).__init__(args, env_ind)

        assert self.env_type == "atari"
        try: import gym
        except ImportError as e: self.logger.warning("WARNING: gym not found")

        self.env = gym.make(self.game)
        self.env.seed(self.seed)    # NOTE: so each env would be different

        # action space setup
        self.actions     = range(self.action_dim)
        self.logger.warning("Action Space: %s", self.actions)
        # state space setup
        self.hei_state = args.hei_state
        self.wid_state = args.wid_state
        self.preprocess_mode = args.preprocess_mode if not None else 0 # 0(crop&resize) | 1(rgb2gray) | 2(rgb2y)
        assert self.hei_state == self.wid_state
        self.logger.warning("State  Space: (" + str(self.state_shape) + " * " + str(self.state_shape) + ")")

    def _preprocessState(self, state):
        if self.preprocess_mode == 3:   # crop then resize
            state = preprocessAtari(state)
        if self.preprocess_mode == 2:   # rgb2y
            state = scale(rgb2y(state), self.hei_state, self.wid_state) / 255.
        elif self.preprocess_mode == 1: # rgb2gray
            state = scale(rgb2gray(state), self.hei_state, self.wid_state) / 255.
        elif self.preprocess_mode == 0: # do nothing
            pass
        return state.reshape(self.hei_state * self.wid_state)

    @property
    def state_shape(self):
        return self.hei_state

    def render(self):
        return self.env.render()

    def visual(self):
        if self.visualize:
            self.win_state1 = self.vis.image(np.transpose(self.exp_state1, (2, 0, 1)), env=self.refs, win=self.win_state1, opts=dict(title="state1"))
        if self.mode == 2:
            frame_name = self.img_dir + "frame_%04d.jpg" % self.frame_ind
            self.imsave(frame_name, self.exp_state1)
            self.logger.warning("Saved  Frame    @ Step: " + str(self.frame_ind) + " To: " + frame_name)
            self.frame_ind += 1

    def sample_random_action(self):
        return self.env.action_space.sample()

    def reset(self):
        # TODO: could add random start here, since random start only make sense for atari games
        self._reset_experience()
        self.exp_state1 = self.env.reset()
        return self._get_experience()

    def step(self, action_index):
        self.exp_action = action_index
        self.exp_state1, self.exp_reward, self.exp_terminal1, _ = self.env.step(self.actions[self.exp_action])
        return self._get_experience()
