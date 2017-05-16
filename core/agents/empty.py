from __future__ import absolute_import
from __future__ import division
import random

from utils.helpers import Experience
from rl.agent import Agent

class EmptyAgent(Agent):
    def __init__(self, args, env_prototype, model_prototype, memory_prototype):
        super(EmptyAgent, self).__init__(args, env_prototype, model_prototype, memory_prototype)
        self.logger.warning("<===================================> Empty")

        # env
        self.env = self.env_prototype(self.env_params)
        self.state_shape = self.env.state_shape
        self.action_dim  = self.env.action_dim

        self._reset_experience()

    def _forward(self, state):
        pass

    def _backward(self, reward, terminal):
        pass

    def _eval_model(self):
        pass

    def fit_model(self):    # the most basic control loop, to ease integration of new envs
        self.step = 0
        should_start_new = True
        while self.step < self.steps:
            if should_start_new:
                self._reset_experience()
                self.experience = self.env.reset()
                assert self.experience.state1 is not None
                if self.visualize: self.env.visual()
                if self.render: self.env.render()
                should_start_new = False
            action = random.randrange(self.action_dim)      # thus we only randomly sample actions here, since the model hasn't been updated at all till now
            self.experience = self.env.step(action)
            if self.experience.terminal1 or self.early_stop and (episode_steps + 1) >= self.early_stop:
                should_start_new = True

    def test_model(self):
        pass
