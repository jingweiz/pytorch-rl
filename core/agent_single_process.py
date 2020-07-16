from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import random
import time
import math
import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.multiprocessing as mp

from utils.helpers import Experience, one_hot

class AgentSingleProcess(mp.Process):
    def __init__(self, master, process_id=0):
        super(AgentSingleProcess, self).__init__(name = "Process-%d" % process_id)
        # NOTE: self.master.* refers to parameters shared across all processes
        # NOTE: self.*        refers to process-specific properties
        # NOTE: we are not copying self.master.* to self.* to keep the code clean

        self.master = master
        self.process_id = process_id

        # env
        self.env = self.master.env_prototype(self.master.env_params, self.process_id)
        # model
        self.model = self.master.model_prototype(self.master.model_params)
        self._sync_local_with_global()

        # experience
        self._reset_experience()

    def _reset_experience(self):    # for getting one set of observation from env for every action taken
        self.experience = Experience(state0 = None,
                                     action = None,
                                     reward = None,
                                     state1 = None,
                                     terminal1 = False) # TODO: should check this again

    def _sync_local_with_global(self):  # grab the current global model for local learning/evaluating
        self.model.load_state_dict(self.master.model.state_dict())

    # NOTE: since no backward passes has ever been run on the global model
    # NOTE: its grad has never been initialized, here we ensure proper initialization
    # NOTE: reference: https://discuss.pytorch.org/t/problem-on-variable-grad-data/957
    def _ensure_global_grads(self):
        for global_param, local_param in zip(self.master.model.parameters(),
                                             self.model.parameters()):
            if global_param.grad is not None:
                return
            global_param._grad = local_param.grad

    def _forward(self, observation):
        raise NotImplementedError("not implemented in base class")

    def _backward(self, reward, terminal):
        raise NotImplementedError("not implemented in base class")

    def run(self):
        raise NotImplementedError("not implemented in base class")
