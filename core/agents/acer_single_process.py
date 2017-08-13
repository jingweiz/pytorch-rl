from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import random
import time
import math
import torch
from torch.autograd import Variable
import torch.nn.functional as F

from utils.helpers import ACERExperience
from core.agent_single_process import AgentSingleProcess

class ACERSingleProcess(AgentSingleProcess):
    def __init__(self, master, process_id=0):
        super(ACERSingleProcess, self).__init__(master, process_id)

        # diff from a3c, acer is capable of off-policy learning and use replay buffer
        self.memory = self.master.memory_prototype(capacity = self.master.memory_params.memory_size,
                                                   max_episode_length = self.master.early_stop)

        # # lstm hidden states
        # if self.master.enable_lstm:
        #     self._reset_lstm_hidden_vb_episode() # clear up hidden state
        #     self._reset_lstm_hidden_vb_rollout() # detach the previous variable from the computation graph
        #
        # # NOTE global variable pi
        # if self.master.enable_continuous:
        #     self.pi_vb = Variable(torch.Tensor([math.pi]).type(self.master.dtype))

        self.master.logger.warning("Registered ACER-SingleProcess-Agent #" + str(self.process_id) + " w/ Env (seed:" + str(self.env.seed) + ").")

class ACERLearner(ACERSingleProcess):
    def __init__(self, master, process_id=0):
        master.logger.warning("<===================================> ACER-Learner #" + str(process_id) + " {Env & Model}")
        super(ACERLearner, self).__init__(master, process_id)

        # learning algorithm    # TODO: adjust learning to each process maybe ???
        self.optimizer = self.master.optim(self.model.parameters(), lr = self.master.lr)

        self._reset_rollout()

        self.training = True    # choose actions by polinomial
        # local counters
        self.frame_step   = 0   # local frame step counter
        self.train_step   = 0   # local train step counter
        # local training stats
        self.p_loss_avg   = 0.  # global policy loss
        self.v_loss_avg   = 0.  # global value loss
        self.loss_avg     = 0.  # global value loss
        self.loss_counter = 0   # storing this many losses
        self._reset_training_loggings()

        # copy local training stats to global every prog_freq
        self.last_prog = time.time()

class ACEREvaluator(ACERSingleProcess):
    def __init__(self, master, process_id=0):
        master.logger.warning("<===================================> ACER-Evaluator {Env & Model}")
        super(ACEREvaluator, self).__init__(master, process_id)

        self.training = False   # choose actions w/ max probability
        self._reset_loggings()

        self.start_time = time.time()
        self.last_eval = time.time()

class ACERTester(ACERSingleProcess):
    def __init__(self, master, process_id=0):
        master.logger.warning("<===================================> ACER-Tester {Env & Model}")
        super(ACERTester, self).__init__(master, process_id)

        self.training = False   # choose actions w/ max probability
        self._reset_loggings()

        self.start_time = time.time()
