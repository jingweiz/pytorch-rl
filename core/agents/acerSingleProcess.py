from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import random
import time
import math

import torch
import torch.multiprocessing as mp
from torch.autograd import Variable
import torch.nn.functional as F

from utils.helpers import Experience, AugmentedExperience, one_hot

class ACERSingleProcess(mp.Process):
    def __init__(self, master, process_id=0):
        super(ACERSingleProcess, self).__init__(name = "Process-%d" % process_id)
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

        # lstm hidden states
        if self.master.enable_lstm:
            self._reset_lstm_hidden_vb_episode() # clear up hidden state
            self._reset_lstm_hidden_vb_rollout() # detach the previous variable from the computation graph

        # NOTE global variable pi
        if self.master.enable_continuous:
            self.pi_vb = Variable(torch.Tensor([math.pi]).type(self.master.dtype))

        self.master.logger.warning("Registered ACER-SingleProcess-Agent #" + str(self.process_id) + " w/ Env (seed:" + str(self.env.seed) + ").")

class ACERLearner(ACERSingleProcess):
    def __init__(self, master, process_id=0):
        master.logger.warning("<===================================> ACER-Learner #" + str(process_id) + " {Env & Model}")
        super(A3CLearner, self).__init__(master, process_id)

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
        super(A3CTester, self).__init__(master, process_id)

        self.training = False   # choose actions w/ max probability
        self._reset_loggings()

        self.start_time = time.time()
