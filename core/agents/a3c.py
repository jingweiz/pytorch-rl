from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.multiprocessing as mp

from core.agent import Agent
from core.agents.a3cSingleProcess import A3CLearner, A3CEvaluator, A3CTester

class A3CAgent(Agent):
    def __init__(self, args, env_prototype, model_prototype, memory_prototype):
        super(A3CAgent, self).__init__(args, env_prototype, model_prototype, memory_prototype)
        self.logger.warning("<===================================> A3C-Master {Env(dummy) & Model}")

        # dummy_env just to get state_shape & action_dim
        self.dummy_env   = self.env_prototype(self.env_params, self.num_processes)
        self.state_shape = self.dummy_env.state_shape
        self.action_dim  = self.dummy_env.action_dim
        del self.dummy_env

        # global shared model
        self.model_params.state_shape = self.state_shape
        self.model_params.action_dim  = self.action_dim
        self.model = self.model_prototype(self.model_params)
        self._load_model(self.model_file)   # load pretrained model if provided
        self.model.share_memory()           # NOTE

        # learning algorithm # TODO: could also linearly anneal learning rate
        self.optimizer = self.optim(self.model.parameters(), lr = self.lr)
        self.optimizer.share_memory()       # NOTE

        # global counters
        self.frame_step   = mp.Value('l', 0) # global frame step counter
        self.train_step   = mp.Value('l', 0) # global train step counter
        # global training stats
        self.p_loss_avg   = mp.Value('d', 0.) # global policy loss
        self.v_loss_avg   = mp.Value('d', 0.) # global value loss
        self.loss_avg     = mp.Value('d', 0.) # global loss
        self.loss_counter = mp.Value('l', 0)  # storing this many losses
        self._reset_training_loggings()

    def _reset_training_loggings(self):
        self.p_loss_avg.value   = 0.
        self.v_loss_avg.value   = 0.
        self.loss_avg.value     = 0.
        self.loss_counter.value = 0

    def fit_model(self):
        self.jobs = []
        for process_id in range(self.num_processes):
            self.jobs.append(A3CLearner(self, process_id))
        self.jobs.append(A3CEvaluator(self, self.num_processes))

        self.logger.warning("<===================================> Training ...")
        for job in self.jobs:
            job.start()
        for job in self.jobs:
            job.join()

    def test_model(self):
        self.jobs = []
        self.jobs.append(A3CTester(self))

        self.logger.warning("<===================================> Testing ...")
        for job in self.jobs:
            job.start()
        for job in self.jobs:
            job.join()
