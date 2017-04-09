from __future__ import absolute_import
from __future__ import division
import torch
import torch.optim as optim

from utils.helpers import Experience

class Agent(object):
    def __init__(self, args, env_prototype, model_prototype, memory_prototype=None):
        # logging
        self.logger = args.logger

        # prototypes for env & model & memory
        self.env_prototype = env_prototype          # NOTE: instantiated in fit_model() of inherited Agents
        self.env_params = args.env_params
        self.model_prototype = model_prototype      # NOTE: instantiated in fit_model() of inherited Agents
        self.model_params = args.model_params
        self.memory_prototype = memory_prototype    # NOTE: instantiated in __init__()  of inherited Agents (dqn needs, a3c doesn't so only pass in None)
        self.memory_params = args.memory_params

        # params
        self.model_name = args.model_name           # NOTE: will save the current model to model_name
        self.model_file = args.model_file           # NOTE: will load pretrained model_file if not None

        self.render = args.render
        self.visualize = args.visualize
        if self.visualize:
            self.vis = args.vis
            self.refs = args.refs

        self.save_best = args.save_best
        if self.save_best:
            self.best_step   = None                 # NOTE: achieves best_reward at this step
            self.best_reward = None                 # NOTE: only save a new model if achieves higher reward

        self.hist_len = args.hist_len
        self.hidden_dim = args.hidden_dim

        self.use_cuda = args.use_cuda
        self.dtype = args.dtype

        # agent_params
        # criteria and optimizer
        self.value_criteria = args.value_criteria
        self.optim = args.optim
        # hyperparameters
        self.steps = args.steps
        self.early_stop = args.early_stop
        self.gamma = args.gamma
        self.clip_grad = args.clip_grad
        self.lr = args.lr
        self.eval_freq = args.eval_freq
        self.eval_steps = args.eval_steps
        self.prog_freq = args.prog_freq
        self.test_nepisodes = args.test_nepisodes
        if args.agent_type == "dqn":
            self.enable_double_dqn  = args.enable_double_dqn
            self.enable_dueling = args.enable_dueling
            self.dueling_type = args.dueling_type

            self.learn_start = args.learn_start
            self.batch_size = args.batch_size
            self.valid_size = args.valid_size
            self.eps_start = args.eps_start
            self.eps_end = args.eps_end
            self.eps_eval = args.eps_eval
            self.eps_decay = args.eps_decay
            self.target_model_update = args.target_model_update
            self.action_repetition = args.action_repetition
            self.memory_interval = args.memory_interval
            self.train_interval = args.train_interval
        elif args.agent_type == "a3c":
            self.enable_lstm = args.enable_lstm
            self.num_processes = args.num_processes

            self.rollout_steps = args.rollout_steps
            self.tau = args.tau

    def _reset_experience(self):
        self.experience = Experience(state0 = None,
                                     action = None,
                                     reward = None,
                                     state1 = None,
                                     terminal1 = False)

    def _load_model(self, model_file):
        if model_file:
            self.logger.warning("Loading Model: " + self.model_file + " ...")
            self.model.load_state_dict(torch.load(model_file))
            self.logger.warning("Loaded  Model: " + self.model_file + " ...")
        else:
            self.logger.warning("No Pretrained Model. Will Train From Scratch.")

    def _save_model(self, step, curr_reward):
        self.logger.warning("Saving Model    @ Step: " + str(step) + ": " + self.model_name + " ...")
        if self.save_best:
            if self.best_step is None:
                self.best_step   = step
                self.best_reward = curr_reward
            if curr_reward >= self.best_reward:
                self.best_step   = step
                self.best_reward = curr_reward
                torch.save(self.model.state_dict(), self.model_name)
            self.logger.warning("Saved  Model    @ Step: " + str(step) + ": " + self.model_name + ". {Best Step: " + str(self.best_step) + " | Best Reward: " + str(self.best_reward) + "}")
        else:
            torch.save(self.model.state_dict(), self.model_name)
            self.logger.warning("Saved  Model    @ Step: " + str(step) + ": " + self.model_name + ".")

    def _forward(self, observation):
        raise NotImplementedError("not implemented in base calss")

    def _backward(self, reward, terminal):
        raise NotImplementedError("not implemented in base calss")

    def _eval_model(self):  # evaluation during training
        raise NotImplementedError("not implemented in base calss")

    def fit_model(self):    # training
        raise NotImplementedError("not implemented in base calss")

    def test_model(self):   # testing pre-trained models
        raise NotImplementedError("not implemented in base calss")
