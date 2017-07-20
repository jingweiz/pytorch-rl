from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import os
import visdom
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils.helpers import loggerConfig
from utils.sharedAdam import SharedAdam

CONFIGS = [
# agent_type, env_type,    game,                       model_type, memory_type
[ "empty",    "gym",       "CartPole-v0",              "mlp",      "none"      ],  # 0
[ "dqn",      "gym",       "CartPole-v0",              "mlp",      "sequential"],  # 1
[ "dqn",      "atari-ram", "Pong-ram-v0",              "mlp",      "sequential"],  # 2
[ "dqn",      "atari",     "PongDeterministic-v3",     "cnn",      "sequential"],  # 3
[ "dqn",      "atari",     "BreakoutDeterministic-v3", "cnn",      "sequential"],  # 4
[ "a3c",      "atari",     "PongDeterministic-v3",     "a3c-cnn",  "none"      ],  # 5
[ "a3c",      "gym",       "InvertedPendulum-v1",      "a3c-mjc",  "none"      ]   # 6
]

class Params(object):   # NOTE: shared across all modules
    def __init__(self):
        self.verbose     = 0            # 0(warning) | 1(info) | 2(debug)

        # training signature
        self.machine     = "alienware"  # "machine_id"
        self.timestamp   = "17052100"   # "yymmdd##"
        # training configuration
        self.mode        = 1            # 1(train) | 2(test model_file)
        self.config      = 5

        self.seed        = 123
        self.render      = False        # whether render the window from the original envs or not
        self.visualize   = True         # whether do online plotting and stuff or not
        self.save_best   = False        # save model w/ highest reward if True, otherwise always save the latest model

        self.agent_type, self.env_type, self.game, self.model_type, self.memory_type = CONFIGS[self.config]

        if self.agent_type == "dqn":
            self.enable_double_dqn  = False
            self.enable_dueling     = False
            self.dueling_type       = 'avg' # avg | max | naive

            if self.env_type == "gym":
                self.hist_len       = 1
                self.hidden_dim     = 16
            else:
                self.hist_len       = 4
                self.hidden_dim     = 256

            self.use_cuda           = torch.cuda.is_available()
            self.dtype              = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        elif self.agent_type == "a3c":
            self.enable_lstm        = True
            if self.model_type == "a3c-mjc":    # NOTE: should be set to True when training Mujoco envs
                self.enable_continuous  = True
            else:
                self.enable_continuous  = False
            self.num_processes      = 8

            self.hist_len           = 1
            self.hidden_dim         = 128

            self.use_cuda           = False
            self.dtype              = torch.FloatTensor
        else:
            self.hist_len           = 1
            self.hidden_dim         = 256

            self.use_cuda           = torch.cuda.is_available()
            self.dtype              = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

        # prefix for model/log/visdom
        self.refs        = self.machine + "_" + self.timestamp # NOTE: using this as env for visdom
        self.root_dir    = os.getcwd()

        # model files
        # NOTE: will save the current model to model_name
        self.model_name  = self.root_dir + "/models/" + self.refs + ".pth"
        # NOTE: will load pretrained model_file if not None
        self.model_file  = None#self.root_dir + "/models/{TODO:FILL_IN_PRETAINED_MODEL_FILE}.pth"
        if self.mode == 2:
            self.model_file  = self.model_name  # NOTE: so only need to change self.mode to 2 to test the current training
            assert self.model_file is not None, "Pre-Trained model is None, Testing aborted!!!"
            self.refs = self.refs + "_test"     # NOTE: using this as env for visdom for testing, to avoid accidentally redraw on the training plots

        # logging configs
        self.log_name    = self.root_dir + "/logs/" + self.refs + ".log"
        self.logger      = loggerConfig(self.log_name, self.verbose)
        self.logger.warning("<===================================>")

        if self.visualize:
            self.vis = visdom.Visdom()
            self.logger.warning("bash$: python -m visdom.server")           # activate visdom server on bash
            self.logger.warning("http://localhost:8097/env/" + self.refs)   # open this address on browser

class EnvParams(Params):    # settings for simulation environment
    def __init__(self):
        super(EnvParams, self).__init__()

        if self.env_type == "gym":
            pass
        elif self.env_type == "atari-ram":
            pass
        elif self.env_type == "atari":
            self.hei_state = 42
            self.wid_state = 42
            self.preprocess_mode = 3    # 0(nothing) | 1(rgb2gray) | 2(rgb2y) | 3(crop&resize)
        elif self.env_type == "lab":
            pass
        elif self.env_type == "gazebo":
            self.hei_state = 60
            self.wid_state = 80
            self.preprocess_mode = 3  # 0(nothing) | 1(rgb2gray) | 2(rgb2y) | 3(crop&resize depth)
            self.img_encoding_type = "passthrough"
        else:
            assert False, "env_type must be: gym | atari-ram | atari | lab"

class ModelParams(Params):  # settings for network architecture
    def __init__(self):
        super(ModelParams, self).__init__()

        self.state_shape = None # NOTE: set in fit_model of inherited Agents
        self.action_dim  = None # NOTE: set in fit_model of inherited Agents

class MemoryParams(Params):     # settings for replay memory
    def __init__(self):
        super(MemoryParams, self).__init__()

        if self.env_type == "gym":
            self.memory_size = 50000
        elif self.env_type == "atari-ram":
            self.memory_size = 1000000
        elif self.env_type == "atari":
            self.memory_size = 1000000
        elif self.env_type == "lab":
            self.memory_size = 1000000
        else:
            self.memory_size = 1000000

class AgentParams(Params):  # hyperparameters for drl agents
    def __init__(self):
        super(AgentParams, self).__init__()

        # criteria and optimizer
        if self.agent_type == "dqn":
            self.value_criteria = F.smooth_l1_loss
            self.optim          = optim.Adam
            # self.optim          = optim.RMSprop
        elif self.agent_type == "a3c":
            self.value_criteria = nn.MSELoss()
            self.optim          = SharedAdam    # share momentum across learners
        else:
            self.value_criteria = F.smooth_l1_loss
            self.optim          = optim.Adam
        # hyperparameters
        if self.agent_type == "dqn" and self.env_type == "gym":
            self.steps               = 100000   # max #iterations
            self.early_stop          = None     # max #steps per episode
            self.gamma               = 0.99
            self.clip_grad           = 1.#np.inf
            self.lr                  = 0.0001
            self.eval_freq           = 2500     # NOTE: here means every this many steps
            self.eval_steps          = 1000
            self.prog_freq           = self.eval_freq
            self.test_nepisodes      = 1

            self.learn_start         = 500      # start update params after this many steps
            self.batch_size          = 32
            self.valid_size          = 250
            self.eps_start           = 1
            self.eps_end             = 0.3
            self.eps_eval            = 0.#0.05
            self.eps_decay           = 50000
            self.target_model_update = 1000#0.0001
            self.action_repetition   = 1
            self.memory_interval     = 1
            self.train_interval      = 1
        elif self.agent_type == "dqn" and self.env_type == "atari-ram" or \
             self.agent_type == "dqn" and self.env_type == "atari":
            self.steps               = 50000000 # max #iterations
            self.early_stop          = None     # max #steps per episode
            self.gamma               = 0.99
            self.clip_grad           = 40.#np.inf
            self.lr                  = 0.00025
            self.eval_freq           = 250000#12500    # NOTE: here means every this many steps
            self.eval_steps          = 125000#2500
            self.prog_freq           = 10000#self.eval_freq
            self.test_nepisodes      = 1

            self.learn_start         = 50000    # start update params after this many steps
            self.batch_size          = 32
            self.valid_size          = 500
            self.eps_start           = 1
            self.eps_end             = 0.1
            self.eps_eval            = 0.#0.05
            self.eps_decay           = 1000000
            self.target_model_update = 10000
            self.action_repetition   = 4
            self.memory_interval     = 1
            self.train_interval      = 4
        elif self.agent_type == "a3c" and self.env_type == "atari-ram" or \
             self.agent_type == "a3c" and (self.env_type == "atari" or self.env_type == "gym"):
            self.steps               = 20000000 # max #iterations
            self.early_stop          = None     # max #steps per episode
            self.gamma               = 0.99
            self.clip_grad           = 40.
            self.lr                  = 0.0001
            self.eval_freq           = 60       # NOTE: here means every this many seconds
            self.eval_steps          = 3000
            self.prog_freq           = self.eval_freq
            self.test_nepisodes      = 10

            self.rollout_steps       = 20       # max look-ahead steps in a single rollout
            self.tau                 = 1.
        else:
            self.steps               = 1000000  # max #iterations
            self.early_stop          = None     # max #steps per episode
            self.gamma               = 0.99
            self.clip_grad           = 1.#np.inf
            self.lr                  = 0.001
            self.eval_freq           = 2500     # NOTE: here means every this many steps
            self.eval_steps          = 1000
            self.prog_freq           = self.eval_freq
            self.test_nepisodes      = 10

            self.learn_start         = 25000    # start update params after this many steps
            self.batch_size          = 32
            self.valid_size          = 500
            self.eps_start           = 1
            self.eps_end             = 0.1
            self.eps_eval            = 0.#0.05
            self.eps_decay           = 50000
            self.target_model_update = 1000
            self.action_repetition   = 1
            self.memory_interval     = 1
            self.train_interval      = 4

            self.rollout_steps       = 20       # max look-ahead steps in a single rollout
            self.tau                 = 1.

        self.env_params    = EnvParams()
        self.model_params  = ModelParams()
        self.memory_params = MemoryParams()

class Options(Params):
    agent_params  = AgentParams()
