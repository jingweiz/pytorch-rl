from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils.init_weights import init_weights, normalized_columns_initializer
from core.model import Model

class ACERMlpDisModel(Model):
    def __init__(self, args):
        super(ACERMlpDisModel, self).__init__(args)
        # build model
        # 0. feature layers
        self.fc1 = nn.Linear(self.input_dims[0] * self.input_dims[1], self.hidden_dim)
        self.rl1 = nn.ReLU()

        # lstm
        if self.enable_lstm:
            self.lstm  = nn.LSTMCell(self.hidden_dim, self.hidden_dim)

        # 1. actor:  /pi_{/theta}(a_t | x_t)
        self.actor_2 = nn.Linear(self.hidden_dim, self.output_dims)
        self.actor_3 = nn.Softmax()
        # 2. critic: Q_{/theta_v}(x_t, a_t)
        self.critic_2 = nn.Linear(self.hidden_dim, self.output_dims)

        self._reset()

    def _init_weights(self):
        self.apply(init_weights)
        self.actor_2.weight.data = normalized_columns_initializer(self.actor_2.weight.data, 0.01)
        self.actor_2.bias.data.fill_(0)
        self.critic_2.weight.data = normalized_columns_initializer(self.critic_2.weight.data, 1.0)
        self.critic_2.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

    def forward(self, x, lstm_hidden_vb=None):
        x = x.view(x.size(0), self.input_dims[0] * self.input_dims[1])
        x = self.rl1(self.fc1(x))
        # x = x.view(-1, 3*3*32)
        if self.enable_lstm:
            x, c = self.lstm(x, lstm_hidden_vb)
        policy = self.actor_3(self.actor_2(x)).clamp(max=1-1e-6, min=1e-6) # TODO: max might not be necessary
        q = self.critic_2(x)
        v = (q * policy).sum(1, keepdim=True)   # expectation of Q under /pi
        if self.enable_lstm:
            return policy, q, v, (x, c)
        else:
            return policy, q, v
