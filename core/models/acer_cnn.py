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

class ACERCnnModel(Model):
    def __init__(self, args):
        super(ACERCnnModel, self).__init__(args)
        # build model
        # 0. feature layers
        self.conv1 = nn.Conv2d(self.input_dims[0], 32, kernel_size=3, stride=2) # NOTE: for pkg="atari"
        self.rl1   = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.rl2   = nn.ReLU()
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.rl3   = nn.ReLU()
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.rl4   = nn.ReLU()
        if self.enable_lstm:
            self.lstm  = nn.LSTMCell(3*3*32, self.hidden_dim, 1)
        # 1. policy output
        self.policy_5 = nn.Linear(self.hidden_dim, self.output_dims)
        self.policy_6 = nn.Softmax()
        # 2. value output
        self.value_5  = nn.Linear(self.hidden_dim, 1)

        self._reset()

    def _init_weights(self):
        self.apply(init_weights)
        self.policy_5.weight.data = normalized_columns_initializer(self.policy_5.weight.data, 0.01)
        self.policy_5.bias.data.fill_(0)
        self.value_5.weight.data = normalized_columns_initializer(self.value_5.weight.data, 1.0)
        self.value_5.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

    def forward(self, x, lstm_hidden_vb=None):
        x = x.view(x.size(0), self.input_dims[0], self.input_dims[1], self.input_dims[1])
        x = self.rl1(self.conv1(x))
        x = self.rl2(self.conv2(x))
        x = self.rl3(self.conv3(x))
        x = self.rl4(self.conv4(x))
        x = x.view(-1, 3*3*32)
        if self.enable_lstm:
            x, c = self.lstm(x, lstm_hidden_vb)
        p = self.policy_5(x)
        p = self.policy_6(p)
        v = self.value_5(x)
        if self.enable_lstm:
            return p, v, (x, c)
        else:
            return p, v
