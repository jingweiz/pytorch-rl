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

class DQNMlpModel(Model):
    def __init__(self, args):
        super(DQNMlpModel, self).__init__(args)
        # build model
        self.fc1 = nn.Linear(self.input_dims[0] * self.input_dims[1], self.hidden_dim)
        self.rl1 = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.rl2 = nn.ReLU()
        self.fc3 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.rl3 = nn.ReLU()
        if self.enable_dueling: # [0]: V(s); [1,:]: A(s, a)
            self.fc4 = nn.Linear(self.hidden_dim, self.output_dims + 1)
            self.v_ind = torch.LongTensor(self.output_dims).fill_(0).unsqueeze(0)
            self.a_ind = torch.LongTensor(np.arange(1, self.output_dims + 1)).unsqueeze(0)
        else: # one q value output for each action
            self.fc4 = nn.Linear(self.hidden_dim, self.output_dims)

        self._reset()

    def _init_weights(self):
        # self.apply(init_weights)
        # self.fc1.weight.data = normalized_columns_initializer(self.fc1.weight.data, 0.01)
        # self.fc1.bias.data.fill_(0)
        # self.fc2.weight.data = normalized_columns_initializer(self.fc2.weight.data, 0.01)
        # self.fc2.bias.data.fill_(0)
        # self.fc3.weight.data = normalized_columns_initializer(self.fc3.weight.data, 0.01)
        # self.fc3.bias.data.fill_(0)
        # self.fc4.weight.data = normalized_columns_initializer(self.fc4.weight.data, 0.01)
        # self.fc4.bias.data.fill_(0)
        pass

    def forward(self, x):
        x = x.view(x.size(0), self.input_dims[0] * self.input_dims[1])
        x = self.rl1(self.fc1(x))
        x = self.rl2(self.fc2(x))
        x = self.rl3(self.fc3(x))
        if self.enable_dueling:
            x = self.fc4(x.view(x.size(0), -1))
            v_ind_vb = Variable(self.v_ind)
            a_ind_vb = Variable(self.a_ind)
            if self.use_cuda:
                v_ind_vb = v_ind_vb.cuda()
                a_ind_vb = a_ind_vb.cuda()
            v = x.gather(1, v_ind_vb.expand(x.size(0), self.output_dims))
            a = x.gather(1, a_ind_vb.expand(x.size(0), self.output_dims))
            # now calculate Q(s, a)
            if self.dueling_type == "avg":      # Q(s,a)=V(s)+(A(s,a)-avg_a(A(s,a)))
                # x = v + (a - a.mean(1)).expand(x.size(0), self.output_dims)   # 0.1.12
                x = v + (a - a.mean(1, keepdim=True))                           # 0.2.0
            elif self.dueling_type == "max":    # Q(s,a)=V(s)+(A(s,a)-max_a(A(s,a)))
                # x = v + (a - a.max(1)[0]).expand(x.size(0), self.output_dims) # 0.1.12
                x = v + (a - a.max(1, keepdim=True)[0])                         # 0.2.0
            elif self.dueling_type == "naive":  # Q(s,a)=V(s)+ A(s,a)
                x = v + a
            else:
                assert False, "dueling_type must be one of {'avg', 'max', 'naive'}"
            del v_ind_vb, a_ind_vb, v, a
            return x
        else:
            return self.fc4(x.view(x.size(0), -1))
