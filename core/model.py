from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils.init_weights import init_weights, normalized_columns_initializer

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        # logging
        self.logger = args.logger
        # params
        self.hidden_dim = args.hidden_dim
        self.use_cuda = args.use_cuda
        self.dtype = args.dtype
        # model_params
        if hasattr(args, "enable_dueling"):     # only set for "dqn"
            self.enable_dueling = args.enable_dueling
            self.dueling_type   = args.dueling_type
        if hasattr(args, "enable_lstm"):        # only set for "dqn"
            self.enable_lstm    = args.enable_lstm

        self.input_dims     = {}
        self.input_dims[0]  = args.hist_len # from params
        self.input_dims[1]  = args.state_shape
        self.output_dims    = args.action_dim

    def _init_weights(self):
        raise NotImplementedError("not implemented in base calss")

    def print_model(self):
        self.logger.warning("<-----------------------------------> Model")
        self.logger.warning(self)

    def _reset(self):           # NOTE: should be called at each child's __init__
        self._init_weights()
        self.type(self.dtype)   # put on gpu if possible
        self.print_model()

    def forward(self, input):
        raise NotImplementedError("not implemented in base calss")

class MlpModel(Model):
    def __init__(self, args):
        super(MlpModel, self).__init__(args)
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
                x = v + (a - a.mean(1).expand(x.size(0), self.output_dims))
            elif self.dueling_type == "max":    # Q(s,a)=V(s)+(A(s,a)-max_a(A(s,a)))
                x = v + (a - a.max(1)[0].expand(x.size(0), self.output_dims))
            elif self.dueling_type == "naive":  # Q(s,a)=V(s)+ A(s,a)
                x = v + a
            else:
                assert False, "dueling_type must be one of {'avg', 'max', 'naive'}"
            del v_ind_vb, a_ind_vb, v, a
            return x
        else:
            return self.fc4(x.view(x.size(0), -1))

class CnnModel(Model):
    def __init__(self, args):
        super(CnnModel, self).__init__(args)
        # 84x84
        # self.conv1 = nn.Conv2d(self.input_dims[0], 32, kernel_size=8, stride=4)
        # self.rl1   = nn.ReLU()
        # self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        # self.rl2   = nn.ReLU()
        # self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # self.rl3   = nn.ReLU()
        # self.fc4   = nn.Linear(64*7*7, self.hidden_dim)
        # self.rl4   = nn.ReLU()
        # 42x42
        self.conv1 = nn.Conv2d(self.input_dims[0], 32, kernel_size=3, stride=2)
        self.rl1   = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.rl2   = nn.ReLU()
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.rl3   = nn.ReLU()
        self.fc4   = nn.Linear(32*5*5, self.hidden_dim)
        self.rl4   = nn.ReLU()
        if self.enable_dueling: # [0]: V(s); [1,:]: A(s, a)
            self.fc5 = nn.Linear(self.hidden_dim, self.output_dims + 1)
            self.v_ind = torch.LongTensor(self.output_dims).fill_(0).unsqueeze(0)
            self.a_ind = torch.LongTensor(np.arange(1, self.output_dims + 1)).unsqueeze(0)
        else: # one q value output for each action
            self.fc5 = nn.Linear(self.hidden_dim, self.output_dims)

        self._reset()

    def _init_weights(self):
        self.apply(init_weights)
        self.fc4.weight.data = normalized_columns_initializer(self.fc4.weight.data, 0.0001)
        self.fc4.bias.data.fill_(0)
        self.fc5.weight.data = normalized_columns_initializer(self.fc5.weight.data, 0.0001)
        self.fc5.bias.data.fill_(0)

    def forward(self, x):
        x = x.view(x.size(0), self.input_dims[0], self.input_dims[1], self.input_dims[1])
        x = self.rl1(self.conv1(x))
        x = self.rl2(self.conv2(x))
        x = self.rl3(self.conv3(x))
        x = self.rl4(self.fc4(x.view(x.size(0), -1)))
        if self.enable_dueling:
            x = self.fc5(x)
            v_ind_vb = Variable(self.v_ind)
            a_ind_vb = Variable(self.a_ind)
            if self.use_cuda:
                v_ind_vb = v_ind_vb.cuda()
                a_ind_vb = a_ind_vb.cuda()
            v = x.gather(1, v_ind_vb.expand(x.size(0), self.output_dims))
            a = x.gather(1, a_ind_vb.expand(x.size(0), self.output_dims))
            # now calculate Q(s, a)
            if self.dueling_type == "avg":      # Q(s,a)=V(s)+(A(s,a)-avg_a(A(s,a)))
                x = v + (a - a.mean(1).expand(x.size(0), self.output_dims))
            elif self.dueling_type == "max":    # Q(s,a)=V(s)+(A(s,a)-max_a(A(s,a)))
                x = v + (a - a.max(1)[0].expand(x.size(0), self.output_dims))
            elif self.dueling_type == "naive":  # Q(s,a)=V(s)+ A(s,a)
                x = v + a
            else:
                assert False, "dueling_type must be one of {'avg', 'max', 'naive'}"
            del v_ind_vb, a_ind_vb, v, a
            return x
        else:
            return self.fc5(x.view(x.size(0), -1))

class A3CMlpModel(Model):
    def __init__(self, args):
        super(A3CMlpModel, self).__init__(args)
        # build model
        # 0. feature layers
        self.fc1 = nn.Linear(self.input_dims[0] * self.input_dims[1], self.hidden_dim) # NOTE: for pkg="gym"
        self.rl1 = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.rl2 = nn.ReLU()
        self.fc3 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.rl3 = nn.ReLU()
        self.fc4 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.rl4 = nn.ReLU()
        # lstm
        if self.enable_lstm:
            self.lstm  = nn.LSTMCell(self.hidden_dim, self.hidden_dim, 1)
        # 1. policy output
        self.policy_5   = nn.Linear(self.hidden_dim, self.output_dims)
        self.policy_6   = nn.Softmax()
        # 2. value output
        self.value_5    = nn.Linear(self.hidden_dim, 1)

        self._reset()

    def _init_weights(self):
        self.apply(init_weights)
        self.fc1.weight.data = normalized_columns_initializer(self.fc1.weight.data, 0.01)
        self.fc1.bias.data.fill_(0)
        self.fc2.weight.data = normalized_columns_initializer(self.fc2.weight.data, 0.01)
        self.fc2.bias.data.fill_(0)
        self.fc3.weight.data = normalized_columns_initializer(self.fc3.weight.data, 0.01)
        self.fc3.bias.data.fill_(0)
        self.fc4.weight.data = normalized_columns_initializer(self.fc4.weight.data, 0.01)
        self.fc4.bias.data.fill_(0)
        self.policy_5.weight.data = normalized_columns_initializer(self.policy_5.weight.data, 0.01)
        self.policy_5.bias.data.fill_(0)
        self.value_5.weight.data = normalized_columns_initializer(self.value_5.weight.data, 1.0)
        self.value_5.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

    def forward(self, x, lstm_hidden_vb=None):
        x = x.view(x.size(0), self.input_dims[0] * self.input_dims[1])
        x = self.rl1(self.fc1(x))
        x = self.rl2(self.fc2(x))
        x = self.rl3(self.fc3(x))
        x = self.rl4(self.fc4(x))
        x = x.view(-1, self.hidden_dim)
        if self.enable_lstm:
            x, c = self.lstm(x, lstm_hidden_vb)
        p = self.policy_5(x)
        p = self.policy_6(p)
        v = self.value_5(x)
        if self.enable_lstm:
            return p, v, (x, c)
        else:
            return p, v

class A3CCnnModel(Model):
    def __init__(self, args):
        super(A3CCnnModel, self).__init__(args)
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

class A3CMjcModel(Model):
    def __init__(self, args):
        super(A3CMjcModel, self).__init__(args)
        # build model
        # 0. feature layers
        self.fc1 = nn.Linear(self.input_dims[0] * self.input_dims[1], self.hidden_dim) # NOTE: for pkg="gym"
        self.rl1 = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.rl2 = nn.ReLU()
        self.fc3 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.rl3 = nn.ReLU()
        self.fc4 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.rl4 = nn.ReLU()


        self.fc1_v = nn.Linear(self.input_dims[0] * self.input_dims[1], self.hidden_dim) # NOTE: for pkg="gym"
        self.rl1_v = nn.ReLU()
        self.fc2_v = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.rl2_v = nn.ReLU()
        self.fc3_v = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.rl3_v = nn.ReLU()
        self.fc4_v = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.rl4_v = nn.ReLU()


        # lstm
        if self.enable_lstm:
            self.lstm  = nn.LSTMCell(self.hidden_dim, self.hidden_dim, 1)
            self.lstm_v  = nn.LSTMCell(self.hidden_dim, self.hidden_dim, 1)

        # 1. policy output
        self.policy_5   = nn.Linear(self.hidden_dim, self.output_dims)
        self.policy_sig = nn.Linear(self.hidden_dim, self.output_dims)
        self.softplus = nn.Softplus()
        # 2. value output
        self.value_5    = nn.Linear(self.hidden_dim, 1)

        self._reset()

    def _init_weights(self):
        self.apply(init_weights)
        self.fc1.weight.data = normalized_columns_initializer(self.fc1.weight.data, 0.01)
        self.fc1.bias.data.fill_(0)
        self.fc2.weight.data = normalized_columns_initializer(self.fc2.weight.data, 0.01)
        self.fc2.bias.data.fill_(0)
        self.fc3.weight.data = normalized_columns_initializer(self.fc3.weight.data, 0.01)
        self.fc3.bias.data.fill_(0)
        self.fc4.weight.data = normalized_columns_initializer(self.fc4.weight.data, 0.01)
        self.fc4.bias.data.fill_(0)
        self.policy_5.weight.data = normalized_columns_initializer(self.policy_5.weight.data, 0.01)
        self.policy_5.bias.data.fill_(0)
        self.value_5.weight.data = normalized_columns_initializer(self.value_5.weight.data, 1.0)
        self.value_5.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        self.lstm_v.bias_ih.data.fill_(0)
        self.lstm_v.bias_hh.data.fill_(0)

    def forward(self, x, lstm_hidden_vb=None):
        p = x.view(x.size(0), self.input_dims[0] * self.input_dims[1])
        p = self.rl1(self.fc1(p))
        p = self.rl2(self.fc2(p))
        p = self.rl3(self.fc3(p))
        p = self.rl4(self.fc4(p))
        p = p.view(-1, self.hidden_dim)
        if self.enable_lstm:
            p_, v_ = torch.split(lstm_hidden_vb[0],1)
            c_p, c_v = torch.split(lstm_hidden_vb[1],1)
            p, c_p = self.lstm(p, (p_, c_p))
        p_out = self.policy_5(p)
        sig = self.policy_sig(p)
        sig = self.softplus(sig)

        v = x.view(x.size(0), self.input_dims[0] * self.input_dims[1])
        v = self.rl1_v(self.fc1_v(v))
        v = self.rl2_v(self.fc2_v(v))
        v = self.rl3_v(self.fc3_v(v))
        v = self.rl4_v(self.fc4_v(v))
        v = v.view(-1, self.hidden_dim)
        if self.enable_lstm:
            v, c_v = self.lstm_v(v, (v_, c_v))
        v_out = self.value_5(v)

        if self.enable_lstm:
            return p_out, sig, v_out, (torch.cat((p,v),0), torch.cat((c_p, c_v),0))
        else:
            return p_out, sig, v_out
