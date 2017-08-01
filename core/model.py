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
