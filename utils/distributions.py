from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import random

# Knuth's algorithm for generating Poisson samples
def sample_poisson(lmbd):
    L, k, p = math.exp(-lmbd), 0, 1
    while p > L:
        k += 1
        p *= random.uniform(0, 1)
    return max(k - 1, 0)

# KL divergence k = DKL[ ref_distribution || input_distribution]
def categorical_kl_div(input_vb, ref_vb):
    """
    kl_div = \sum ref * (log(ref) - log(input))
    variables needed:
            input_vb: [batch_size x state_dim]
              ref_vb: [batch_size x state_dim]
    returns:
           kl_div_vb: [batch_size x 1]
    """
    return (ref_vb * (ref_vb.log() - input_vb.log())).sum(1, keepdim=True)

# import torch
# from torch.autograd import Variable
# import torch.nn.functional as F
# # input_vb = Variable(torch.Tensor([0.2, 0.8])).view(1, 2)
# # ref_vb   = Variable(torch.Tensor([0.3, 0.7])).view(1, 2)
# input_vb = Variable(torch.Tensor([0.0002, 0.9998])).view(1, 2)
# ref_vb   = Variable(torch.Tensor([0.3, 0.7])).view(1, 2)
# input_vb = Variable(torch.Tensor([0.2, 0.8, 0.5, 0.5, 0.7, 0.3])).view(3, 2)
# ref_vb   = Variable(torch.Tensor([0.3, 0.7, 0.5, 0.5, 0.1, 0.9])).view(3, 2)
# print(F.kl_div(input_vb.log(), ref_vb, size_average=False))
# print(kl_div(input_vb, ref_vb))
