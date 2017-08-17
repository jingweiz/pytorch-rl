from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# NOTE: refer to: https://discuss.pytorch.org/t/adaptive-learning-rate/320/31
def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
