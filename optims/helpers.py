from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# NOTE: refer to: https://discuss.pytorch.org/t/adaptive-learning-rate/320/31
def adapt_learning_rate(optimiser, lr):
  for param_group in optimiser.param_groups:
    param_group['lr'] = lr
