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
