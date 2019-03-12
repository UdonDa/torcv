# Ref: https://github.com/vacancy/Synchronized-BatchNorm-PyTorch


import collections

import torch
import torch.nn as nn

from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.parallel._functions import ReduceAddCoalesced, Broadcast

from .comm import SyncMaster


__all__ = ['SynchronizedBatchNorm1d', 'SynchronizedBatchNorm2d', 'SynchronizedBatchNorm3d']
