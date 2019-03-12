# Ref: https://github.com/vacancy/Synchronized-BatchNorm-PyTorchfrom

from .batchnorm import SynchronizedBatchNorm1d, SynchronizedBatchNorm2d, SynchronizedBatchNorm3d
from .replicate import DataParallelWithCallback, patch_replication_callback