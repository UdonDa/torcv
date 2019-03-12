import sys
sys.path.append('../../')

import torcv.utils.debug.print_network as print_network
from torcv.links.model.drn.drn import drn_a_resnet50, drn_c_26, drn_c_42, drn_c_58, drn_d_22, drn_d_24, drn_d_38, drn_d_40, drn_d_54, drn_d_105
from torcv.links.model.deeplabv3plus.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

import torch
import torch.nn as nn

if __name__ == '__main__':
    model = drn_a_resnet50(pretrained=False, BatchNorm=nn.BatchNorm2d)
    
    model = drn_c_26(pretrained=False, BatchNorm=nn.BatchNorm2d)
    model = drn_c_42(pretrained=False, BatchNorm=nn.BatchNorm2d)
    model = drn_c_58(pretrained=False, BatchNorm=nn.BatchNorm2d)

    model = drn_d_22(pretrained=False, BatchNorm=nn.BatchNorm2d)
    model = drn_d_24(pretrained=False, BatchNorm=nn.BatchNorm2d)
    model = drn_d_38(pretrained=False, BatchNorm=SynchronizedBatchNorm2d)
    model = drn_d_40(pretrained=False, BatchNorm=SynchronizedBatchNorm2d)
    model = drn_d_54(pretrained=False, BatchNorm=SynchronizedBatchNorm2d)
    model = drn_d_105(pretrained=False, BatchNorm=SynchronizedBatchNorm2d)
        # -> [1, 512, 64, 64], [1, 256, 128, 128]

    # print_network(model, 'xception model')

    input = torch.randn(1, 3, 512, 512)
    output, low_level_feat = model(input)
    print('output.size(): ', output.size()) # -> [1, 2048, 32, 32]
    print('low_level_feat.size(): ', low_level_feat.size()) # -> [1, 128, 128, 128]