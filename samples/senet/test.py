import sys
sys.path.append('../../')

import torcv.utils.debug.print_network as print_network
from torcv.links.model.senet.senet import senet154, se_resnet50, se_resnet101, se_resnet152, se_resnext50_32x4d, se_resnext101_32x4d

import torch

if __name__ == '__main__':
    model = senet154(pretrained=None, num_classes=100)
    # model = se_resnet50()

    z = torch.randn(1,3,224,224)
    output, features = model(z)
    print(output.size()) # -> [1, num_classes]
    print(features.size()) # -> [1, 2048, 7, 7]