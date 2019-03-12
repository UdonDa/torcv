import sys
sys.path.append('../../')

import torcv.utils.debug.print_network as print_network
from torcv.links.model.vgg.vgg import vgg16_bn
import torch


if __name__ == '__main__':
    model = vgg16_bn(pretrained=False)
    # print_network(model, 'vgg16_bn')

    input = torch.randn(1,3,224,224)
    output, feature = model(input)
