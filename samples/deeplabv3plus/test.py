import sys
sys.path.append('../../')

import torch
import torcv.utils.debug.print_network as print_network
from torcv.links.model.deeplabv3plus.deeplabv3plus import deeplabV3plus
from torcv.links.model.vgg.vgg import vgg19_bn


if __name__ == '__main__':
    # model = deeplabV3plus(backbone='mobilenet', output_stride=16, num_classes=21)
    model = deeplabV3plus(backbone='drn_c_58', output_stride=16, num_classes=21)
    # model = deeplabV3plus(backbone='xception', output_stride=16, num_classes=21)
    
    input = torch.randn(1, 3, 513, 513)
    output = model(input)
    print(output.size()) # -> [1, 21, 513, 513]