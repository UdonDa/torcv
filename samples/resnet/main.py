import sys
sys.path.append('../../')

import torcv.utils.debug.print_network as print_network
from torcv.links.model.resnet.resnet import resnet152, resnet18


if __name__ == '__main__':
    model = resnet152(pretrained=False)
    # model = resnet18(pretrained=False)
    
    import torch
    input = torch.randn(1,3,224,224)
    output, feature = model(input)
    print(output.size()) # -> [1, 1000]
    print(feature.size()) # -> [1, 2048, 7, 7]