import sys
sys.path.append('../../')

import torcv.utils.debug.print_network as print_network
from torcv.links.model.densenet.densenet import densenet161, densenet121, densenet169, densenet201


if __name__ == '__main__':
    model = densenet161(pretrained=False)
    model = densenet169(pretrained=False)
    model = densenet121(pretrained=False)
    model = densenet201(pretrained=False)
    
    import torch
    input = torch.randn(1,3,224,224)
    output, feature = model(input)
    print(output.size()) # -> [1, 1000]
    print(feature.size()) # -> [1, 1920, 7, 7]