import sys
sys.path.append('../../')

import torcv.utils.debug.print_network as print_network
from torcv.links.model.squeezenet.squeezenet import squeezenet1_0, squeezenet1_1


if __name__ == '__main__':
    model = squeezenet1_0(pretrained=False)
    model = squeezenet1_1(pretrained=False)

    import torch
    input = torch.randn(1,3,224,224)
    output, feature = model(input)
    print(output.size()) # -> [1, 1000]
    print(feature.size()) # -> [1, 512, 13, 13]