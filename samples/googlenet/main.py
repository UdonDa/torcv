import sys
sys.path.append('../../')

import torcv.utils.debug.print_network as print_network
import torcv.links.model.googlenet.googlenet as googlenet


if __name__ == '__main__':
    model = googlenet(pretrained=False)
    # print_network(model, 'google net')


    import torch
    input = torch.randn(1,3,224,224)
    aux1, aux2, x, feature = model(input)
    print(aux1.size())
    print(aux2.size())
    print(x.size()) # -> [1, 1000]
    print(feature.size()) # -> [1, 1920, 7, 7]