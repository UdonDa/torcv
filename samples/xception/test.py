import sys
sys.path.append('../../')

import torcv.utils.debug.print_network as print_network
from torcv.links.model.xception.xception import aligned_xception

import torch

if __name__ == '__main__':
    model = aligned_xception(pretrained=False)
    # print_network(model, 'xception model')

    input = torch.randn(1, 3, 512, 512)
    output, low_level_feat = model(input)
    print('output.size(): ', output.size()) # -> [1, 2048, 32, 32]
    print('low_level_feat.size(): ', low_level_feat.size()) # -> [1, 128, 128, 128]