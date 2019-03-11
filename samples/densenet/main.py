import sys
sys.path.append('../../')

import torcv.utils.debug.print_network as print_network
from torcv.links.model.densenet.densenet import densenet161, densenet121, densenet169, densenet201


if __name__ == '__main__':
    model = densenet161(pretrained=False)
    model = densenet169(pretrained=False)
    model = densenet121(pretrained=False)
    model = densenet201(pretrained=False)
    # print_network(model, 'dense net ')