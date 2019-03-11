import sys
sys.path.append('../../')

import torcv.utils.debug.print_network as print_network
from torcv.links.model.inception.inception import inception_v3


if __name__ == '__main__':
    model = inception_v3(pretrained=False)
    print_network(model, 'InceptionV3')