import sys
sys.path.append('../../')

import torcv.utils.debug.print_network as print_network
from torcv.links.model.vgg.vgg import vgg16_bn


if __name__ == '__main__':
    model = vgg16_bn(pretrained=False)
    print_network(model, 'vgg16_bn')