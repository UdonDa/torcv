import sys
sys.path.append('../../')

import torcv.utils.debug.print_network as print_network
from torcv.links.model.resnet.resnet import resnet152, resnet18


if __name__ == '__main__':
    model = resnet152(pretrained=False)
    model = resnet18(pretrained=False)
    print_network(model, 'vgg16_bn')