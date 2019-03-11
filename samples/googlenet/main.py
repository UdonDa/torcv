import sys
sys.path.append('../../')

import torcv.utils.debug.print_network as print_network
import torcv.links.model.googlenet.googlenet as googlenet


if __name__ == '__main__':
    model = googlenet(pretrained=False)
    print_network(model, 'google net')