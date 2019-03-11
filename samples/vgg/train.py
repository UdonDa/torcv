import sys
sys.path.append('../../')

from torcv.links.model.vgg.vgg import vgg19_bn
import torcv.utils.debug.save_image as save_image
from torcv.solver.cifar10.cifar10_solver import Ciffar10_Solver
from torcv.dataset.cifar10 import get_cifar10


import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T

import argparse
import matplotlib.pyplot as plt
import numpy as np


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def main(config):
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

    train_loader, test_loader = get_cifar10(download=False, transform=transform)
    model = vgg19_bn(pretrained=False)

    solver = Ciffar10_Solver(config, train_loader=train_loader, test_loader=test_loader, num_classes=10, model=model)
    solver.train()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)

    parser.add_argument('--total_epochs', type=float, default=10)

    parser.add_argument('--gpu_number', type=str, default='0')

    config = parser.parse_args()
    main(config)