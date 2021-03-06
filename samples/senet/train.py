import sys
sys.path.append('../../')

from torcv.links.model.senet.senet import senet154, se_resnet101, se_resnet50, se_resnet152, se_resnext50_32x4d, se_resnext101_32x4d
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
    model = senet154()

    solver = Ciffar10_Solver(config, train_loader=train_loader, test_loader=test_loader, model=model)
    solver.train()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)

    parser.add_argument('--total_epochs', type=float, default=300)

    parser.add_argument('--gpu_number', type=str, default='0')

    config = parser.parse_args()
    main(config)