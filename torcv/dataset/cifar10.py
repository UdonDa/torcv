import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T


def get_cifar10(download=True, transform=None):
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=download, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=True, num_workers=4, drop_last=True, pin_memory=True)

    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform,)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=16, shuffle=True, num_workers=4)

    return train_loader, test_loader
