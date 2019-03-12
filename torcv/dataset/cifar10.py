import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T


def get_cifar10(config = None, download=True, transform=None):
    """To get train and test dataloader of cifar10
    Args:
        config (argparse): Conclude batch_size.
        download (bool): If True, download cifar10 dataset into `./data`
        transform (torchvision.transform): Do data augmentation.
    """
    train_set = torchvision.datasets.CIFAR10(root='../../data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=4, drop_last=True, pin_memory=True)

    test_set = torchvision.datasets.CIFAR10(root='../../data', train=False, download=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=config.batch_size, shuffle=True, num_workers=4)

    return train_loader, test_loader
