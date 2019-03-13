import torch
import torch.nn as nn
import torchvision

import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

from torch.utils.data import DataLoader

__all__ = ['get_cifar10', 'get_pascalvoc']



def trainsforms_default(sample):
    t = transforms.Compose([
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        transforms.ToTensor()
    ])
    return t(sample)


def get_cifar10(args = None, download=True, transforms_train=None, transforms_test=trainsforms_default):
    """To get train and test dataloader of cifar10
    Args:
        args (argparse): Conclude batch_size.
        download (bool): If True, download cifar10 dataset into `./data`
        transform (torchvision.transform): Do data augmentation.
    """
    train_set = torchvision.datasets.CIFAR10(root='../../data', train=True, download=True, transform=transforms_train)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True, pin_memory=True)

    test_set = torchvision.datasets.CIFAR10(root='../../data', train=False, download=False, transform=transforms_test)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=True, num_workers=4)

    return train_loader, test_loader


def get_pascalvoc(args, base_dir=None, transforms_train=None, transforms_test=trainsforms_default):
    """To get train and val dataloader of Pascal VOC
    Args:
        args (argparse): Conclude batch_size.
        transforms_train (torchvision.transform): Do train data augmentation.
    Return:
        voc_train_loader (DataLoader): for train.
        voc_val_loader (DataLoader): for test.
        voc_test_loader (None): None
        num_class (int): the number of classes.
    """
    voc_train = VOCSegmentation(args, base_dir=base_dir, split='train', transform=transforms_train)
    voc_val = VOCSegmentation(args, base_dir=base_dir, split='val', transform=transforms_test)
    voc_test = None
    num_class = VOCSegmentation.NUM_CLASSES

    voc_train_loader = DataLoader(voc_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True, pin_memory=True)
    voc_val_loader = DataLoader(voc_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    voc_test_loader = None

    return voc_train_loader, voc_val_loader, voc_test_loader, num_class








class VOCSegmentation(Dataset):
    """Pascal VOC dataset"""
    NUM_CLASSES = 21

    def __init__(self, args, base_dir=None, split='train', transform=None):
        """
        Args:
            base_dir (str): path to VOC dataset directory.
            split (str): train/val.
            transform (transform): transfrom to apply.
        """
        super().__init__()
        self.base_dir = base_dir
        print('base_dir: ', self.base_dir)
        self.image_dir = os.path.join(self.base_dir, 'JPEGImages')
        self.cat_dir = os.path.join(self.base_dir, 'SegmentationClass')


        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        self.args = args
        self.transform = transform

        splits_dir = os.path.join(self.base_dir, 'ImageSets', 'Segmentation')

        self.im_ids = []
        self.images = []
        self.categories = []

        for splt in self.split:
            with open(os.path.join(os.path.join(splits_dir, splt + '.txt')), "r") as f:
                lines = f.read().splitlines()

            for ii, line in enumerate(lines):
                image = os.path.join(self.image_dir, line + ".jpg")
                cat = os.path.join(self.cat_dir, line + ".png")

                assert os.path.isfile(image)
                assert os.path.isfile(cat)

                self.im_ids.append(line)
                self.images.append(image)
                self.categories.append(cat)
        assert (len(self.images) == len(self.categories))

        print('Number of images in {}: {:d}'.format(split, len(self.images)))


    def __len__(self):
        return len(self.images)

    
    def __getitem__(self, index):
        img, target = self.make_img_gt_point_pair(index)
        sample = {'image': img, 'label': target}

        for split in self.split:
            if split == 'train':
                return self.transform(sample)
            elif split == 'val':
                return self.transform_val(sample)

    
    def make_img_gt_point_pair(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.categories[index])

        return img, target

    def transform_val(self, sample):
        t = transforms.Compose([
            FixScaleCrop(crop_size=self.args.crop_size),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            transforms.ToTensor()
        ])
        return t(sample)

    def __str__(self):
        return 'VOC2012(split=' + str(self.split) + ')'


class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        w, h = img.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img,
                'label': mask}