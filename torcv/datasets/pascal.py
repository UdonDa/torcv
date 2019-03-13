import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

from torch.utils.data import DataLoader


def trainsforms_default(sample):
    t = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        transforms.ToTensor()
    ])
    return t(sample)
    




class VOCSegmentation(Dataset):
    """Pascal VOC dataset"""
    NUM_CLASSES = 21

    def __init__(self, args, base_dir=None, split='train', transforms_train=transforms_default):
        """
        Args:
            base_dir (str): path to VOC dataset directory.
            split (str): train/val.
            train_transforms (transform): transfrom to apply.
        """
        super().__init__()
        self.base_dir = base_dir
        self.image_dir = os.path.join(self.base_dir, 'JPEGImages')
        self.cat_dir = os.path.join(self.base_dir, 'SegmentationClass')


        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        self.args = args
        self.transforms_train = transforms_train

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
                return self.transforms_train(sample)
            elif split == 'val':
                return self.transforms_val(sample)

    
    def make_img_gt_point_pair(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.categories[index])

        return img, target

    def transforms_val(self, sample):
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