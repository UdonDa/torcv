import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms as T

from torch.utils.data import DataLoader


class VOCSegmentation(Dataset):
    """Pascal VOC dataset"""
    NUM_CLASSES = 21

    def __init__(self, args, base_dir=None, split='train'):
        """
        Args:
            base_dir (str): path to VOC dataset directory.
            split (str): train/val
            transform (transform): transfrom to apply
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

    
    def make_img_gt_point_pair(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.categories[index])

        return img, target

    def transform_val(self, sample):
        transforms = T.Compose([
            
        ])