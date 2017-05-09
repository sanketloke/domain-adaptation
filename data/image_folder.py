################################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
################################################################################

import torch.utils.data as data
from random import shuffle

from PIL import Image
import os
import os.path
from math import floor

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir,sort=True):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    q=sorted(os.walk(dir))
    for root, _, fnames in q:
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images


def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader,sort=True,split_ratio=1):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        train_size = int(floor(split_ratio*len(imgs)))

        self.root = root
        self.imgs = imgs[:train_size]
        if not sort:
            imgs=shuffle(imgs)
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __shuffle__(self):
        self.imgs = shuffle(self.imgs)

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return (img, path)
        else:
            return img

    def __len__(self):
        return len(self.imgs)
