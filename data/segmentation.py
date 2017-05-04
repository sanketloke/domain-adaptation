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

def make_dataset(dir,sort=True,silent=True):
    images = os.path.join(dir,'images')
    labels = os.path.join(dir,'labels')
    imageList=[]
    labelList=[]
    for filename in os.listdir(images):
        if os.path.isfile(os.path.join(labels,filename)) and is_image_file(os.path.join(labels,filename)):
            imageList.append(os.path.join(images,filename))
            labelList.append(os.path.join(labels,filename))
        else:
            if not silent:
                raise ValueError('File Missing')
    return imageList,labelList


def default_loader(path):
    return Image.open(path).convert('RGB')


class SegmentationDataset(data.Dataset):

    def __init__(self, root, transform=None,target_transform=None, split_ratio=1, return_paths=False,
                 loader=default_loader,sort=True):
        images,labels = make_dataset(root)
        if len(images) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.mode = 'train'
        self.images= images
        self.labels = labels
        from sklearn.utils import shuffle
        images, labels = shuffle(images, labels)


        train_size = int(floor(split_ratio*len(images)))
        self.image_train=  images[:train_size]
        self.image_test= images[train_size:]

        self. label_train=labels[:train_size]
        self.label_test = labels[train_size:]

        self.transform = transform
        self.target_transform=target_transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        if self.mode is 'train':
            image_path = self.image_train[index]
            label_path = self.label_train[index]
            image = self.loader(image_path)
            label = self.loader(label_path)
            if self.transform is not None:
                image = self.transform(image)
            if self.target_transform is not None:
                label = self.target_transform(label)

            if self.return_paths:
                return ((image, image_path) , (label,label_path))
            else:
                return (image,label)
        else:
            image_path = self.image_test[index]
            label_path = self.label_test[index]
            image = self.loader(image_path)
            label = self.loader(label_path)
            if self.transform is not None:
                image = self.transform(image)
            if self.target_transform is not None:
                label = self.target_transform(label)
            if self.return_paths:
                return ((image, image_path) , (label,label_path))
            else:
                return (image,label)


    def __len__(self):
        return len(self.image_train)

    def __len_train__(self):
        return len(self.image_train)

    def __len_test__(self):
        return len(self.image_test)

    def change_mode(self,flag):
        if flag==1:
            self.mode='train'
        else:
            self.mode ='test'

