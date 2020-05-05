import random
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
import torch
from torchvision import datasets, transforms
from collections import OrderedDict
import matplotlib
matplotlib.use('agg')

DATASET_PATH = {
    'market1501': './Market-1501-v15.09.15/pytorch'
    }

def getMarket1501DataLoader(dataset, batch_size, part, shuffle=True, augment=True):
    """Return the dataloader and imageset of the given dataset

    Arguments:
        dataset {string} -- name of the dataset: [market1501, duke, cuhk03]
        batch_size {int} -- the batch size to load
        part {string} -- which part of the dataset: [train, query, gallery]

    Returns:
        (torch.utils.data.DataLoader, torchvision.datasets.ImageFolder) -- the data loader and the image set
    """

    transform_list = [
        transforms.Resize(size=(384, 128), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

    if augment:
        transform_list.insert(1, transforms.RandomHorizontalFlip())

    data_transform = transforms.Compose(transform_list)

    image_dataset = datasets.ImageFolder(os.path.join(DATASET_PATH[dataset], part),
                                         data_transform)

    dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=batch_size,
                                             shuffle=shuffle, num_workers=4)

    return dataloader
