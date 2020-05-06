
import os
import torch
from torchvision import datasets, transforms

# ---------------------- Global settings ----------------------
DATASET_PATH = {
    'market1501': './Market-1501-v15.09.15/pytorch'
}


def getDataLoader(dataset, batch_size, part, shuffle=True, augment=True):
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

    assert part in {'train', 'query', 'gallery'}, 'part not in folders'
    image_dataset = datasets.ImageFolder(os.path.join(DATASET_PATH[dataset], part),
                                         data_transform)

    dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=batch_size,
                                             shuffle=shuffle, num_workers=4)

    return dataloader


def check_data(images, fids, img_save_path):
    """
    check data of image
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import torchvision.utils as vutils

    # [weight, hight]
    plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.title(fids)
    plt.imshow(np.transpose(vutils.make_grid(images,
                                             padding=2,
                                             normalize=True),
                            (1, 2, 0)))
    plt.savefig(img_save_path)


if __name__ == "__main__":
    dataloader = getDataLoader('market1501', 3, 'train')
    data_t = next(iter(dataloader))
    imgs, fids = data_t
    print(imgs.shape)
    print(fids.shape)
    # check train data
    check_data(imgs, fids, './experiments/check_train_data.jpg')
