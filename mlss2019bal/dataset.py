import numpy as np

import torch
from torch.utils.data import TensorDataset
from torchvision import datasets

from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split

from collections import namedtuple


ImageDataset = namedtuple("ImageDataset", ["data", "targets", "classes"])


def get_mnist(path="./data", train=True):
    dataset = datasets.MNIST(path, train=train, download=True)
    return ImageDataset((dataset.data.float().unsqueeze(1) - 33.3285) / 78.5655,
                        dataset.targets,
                        dataset.classes)


def get_kmnist(path="./data", train=True):
    dataset = datasets.KMNIST(path, train=train, download=True)
    return ImageDataset(dataset.data.float().unsqueeze(1) / 255.,
                        dataset.targets,
                        dataset.classes)


def get_dataset(n_train=20, n_valid=5000, random_state=None, name="MNIST", path="./data"):
    if name == "MNIST":
        dataset, holdout = get_mnist(path, train=True), get_mnist(path, train=False)

    elif name == "KMNIST":
        dataset, holdout = get_kmnist(path, train=True), get_kmnist(path, train=False)

    n_classes = len(dataset.classes)
    random_state = check_random_state(random_state)

    distribution = random_state.dirichlet([0.1] * n_classes)

    # create an imbalanced class label distribution for the train
    targets = dataset.targets.numpy()

    # split the dataset into validaton and everything else
    ix_rest, ix_valid = train_test_split(
        np.arange(len(targets)), stratify=targets, test_size=n_valid,
        shuffle=True, random_state=random_state)

    # get indices for each subsample
    indices = []
    for label, freq in enumerate(np.round(distribution * n_train)):
        ix = np.flatnonzero(targets[ix_rest] == label)
        indices.extend(ix[:int(freq)])

    ix_train, ix_pool = np.take(ix_rest, indices), np.delete(ix_rest, indices)

    # collect and split into datasets

    S_train = TensorDataset(dataset.data[ix_train], dataset.targets[ix_train])
    S_valid = TensorDataset(dataset.data[ix_valid], dataset.targets[ix_valid])
    S_pool = TensorDataset(dataset.data[ix_pool], dataset.targets[ix_pool])
    S_test = TensorDataset(holdout.data, holdout.targets)

    return S_train, S_pool, S_valid, S_test


def take(pool, indices):
    """Copy the specified samples from the pool."""
    # a binary mask of selected samples (duplicated indices)
    mask = torch.zeros(len(pool), dtype=torch.bool)
    mask[indices] = True

    return TensorDataset(*pool[mask])


def delete(pool, indices):
    """Drop the specified samples from the pool."""

    # mask out the selected samples
    mask = torch.ones(len(pool), dtype=torch.bool)
    mask[indices] = False

    return TensorDataset(*pool[mask])


def append(train, new):
    """Append new samples to the train dataset."""
    tensors = [
        torch.cat(pair, dim=0)
        for pair in zip(train.tensors, new.tensors)
    ]

    # Classes derived from Dataset support appending via
    #  `+` (__add__), but this breaks slicing.
    return TensorDataset(*tensors)
