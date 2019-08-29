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


def remove(indices, dataset):
    """Extract the specified samples from the dataset and remove."""
    mask = torch.zeros(len(dataset), dtype=torch.uint8)
    mask[indices] = True

    removed = TensorDataset(*dataset[mask])

    dataset.tensors = dataset[~mask]

    return removed


def merge(*datasets, out=None):
    # Classes derived from Dataset support appending via
    #  `+` (__add__), but this breaks slicing.

    data = [d.tensors for d in datasets if d is not None]
    tensors = [torch.cat(tup, dim=0) for tup in zip(*data)]

    if isinstance(out, TensorDataset):
        out.tensors = tensors
        return out

    return TensorDataset(*tensors)
