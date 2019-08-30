import numpy as np

import torch
from torch.utils.data import TensorDataset
from torchvision import datasets

from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split


def get_data(name, path="./data", train=True):
    if name == "MNIST":
        dataset = datasets.MNIST(path, train=train, download=True)
    elif name == "KMNIST":
        dataset = datasets.KMNIST(path, train=train, download=True)

    images = dataset.data.float().unsqueeze(1)
    return TensorDataset(images / 255., dataset.targets)


def get_dataset(n_train=20, n_valid=5000, n_pool=5000,
                name="MNIST", path="./data", random_state=None):
    random_state = check_random_state(random_state)

    dataset = get_data(name, path, train=True)
    S_test = get_data(name, path, train=False)

    # create an imbalanced class label distribution for the train
    targets = dataset.tensors[-1].cpu().numpy()

    # split the dataset into validaton and train
    ix_all = np.r_[:len(targets)]
    ix_train, ix_valid = train_test_split(
        ix_all, stratify=targets, shuffle=True,
        train_size=max(n_train, 1), test_size=max(n_valid, 1),
        random_state=random_state)

    if n_train < 1:
        ix_train = np.r_[:0]

    if n_valid < 1:
        ix_valid = np.r_[:0]

    ix_pool = np.delete(ix_all, np.r_[ix_train, ix_valid])
    if n_pool > 0:
        ix_pool = random_state.permutation(ix_pool)[:n_pool]

    S_train = TensorDataset(*dataset[ix_train])
    S_valid = TensorDataset(*dataset[ix_valid])
    S_pool = TensorDataset(*dataset[ix_pool])

    return S_train, S_pool, S_valid, S_test


def collect(indices, dataset):
    """Collect the specified samples from the dataset and remove."""
    assert len(dataset) > 0

    mask = torch.zeros(len(dataset), dtype=torch.uint8)
    mask[indices] = True

    collected = TensorDataset(*dataset[mask])

    dataset.tensors = dataset[~mask]

    return collected


def merge(*datasets, out=None):
    # Classes derived from Dataset support appending via
    #  `+` (__add__), but this breaks slicing.

    data = [d.tensors for d in datasets if d is not None and d.tensors]
    assert all(len(data[0]) == len(d) for d in data)

    tensors = [torch.cat(tup, dim=0) for tup in zip(*data)]

    if isinstance(out, TensorDataset):
        out.tensors = tensors
        return out

    return TensorDataset(*tensors)
