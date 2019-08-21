import torch

from torch.utils.data import TensorDataset


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
