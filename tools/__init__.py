import torch

from torch.utils.data import TensorDataset


def dataset_from_numpy(*ndarrays, dtype=None, device=None):
    """Creates :class:`TensorDataset` from the passed :class:`numpy.ndarray`-s.

    Each returned tensor in the TensorDataset and :attr:`ndarray` share
    the same memory, unless a type cast or device transfer took place.
    Modifications to any tensor in the dataset will be reflected in respective
    :attr:`ndarray` and vice versa.

    Each returned tensor in the dataset is not resizable.

    See Also
    --------
    torch.from_numpy : create a tensor from an ndarray.
    """
    tensors = map(torch.from_numpy, ndarrays)

    return TensorDataset(*[t.to(device, dtype) for t in tensors])
