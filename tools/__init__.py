import tqdm
import torch

import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader


def dataset_from_numpy(*ndarrays, device=None, dtype=torch.float32):
    """Create :class:`TensorDataset` from the passed :class:`numpy.ndarray`-s.

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


default_criteria = {
    "nll": lambda model, X, y: F.nll_loss(model(X), y, reduction="mean"),
    "mse": lambda model, X, y: 0.5 * F.mse_loss(model(X), y, reduction="mean"),
}


def fit(model, dataset, criterion="nll", batch_size=32,
        n_epochs=1, weight_decay=0, verbose=False):
    """Fit the model with SGD (Adam) on the specified dataset and criterion.

    This bare minimum of a fit loop creates a minibatch generator from
    the `dataset` with batches of size `batch_size`. On each batch it
    computes the backward pass through the `criterion` and the `model`
    and updates the `model`-s parameters with the Adam optimizer step.
    The loop passes through the dataset `n_epochs` times. It does not
    output any running debugging information, except for a progress bar.

    The criterion can be either "mse" for mean sqaured error, "nll" for
    negative loglikelihood (categorical), or a callable taking `model, X, y`
    as arguments.
    """
    criterion = default_criteria.get(criterion, criterion)
    assert callable(criterion)

    # get the model's device
    device = next(model.parameters()).device

    # an optimizer for model's parameters
    optim = torch.optim.Adam(model.parameters(), lr=2e-3,
                             weight_decay=weight_decay)

    # stochastic minibatch generator for the training loop
    feed = DataLoader(dataset, shuffle=True,
                                       batch_size=batch_size)
    for epoch in tqdm.tqdm(range(n_epochs), disable=not verbose):

        model.train()
        for X, y in feed:
            # forward pass through the criterion (batch-average loss)
            loss = criterion(model, X.to(device), y.to(device))

            # get gradients with backward pass
            optim.zero_grad()
            loss.backward()

            # SGD update
            optim.step()

    return model


def apply(model, dataset, batch_size=512):
    """Collect model's outputs on the dataset.

    This straightforward function switches the model into `evaluation`
    regime, computes the forward pass on the `dataset` (in batches of
    size `batch_size`) and stacks the results into a `cpu` tensor. It
    temporarily disables `autograd` to gain some speed-up.
    """
    model.eval()

    # get the model's device
    device = next(model.parameters()).device

    # batch generator for the evaluation loop
    feed = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # compute and collect the outputs
    with torch.no_grad():
        return torch.cat([
            model(X.to(device)).cpu() for X, *rest in feed
        ], dim=0)
