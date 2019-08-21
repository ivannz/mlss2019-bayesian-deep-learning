"""Handy plotting procedures for small 2d images."""
import numpy as np

from torch import Tensor
from math import sqrt


def get_dimensions(n_samples, height, width,
                   n_row=None, n_col=None, aspect=(16, 9)):
    """Get the dimensions that aesthetically conform to the aspect ratio."""
    if n_row is None and n_col is None:
        ratio = (width * aspect[1]) / (height * aspect[0])
        n_row = int(sqrt(n_samples * ratio))

    if n_row is None:
        n_row = (n_samples + n_col - 1) // n_col

    elif n_col is None:
        n_col = (n_samples + n_row - 1) // n_row

    return n_row, n_col


def setup_canvas(ax, height, width, n_row, n_col):
    """Setup the ticks and labels for the canvas."""
    # A pair of index arrays
    row_index, col_index = np.r_[:n_row], np.r_[:n_col]

    # Setup major ticks to the seams between images and disable labels
    ax.set_yticks((row_index[:-1] + 1) * height - 0.5, minor=False)
    ax.set_xticks((col_index[:-1] + 1) * width - 0.5, minor=False)

    ax.set_yticklabels([], minor=False)
    ax.set_xticklabels([], minor=False)

    # Set minor ticks so that they are exactly between the major ones
    ax.set_yticks((row_index + 0.5) * height, minor=True)
    ax.set_xticks((col_index + 0.5) * width, minor=True)

    # ... and make their labels into i-j coordinates
    ax.set_yticklabels([f"{i:d}" for i in row_index], minor=True)
    ax.set_xticklabels([f"{j:d}" for j in col_index], minor=True)

    # Orient tick marks outward
    ax.tick_params(axis="both", which="both", direction="out")
    return ax


def arrange(n_row, n_col, data, fill_value=0):
    """Create a grid and populate it with images."""
    n_samples, height, width, *color = data.shape
    grid = np.full((n_row * height, n_col * width, *color),
                   fill_value, dtype=data.dtype)

    for k in range(min(n_samples, n_col * n_row)):
        i, j = (k // n_col) * height, (k % n_col) * width
        grid[i:i + height, j:j + width] = data[k]

    return grid


def to_hwc(images, format):
    assert format in ("chw", "hwc"), f"Unrecognized format `{format}`."

    if images.ndim == 3:
        return images[:, np.newaxis]

    assert images.ndim == 4, f"Images must be Nx{'x'.join(format.upper())}."

    if format == "chw":
        return images.transpose(0, 2, 3, 1)

    elif format == "hwc":
        return images


def plot(ax, images, *, n_col=None, n_row=None, format="chw", **kwargs):
    """Plot images in the numpy array on the specified matplotlib Axis."""
    if isinstance(images, Tensor):
        images = images.data.cpu().numpy()

    images = to_hwc(images, format)

    n_samples, height, width, *color = images.shape

    n_row, n_col = get_dimensions(n_samples, height, width, n_row, n_col)
    ax = setup_canvas(ax, height, width, n_row, n_col)

    image = arrange(n_row, n_col, images)
    return ax.imshow(image.squeeze(), **kwargs, origin="upper")
