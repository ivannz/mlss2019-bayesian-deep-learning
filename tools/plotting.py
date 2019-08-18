import matplotlib.pyplot as plt


def darker(color, a=0.5):
    """Adapted from this stackoverflow question_.

    .. _question: https://stackoverflow.com/questions/37765197/
    """
    from matplotlib.colors import to_rgb
    from colorsys import rgb_to_hls, hls_to_rgb

    h, l, s = rgb_to_hls(*to_rgb(color))
    return hls_to_rgb(h, max(0, min(a * l, 1)), s)


def canvas1d(*, figsize=(12, 5)):
    """Setup canvas for 1d function plot."""
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    fig.patch.set_alpha(1.0)
    ax.set_xlim(-7, +7) ; ax.set_ylim(-7, +9)

    return fig, ax


def plot1d(X, y, bands, ax=None, **kwargs):
    assert y.ndim == 2 and X.ndim == 1
    ax = plt.gca() if ax is None else ax

    # plot the predictive mean with the specified colour
    y_mean, y_std = y.mean(axis=-1), y.std(axis=-1)
    line, = ax.plot(X, y_mean, **kwargs)

    # plot paths or bands with a lighter color and slightly behind the mean
    color, zorder = darker(line.get_color(), 1.25), line.get_zorder()
    if bands is None:
        ax.plot(X, y, c=color, alpha=0.08, zorder=zorder - 1)

    else:
        for band in sorted(bands):
            ax.fill_between(X, y_mean + band * y_std, y_mean - band * y_std,
                            color=color, alpha=0.4 / len(bands), zorder=zorder - 1)

    return line


def plot1d_bands(X, y, ax=None, **kwargs):
    return plot1d(X, y, bands=(0.5, 1.0, 1.5, 2.0), ax=ax, **kwargs)


def plot1d_paths(X, y, ax=None, **kwargs):
    return plot1d(X, y, bands=None, ax=ax, **kwargs)
