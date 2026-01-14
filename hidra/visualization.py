"""
Visualization utilities for HiDRA embeddings.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


def plot_embedding(Y, labels=None, title=None, ax=None, cmap='viridis',
                   point_size=10, alpha=0.7, colorbar=True):
    """
    Plot a 2D embedding.

    Parameters
    ----------
    Y : ndarray of shape (n_samples, 2)
        2D embedding coordinates.
    labels : ndarray or None
        Point labels for coloring.
    title : str or None
        Plot title.
    ax : matplotlib.axes.Axes or None
        Axes to plot on. Creates new figure if None.
    cmap : str, default='viridis'
        Colormap name.
    point_size : float, default=10
        Size of scatter points.
    alpha : float, default=0.7
        Point transparency.
    colorbar : bool, default=True
        Whether to show colorbar.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes with the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    if labels is None:
        labels = np.zeros(len(Y))

    scatter = ax.scatter(Y[:, 0], Y[:, 1], c=labels, cmap=cmap,
                         s=point_size, alpha=alpha)

    if colorbar and len(np.unique(labels)) > 1:
        plt.colorbar(scatter, ax=ax)

    ax.set_xticks([])
    ax.set_yticks([])

    if title:
        ax.set_title(title)

    return ax


def plot_embedding_with_uncertainty(Y, uncertainty, labels=None, title=None,
                                     figsize=(12, 5)):
    """
    Plot embedding colored by labels and by uncertainty side-by-side.

    Parameters
    ----------
    Y : ndarray of shape (n_samples, 2)
        2D embedding coordinates.
    uncertainty : ndarray of shape (n_samples,)
        Uncertainty values per point.
    labels : ndarray or None
        Point labels for coloring.
    title : str or None
        Overall title.
    figsize : tuple, default=(12, 5)
        Figure size.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    if labels is None:
        labels = np.zeros(len(Y))

    # Plot by labels
    scatter1 = ax1.scatter(Y[:, 0], Y[:, 1], c=labels, cmap='viridis',
                           s=10, alpha=0.7)
    ax1.set_title('Colored by Labels')
    ax1.set_xticks([])
    ax1.set_yticks([])
    plt.colorbar(scatter1, ax=ax1)

    # Plot by uncertainty
    scatter2 = ax2.scatter(Y[:, 0], Y[:, 1], c=uncertainty, cmap='Reds',
                           s=10, alpha=0.7)
    ax2.set_title('Colored by Uncertainty')
    ax2.set_xticks([])
    ax2.set_yticks([])
    plt.colorbar(scatter2, ax=ax2, label='Uncertainty')

    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()
    return fig


def plot_comparison(embeddings, labels=None, titles=None, figsize=None):
    """
    Plot multiple embeddings side by side for comparison.

    Parameters
    ----------
    embeddings : dict or list
        Dictionary {name: Y} or list of embeddings.
    labels : ndarray or None
        Point labels for coloring.
    titles : list or None
        Titles for each subplot.
    figsize : tuple or None
        Figure size.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    """
    if isinstance(embeddings, dict):
        names = list(embeddings.keys())
        embeds = list(embeddings.values())
    else:
        names = [f'Embedding {i+1}' for i in range(len(embeddings))]
        embeds = embeddings

    n = len(embeds)
    if figsize is None:
        figsize = (5 * n, 5)

    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]

    if titles is None:
        titles = names

    for ax, Y, title in zip(axes, embeds, titles):
        plot_embedding(Y, labels=labels, title=title, ax=ax, colorbar=False)

    plt.tight_layout()
    return fig


def plot_loss_history(loss_history, title='Optimization Loss', ax=None):
    """
    Plot the optimization loss history.

    Parameters
    ----------
    loss_history : list
        List of loss values per iteration.
    title : str, default='Optimization Loss'
        Plot title.
    ax : matplotlib.axes.Axes or None
        Axes to plot on.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes with the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(loss_history, linewidth=1.5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    return ax


def plot_metrics_comparison(metrics_dict, metric_names=None, figsize=(10, 6)):
    """
    Create a bar chart comparing metrics across methods.

    Parameters
    ----------
    metrics_dict : dict
        {method_name: {metric_name: value}}
    metric_names : list or None
        Metrics to include. If None, uses all.
    figsize : tuple, default=(10, 6)
        Figure size.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    """
    methods = list(metrics_dict.keys())

    if metric_names is None:
        metric_names = list(metrics_dict[methods[0]].keys())

    n_metrics = len(metric_names)
    n_methods = len(methods)

    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]

    x = np.arange(n_methods)
    colors = plt.cm.Set2(np.linspace(0, 1, n_methods))

    for ax, metric in zip(axes, metric_names):
        values = [metrics_dict[m][metric] for m in methods]
        bars = ax.bar(x, values, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.set_title(metric)
        ax.grid(True, axis='y', alpha=0.3)

        # Highlight best
        best_idx = np.argmax(values) if 'Distortion' not in metric else np.argmin(values)
        bars[best_idx].set_edgecolor('red')
        bars[best_idx].set_linewidth(3)

    plt.tight_layout()
    return fig


def plot_density_scatter(Y, labels=None, title=None, ax=None, bins=50):
    """
    Plot embedding as a density scatter plot.

    Parameters
    ----------
    Y : ndarray of shape (n_samples, 2)
        2D embedding coordinates.
    labels : ndarray or None
        Not used, for API consistency.
    title : str or None
        Plot title.
    ax : matplotlib.axes.Axes or None
        Axes to plot on.
    bins : int, default=50
        Number of bins for density estimation.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes with the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    # Compute 2D histogram
    h, xedges, yedges = np.histogram2d(Y[:, 0], Y[:, 1], bins=bins)

    # Plot as image
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im = ax.imshow(h.T, extent=extent, origin='lower', cmap='viridis',
                   aspect='auto', interpolation='gaussian')

    plt.colorbar(im, ax=ax, label='Density')

    ax.set_xticks([])
    ax.set_yticks([])

    if title:
        ax.set_title(title)

    return ax
