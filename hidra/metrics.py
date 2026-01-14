"""
Evaluation metrics for dimensionality reduction.
"""

import numpy as np
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors


def knn_recall(X_high, X_low, k=15):
    """
    Compute k-NN recall: fraction of high-dim neighbors preserved in low-dim.

    Parameters
    ----------
    X_high : ndarray of shape (n_samples, n_features)
        High-dimensional data.
    X_low : ndarray of shape (n_samples, n_components)
        Low-dimensional embedding.
    k : int, default=15
        Number of neighbors.

    Returns
    -------
    recall : float
        Mean k-NN recall across all points.
    """
    n_samples = X_high.shape[0]
    k = min(k, n_samples - 1)

    # Get k-NN in high-dim
    nn_high = NearestNeighbors(n_neighbors=k + 1).fit(X_high)
    _, idx_high = nn_high.kneighbors(X_high)
    idx_high = idx_high[:, 1:]  # Exclude self

    # Get k-NN in low-dim
    nn_low = NearestNeighbors(n_neighbors=k + 1).fit(X_low)
    _, idx_low = nn_low.kneighbors(X_low)
    idx_low = idx_low[:, 1:]

    # Compute recall
    recalls = []
    for i in range(n_samples):
        overlap = len(set(idx_high[i]) & set(idx_low[i]))
        recalls.append(overlap / k)

    return np.mean(recalls)


def spearman_correlation(X_high, X_low, n_samples=5000):
    """
    Compute Spearman correlation between high-dim and low-dim pairwise distances.

    Parameters
    ----------
    X_high : ndarray of shape (n_samples, n_features)
        High-dimensional data.
    X_low : ndarray of shape (n_samples, n_components)
        Low-dimensional embedding.
    n_samples : int, default=5000
        Number of pairs to sample (for efficiency).

    Returns
    -------
    rho : float
        Spearman correlation coefficient.
    """
    n = X_high.shape[0]
    n_pairs = min(n_samples, n * (n - 1) // 2)

    # Sample random pairs
    pairs = set()
    while len(pairs) < n_pairs:
        i, j = np.random.randint(0, n, 2)
        if i != j:
            pairs.add((min(i, j), max(i, j)))
    pairs = np.array(list(pairs))

    # Compute distances
    d_high = np.linalg.norm(X_high[pairs[:, 0]] - X_high[pairs[:, 1]], axis=1)
    d_low = np.linalg.norm(X_low[pairs[:, 0]] - X_low[pairs[:, 1]], axis=1)

    rho, _ = spearmanr(d_high, d_low)
    return rho


def distortion_ratio(X_high, X_low, n_samples=5000):
    """
    Compute distortion ratio: (max d_low/d_high) / (min d_low/d_high).

    Lower is better. A value of 1 means perfect distance preservation.

    Parameters
    ----------
    X_high : ndarray of shape (n_samples, n_features)
        High-dimensional data.
    X_low : ndarray of shape (n_samples, n_components)
        Low-dimensional embedding.
    n_samples : int, default=5000
        Number of pairs to sample.

    Returns
    -------
    ratio : float
        Distortion ratio.
    """
    n = X_high.shape[0]
    n_pairs = min(n_samples, n * (n - 1) // 2)

    # Sample random pairs
    pairs = set()
    while len(pairs) < n_pairs:
        i, j = np.random.randint(0, n, 2)
        if i != j:
            pairs.add((min(i, j), max(i, j)))
    pairs = np.array(list(pairs))

    # Compute distances
    eps = 1e-10
    d_high = np.linalg.norm(X_high[pairs[:, 0]] - X_high[pairs[:, 1]], axis=1) + eps
    d_low = np.linalg.norm(X_low[pairs[:, 0]] - X_low[pairs[:, 1]], axis=1) + eps

    ratios = d_low / d_high
    return ratios.max() / ratios.min()


def trustworthiness(X_high, X_low, k=15):
    """
    Compute trustworthiness: penalizes points that become false neighbors in embedding.

    Parameters
    ----------
    X_high : ndarray of shape (n_samples, n_features)
        High-dimensional data.
    X_low : ndarray of shape (n_samples, n_components)
        Low-dimensional embedding.
    k : int, default=15
        Number of neighbors.

    Returns
    -------
    T : float
        Trustworthiness score in [0, 1]. Higher is better.
    """
    n = X_high.shape[0]
    k = min(k, n - 1)

    # Get k-NN in high-dim and low-dim
    nn_high = NearestNeighbors(n_neighbors=n).fit(X_high)
    _, idx_high = nn_high.kneighbors(X_high)

    nn_low = NearestNeighbors(n_neighbors=k + 1).fit(X_low)
    _, idx_low = nn_low.kneighbors(X_low)
    idx_low = idx_low[:, 1:]

    # Compute ranks in high-dim
    ranks_high = np.zeros((n, n), dtype=int)
    for i in range(n):
        for rank, j in enumerate(idx_high[i]):
            ranks_high[i, j] = rank

    # Compute trustworthiness
    penalty = 0
    for i in range(n):
        neighbors_low = set(idx_low[i])
        neighbors_high = set(idx_high[i, 1:k+1])

        # Points in low-dim neighbors but not in high-dim neighbors
        false_neighbors = neighbors_low - neighbors_high
        for j in false_neighbors:
            penalty += ranks_high[i, j] - k

    max_penalty = n * k * (2 * n - 3 * k - 1) / 2
    if max_penalty == 0:
        return 1.0

    T = 1 - (2 / max_penalty) * penalty
    return max(0, T)


def continuity(X_high, X_low, k=15):
    """
    Compute continuity: penalizes points that were neighbors but got separated.

    Parameters
    ----------
    X_high : ndarray of shape (n_samples, n_features)
        High-dimensional data.
    X_low : ndarray of shape (n_samples, n_components)
        Low-dimensional embedding.
    k : int, default=15
        Number of neighbors.

    Returns
    -------
    C : float
        Continuity score in [0, 1]. Higher is better.
    """
    n = X_high.shape[0]
    k = min(k, n - 1)

    # Get k-NN in high-dim and low-dim
    nn_high = NearestNeighbors(n_neighbors=k + 1).fit(X_high)
    _, idx_high = nn_high.kneighbors(X_high)
    idx_high = idx_high[:, 1:]

    nn_low = NearestNeighbors(n_neighbors=n).fit(X_low)
    _, idx_low = nn_low.kneighbors(X_low)

    # Compute ranks in low-dim
    ranks_low = np.zeros((n, n), dtype=int)
    for i in range(n):
        for rank, j in enumerate(idx_low[i]):
            ranks_low[i, j] = rank

    # Compute continuity
    penalty = 0
    for i in range(n):
        neighbors_high = set(idx_high[i])
        neighbors_low = set(idx_low[i, 1:k+1])

        # Points in high-dim neighbors but not in low-dim neighbors
        missing_neighbors = neighbors_high - neighbors_low
        for j in missing_neighbors:
            penalty += ranks_low[i, j] - k

    max_penalty = n * k * (2 * n - 3 * k - 1) / 2
    if max_penalty == 0:
        return 1.0

    C = 1 - (2 / max_penalty) * penalty
    return max(0, C)


def evaluate_embedding(X_high, X_low, k=15):
    """
    Compute all metrics for an embedding.

    Parameters
    ----------
    X_high : ndarray of shape (n_samples, n_features)
        High-dimensional data.
    X_low : ndarray of shape (n_samples, n_components)
        Low-dimensional embedding.
    k : int, default=15
        Number of neighbors for local metrics.

    Returns
    -------
    metrics : dict
        Dictionary with all metric values.
    """
    return {
        'knn_recall': knn_recall(X_high, X_low, k),
        'spearman_rho': spearman_correlation(X_high, X_low),
        'distortion_ratio': distortion_ratio(X_high, X_low),
        'trustworthiness': trustworthiness(X_high, X_low, k),
        'continuity': continuity(X_high, X_low, k)
    }
