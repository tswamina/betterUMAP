"""
Utility functions for HiDRA.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors


def estimate_intrinsic_dimension(X, k=10, method='mle'):
    """
    Estimate the intrinsic dimensionality of the data.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input data.
    k : int, default=10
        Number of neighbors to use.
    method : str, default='mle'
        Method to use: 'mle' (maximum likelihood) or 'correlation'.

    Returns
    -------
    dim : float
        Estimated intrinsic dimension.
    """
    n_samples = X.shape[0]
    k = min(k, n_samples - 1)

    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    distances, _ = nn.kneighbors(X)
    distances = distances[:, 1:]  # Exclude self

    if method == 'mle':
        # Levina-Bickel MLE estimator
        eps = 1e-10
        log_ratios = np.log(distances[:, -1:] / (distances[:, :-1] + eps) + eps)
        dim = (k - 1) / np.sum(log_ratios, axis=1)
        return np.median(dim)

    elif method == 'correlation':
        # Correlation dimension estimator
        log_r = np.log(distances + 1e-10)
        log_k = np.log(np.arange(1, k + 1))
        slopes = []
        for i in range(n_samples):
            slope = np.polyfit(log_k, log_r[i], 1)[0]
            slopes.append(1 / slope if slope > 0 else 0)
        return np.median(slopes)

    else:
        raise ValueError(f"Unknown method: {method}")


def compute_neighborhood_overlap(X1, X2, k=15):
    """
    Compute the neighborhood overlap between two representations.

    Parameters
    ----------
    X1, X2 : ndarray
        Two representations of the same data.
    k : int, default=15
        Number of neighbors.

    Returns
    -------
    overlap : float
        Mean fraction of shared neighbors.
    """
    n_samples = X1.shape[0]
    k = min(k, n_samples - 1)

    nn1 = NearestNeighbors(n_neighbors=k + 1).fit(X1)
    nn2 = NearestNeighbors(n_neighbors=k + 1).fit(X2)

    _, idx1 = nn1.kneighbors(X1)
    _, idx2 = nn2.kneighbors(X2)

    idx1 = idx1[:, 1:]
    idx2 = idx2[:, 1:]

    overlaps = []
    for i in range(n_samples):
        shared = len(set(idx1[i]) & set(idx2[i]))
        overlaps.append(shared / k)

    return np.mean(overlaps)


def normalize_embedding(Y, method='standard'):
    """
    Normalize an embedding.

    Parameters
    ----------
    Y : ndarray of shape (n_samples, n_components)
        Embedding to normalize.
    method : str, default='standard'
        Normalization method: 'standard', 'minmax', or 'unit'.

    Returns
    -------
    Y_norm : ndarray
        Normalized embedding.
    """
    if method == 'standard':
        return (Y - Y.mean(axis=0)) / (Y.std(axis=0) + 1e-10)
    elif method == 'minmax':
        return (Y - Y.min(axis=0)) / (Y.max(axis=0) - Y.min(axis=0) + 1e-10)
    elif method == 'unit':
        norms = np.linalg.norm(Y, axis=1, keepdims=True)
        return Y / (norms + 1e-10)
    else:
        raise ValueError(f"Unknown method: {method}")


def align_embeddings(Y_ref, Y_target):
    """
    Align target embedding to reference using Procrustes analysis.

    Parameters
    ----------
    Y_ref : ndarray of shape (n_samples, n_components)
        Reference embedding.
    Y_target : ndarray of shape (n_samples, n_components)
        Target embedding to align.

    Returns
    -------
    Y_aligned : ndarray
        Aligned target embedding.
    """
    # Center both
    Y_ref_c = Y_ref - Y_ref.mean(axis=0)
    Y_target_c = Y_target - Y_target.mean(axis=0)

    # Compute optimal rotation
    H = Y_target_c.T @ Y_ref_c
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Apply rotation
    Y_aligned = Y_target_c @ R

    # Match scale
    scale = np.std(Y_ref_c) / (np.std(Y_aligned) + 1e-10)
    Y_aligned *= scale

    # Match center
    Y_aligned += Y_ref.mean(axis=0)

    return Y_aligned


def subsample_data(X, n_samples=1000, method='random', random_state=None):
    """
    Subsample large datasets for faster processing.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input data.
    n_samples : int, default=1000
        Number of samples to select.
    method : str, default='random'
        Subsampling method: 'random' or 'stratified_density'.
    random_state : int or None
        Random seed.

    Returns
    -------
    X_sub : ndarray
        Subsampled data.
    indices : ndarray
        Indices of selected samples.
    """
    if random_state is not None:
        np.random.seed(random_state)

    N = X.shape[0]
    if n_samples >= N:
        return X, np.arange(N)

    if method == 'random':
        indices = np.random.choice(N, size=n_samples, replace=False)

    elif method == 'stratified_density':
        # Stratify by local density
        nn = NearestNeighbors(n_neighbors=min(10, N - 1)).fit(X)
        distances, _ = nn.kneighbors(X)
        densities = 1.0 / (distances.mean(axis=1) + 1e-10)

        # Bin by density and sample from each bin
        n_bins = 10
        bins = np.percentile(densities, np.linspace(0, 100, n_bins + 1))
        bin_indices = np.digitize(densities, bins[1:-1])

        indices = []
        samples_per_bin = n_samples // n_bins
        for b in range(n_bins):
            bin_mask = bin_indices == b
            bin_idx = np.where(bin_mask)[0]
            if len(bin_idx) > 0:
                n_select = min(samples_per_bin, len(bin_idx))
                selected = np.random.choice(bin_idx, size=n_select, replace=False)
                indices.extend(selected)

        indices = np.array(indices)
        # Fill remaining if needed
        if len(indices) < n_samples:
            remaining = list(set(range(N)) - set(indices))
            additional = np.random.choice(remaining, size=n_samples - len(indices), replace=False)
            indices = np.concatenate([indices, additional])

    else:
        raise ValueError(f"Unknown method: {method}")

    return X[indices], indices
