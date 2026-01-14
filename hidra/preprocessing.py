"""
Data preprocessing utilities for HiDRA.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold


class HiDRAPreprocessor:
    """
    Preprocessing pipeline for HiDRA.

    Combines scaling, dimensionality reduction, and feature selection
    into a single transformer.

    Parameters
    ----------
    scaling : str or None, default='standard'
        Scaling method: 'standard', 'minmax', 'robust', or None.
    pca_components : int, float, or None, default=None
        If int, number of PCA components. If float (0-1), variance to retain.
        If None, no PCA is applied.
    min_variance : float or None, default=None
        Minimum variance threshold for feature selection.
    """

    def __init__(self, scaling='standard', pca_components=None, min_variance=None):
        self.scaling = scaling
        self.pca_components = pca_components
        self.min_variance = min_variance

        self.scaler_ = None
        self.pca_ = None
        self.var_selector_ = None

    def fit(self, X):
        """Fit the preprocessor to data."""
        X = np.asarray(X, dtype=np.float64)

        # Feature selection by variance
        if self.min_variance is not None:
            self.var_selector_ = VarianceThreshold(threshold=self.min_variance)
            self.var_selector_.fit(X)
            X = self.var_selector_.transform(X)

        # Scaling
        if self.scaling == 'standard':
            self.scaler_ = StandardScaler()
        elif self.scaling == 'minmax':
            self.scaler_ = MinMaxScaler()
        elif self.scaling == 'robust':
            self.scaler_ = RobustScaler()
        elif self.scaling is not None:
            raise ValueError(f"Unknown scaling method: {self.scaling}")

        if self.scaler_ is not None:
            self.scaler_.fit(X)
            X = self.scaler_.transform(X)

        # PCA
        if self.pca_components is not None:
            self.pca_ = PCA(n_components=self.pca_components)
            self.pca_.fit(X)

        return self

    def transform(self, X):
        """Transform data using fitted preprocessor."""
        X = np.asarray(X, dtype=np.float64)

        if self.var_selector_ is not None:
            X = self.var_selector_.transform(X)

        if self.scaler_ is not None:
            X = self.scaler_.transform(X)

        if self.pca_ is not None:
            X = self.pca_.transform(X)

        return X

    def fit_transform(self, X):
        """Fit and transform data."""
        self.fit(X)
        return self.transform(X)

    def get_feature_info(self):
        """Get information about feature selection and reduction."""
        info = {}

        if self.var_selector_ is not None:
            info['features_selected'] = np.sum(self.var_selector_.get_support())
            info['features_removed'] = np.sum(~self.var_selector_.get_support())

        if self.pca_ is not None:
            info['pca_components'] = self.pca_.n_components_
            info['variance_explained'] = np.sum(self.pca_.explained_variance_ratio_)

        return info


def remove_outliers(X, method='iqr', threshold=1.5):
    """
    Remove outliers from data.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input data.
    method : str, default='iqr'
        Method: 'iqr' (interquartile range) or 'zscore'.
    threshold : float, default=1.5
        Threshold for outlier detection.

    Returns
    -------
    X_clean : ndarray
        Data with outliers removed.
    mask : ndarray of bool
        Boolean mask indicating kept samples.
    """
    if method == 'iqr':
        Q1 = np.percentile(X, 25, axis=0)
        Q3 = np.percentile(X, 75, axis=0)
        IQR = Q3 - Q1

        lower = Q1 - threshold * IQR
        upper = Q3 + threshold * IQR

        mask = np.all((X >= lower) & (X <= upper), axis=1)

    elif method == 'zscore':
        z_scores = np.abs((X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10))
        mask = np.all(z_scores < threshold, axis=1)

    else:
        raise ValueError(f"Unknown method: {method}")

    return X[mask], mask


def handle_missing_values(X, strategy='mean'):
    """
    Handle missing values in data.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input data with potential NaN values.
    strategy : str, default='mean'
        Strategy: 'mean', 'median', 'zero', or 'drop'.

    Returns
    -------
    X_filled : ndarray
        Data with missing values handled.
    """
    X = np.asarray(X, dtype=np.float64)

    if strategy == 'drop':
        mask = ~np.any(np.isnan(X), axis=1)
        return X[mask]

    if strategy == 'mean':
        fill_values = np.nanmean(X, axis=0)
    elif strategy == 'median':
        fill_values = np.nanmedian(X, axis=0)
    elif strategy == 'zero':
        fill_values = np.zeros(X.shape[1])
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    X_filled = X.copy()
    for j in range(X.shape[1]):
        mask = np.isnan(X_filled[:, j])
        X_filled[mask, j] = fill_values[j]

    return X_filled


def compute_distance_matrix(X, metric='euclidean', n_jobs=1):
    """
    Compute pairwise distance matrix.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input data.
    metric : str, default='euclidean'
        Distance metric: 'euclidean', 'manhattan', 'cosine', or 'correlation'.
    n_jobs : int, default=1
        Number of parallel jobs.

    Returns
    -------
    D : ndarray of shape (n_samples, n_samples)
        Distance matrix.
    """
    from scipy.spatial.distance import pdist, squareform

    if metric == 'euclidean':
        D = squareform(pdist(X, metric='euclidean'))
    elif metric == 'manhattan':
        D = squareform(pdist(X, metric='cityblock'))
    elif metric == 'cosine':
        D = squareform(pdist(X, metric='cosine'))
    elif metric == 'correlation':
        D = squareform(pdist(X, metric='correlation'))
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return D


def batch_generator(X, batch_size=1000, shuffle=True, random_state=None):
    """
    Generate batches of data for large-scale processing.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input data.
    batch_size : int, default=1000
        Size of each batch.
    shuffle : bool, default=True
        Whether to shuffle before batching.
    random_state : int or None
        Random seed.

    Yields
    ------
    X_batch : ndarray
        Batch of data.
    indices : ndarray
        Indices of batch samples.
    """
    n_samples = X.shape[0]
    indices = np.arange(n_samples)

    if shuffle:
        if random_state is not None:
            np.random.seed(random_state)
        np.random.shuffle(indices)

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch_idx = indices[start:end]
        yield X[batch_idx], batch_idx
