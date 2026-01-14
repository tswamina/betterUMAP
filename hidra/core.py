"""
HiDRA: Hierarchical Distance-Regularized Approximation

A dimensionality reduction method that explicitly preserves distances
and quantifies embedding uncertainty.
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import MDS


class HiDRA:
    """
    Hierarchical Distance-Regularized Approximation for dimensionality reduction.

    Parameters
    ----------
    n_components : int, default=2
        Number of dimensions in the embedding.
    n_neighbors : int, default=15
        Number of neighbors for k-NN graph.
    n_iter : int, default=450
        Total number of optimization iterations.
    learning_rate : float, default=1.0
        Learning rate for gradient descent.
    random_state : int or None, default=None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_components=2,
        n_neighbors=15,
        n_iter=450,
        learning_rate=1.0,
        random_state=None
    ):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.random_state = random_state

        self.embedding_ = None
        self.knn_graph_ = None

    def fit_transform(self, X):
        """
        Fit the model and return the embedding.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            High-dimensional input data.

        Returns
        -------
        Y : ndarray of shape (n_samples, n_components)
            Low-dimensional embedding.
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        n_samples = X.shape[0]
        k = min(self.n_neighbors, n_samples - 1)

        # Step 1: Build k-NN graph
        self.knn_graph_, knn_indices, knn_distances = self._build_knn_graph(X, k)

        # Step 2: Compute geodesic distances
        D_geo = self._compute_geodesic_distances(self.knn_graph_)

        # Step 3: Initialize with MDS on geodesic distances
        Y = self._initialize_embedding(D_geo)

        # Step 4: Sample pairs
        near_pairs = self._get_near_pairs(knn_indices)
        mid_pairs = self._get_mid_pairs(X, k)
        far_pairs = self._get_far_pairs(n_samples, knn_indices, k)

        # Precompute high-dimensional distances for distortion loss
        dist_sample_pairs = self._sample_distance_pairs(n_samples)
        hd_distances = self._compute_pairwise_distances(X, dist_sample_pairs)

        # Step 5: Optimize with phased weights
        Y = self._optimize(
            Y, X, near_pairs, mid_pairs, far_pairs,
            dist_sample_pairs, hd_distances
        )

        self.embedding_ = Y
        return Y

    def _build_knn_graph(self, X, k):
        """Build k-nearest neighbors graph."""
        nn = NearestNeighbors(n_neighbors=k + 1, algorithm='auto')
        nn.fit(X)
        distances, indices = nn.kneighbors(X)

        # Exclude self (first neighbor)
        distances = distances[:, 1:]
        indices = indices[:, 1:]

        # Build sparse adjacency matrix
        n_samples = X.shape[0]
        row_indices = np.repeat(np.arange(n_samples), k)
        col_indices = indices.flatten()
        data = distances.flatten()

        graph = csr_matrix((data, (row_indices, col_indices)), shape=(n_samples, n_samples))
        # Make symmetric
        graph = graph + graph.T
        graph.data = np.minimum.reduceat(
            np.concatenate([graph.data, graph.data]),
            np.arange(0, len(graph.data) * 2, 2)
        ) if len(graph.data) > 0 else graph.data

        return graph, indices, distances

    def _compute_geodesic_distances(self, graph):
        """Compute shortest path distances on k-NN graph."""
        D_geo = shortest_path(graph, directed=False)
        # Handle disconnected components
        D_geo[np.isinf(D_geo)] = D_geo[~np.isinf(D_geo)].max() * 2
        return D_geo

    def _initialize_embedding(self, D_geo):
        """Initialize embedding using classical MDS on geodesic distances."""
        mds = MDS(
            n_components=self.n_components,
            dissimilarity='precomputed',
            random_state=self.random_state,
            normalized_stress='auto'
        )
        Y = mds.fit_transform(D_geo)
        return Y

    def _get_near_pairs(self, knn_indices):
        """Get near pairs from k-NN indices."""
        n_samples, k = knn_indices.shape
        i_indices = np.repeat(np.arange(n_samples), k)
        j_indices = knn_indices.flatten()
        return np.column_stack([i_indices, j_indices])

    def _get_mid_pairs(self, X, k):
        """
        Get mid-near pairs: for each point, sample 6 random points,
        pick the 2nd closest.
        """
        n_samples = X.shape[0]
        n_mid = max(1, int(k * 0.5))  # MN_ratio = 0.5
        mid_pairs = []

        for i in range(n_samples):
            # Sample 6 random points per mid-pair needed
            for _ in range(n_mid):
                candidates = np.random.choice(n_samples, size=min(6, n_samples), replace=False)
                candidates = candidates[candidates != i]
                if len(candidates) < 2:
                    continue

                # Compute distances to candidates
                dists = np.linalg.norm(X[candidates] - X[i], axis=1)
                # Pick 2nd closest
                sorted_idx = np.argsort(dists)
                if len(sorted_idx) >= 2:
                    mid_pairs.append([i, candidates[sorted_idx[1]]])

        return np.array(mid_pairs) if mid_pairs else np.empty((0, 2), dtype=int)

    def _get_far_pairs(self, n_samples, knn_indices, k):
        """Get far pairs: random non-neighbors."""
        n_far = k * 2  # FP_ratio = 2
        far_pairs = []

        knn_set = [set(knn_indices[i]) for i in range(n_samples)]

        for i in range(n_samples):
            non_neighbors = list(set(range(n_samples)) - knn_set[i] - {i})
            if len(non_neighbors) > 0:
                selected = np.random.choice(
                    non_neighbors,
                    size=min(n_far, len(non_neighbors)),
                    replace=False
                )
                for j in selected:
                    far_pairs.append([i, j])

        return np.array(far_pairs) if far_pairs else np.empty((0, 2), dtype=int)

    def _sample_distance_pairs(self, n_samples, n_pairs=5000):
        """Sample random pairs for distortion loss."""
        n_pairs = min(n_pairs, n_samples * (n_samples - 1) // 2)
        pairs = set()
        while len(pairs) < n_pairs:
            i, j = np.random.randint(0, n_samples, 2)
            if i != j:
                pairs.add((min(i, j), max(i, j)))
        return np.array(list(pairs))

    def _compute_pairwise_distances(self, X, pairs):
        """Compute distances for given pairs."""
        return np.linalg.norm(X[pairs[:, 0]] - X[pairs[:, 1]], axis=1)

    def _get_phase_weights(self, iteration):
        """Get weights for current phase."""
        if iteration < 100:
            # Phase 1: Global
            t = iteration / 100
            w_mid = 1000 * (1 - t) + 3 * t  # Linear decay 1000 -> 3
            return 2.0, w_mid, 1.0, 10.0
        elif iteration < 200:
            # Phase 2: Bridge
            return 3.0, 3.0, 1.0, 5.0
        else:
            # Phase 3: Local
            return 1.0, 0.0, 1.0, 1.0

    def _loss_near(self, Y, pairs):
        """Near pair attraction loss."""
        if len(pairs) == 0:
            return 0.0, np.zeros_like(Y)

        diff = Y[pairs[:, 0]] - Y[pairs[:, 1]]
        d_sq = np.sum(diff ** 2, axis=1) + 1  # d_tilde = ||y_i - y_j||^2 + 1

        loss = np.sum(d_sq / (10 + d_sq))

        # Gradient
        grad = np.zeros_like(Y)
        coef = 10 / ((10 + d_sq) ** 2)
        for idx, (i, j) in enumerate(pairs):
            g = 2 * coef[idx] * diff[idx]
            grad[i] += g
            grad[j] -= g

        return loss, grad

    def _loss_mid(self, Y, pairs):
        """Mid-near pair attraction loss."""
        if len(pairs) == 0:
            return 0.0, np.zeros_like(Y)

        diff = Y[pairs[:, 0]] - Y[pairs[:, 1]]
        d_sq = np.sum(diff ** 2, axis=1) + 1

        loss = np.sum(d_sq / (10000 + d_sq))

        # Gradient
        grad = np.zeros_like(Y)
        coef = 10000 / ((10000 + d_sq) ** 2)
        for idx, (i, j) in enumerate(pairs):
            g = 2 * coef[idx] * diff[idx]
            grad[i] += g
            grad[j] -= g

        return loss, grad

    def _loss_far(self, Y, pairs):
        """Far pair repulsion loss."""
        if len(pairs) == 0:
            return 0.0, np.zeros_like(Y)

        diff = Y[pairs[:, 0]] - Y[pairs[:, 1]]
        d_sq = np.sum(diff ** 2, axis=1) + 1

        loss = np.sum(1 / (1 + d_sq))

        # Gradient (repulsive, so negative)
        grad = np.zeros_like(Y)
        coef = -2 / ((1 + d_sq) ** 2)
        for idx, (i, j) in enumerate(pairs):
            g = coef[idx] * diff[idx]
            grad[i] += g
            grad[j] -= g

        return loss, grad

    def _loss_distortion(self, Y, dist_pairs, hd_distances):
        """Distance preservation loss."""
        if len(dist_pairs) == 0:
            return 0.0, np.zeros_like(Y)

        eps = 1e-8
        diff = Y[dist_pairs[:, 0]] - Y[dist_pairs[:, 1]]
        ld_distances = np.linalg.norm(diff, axis=1) + eps

        log_ratio = np.log(ld_distances) - np.log(hd_distances + eps)
        loss = np.mean(log_ratio ** 2)

        # Gradient
        grad = np.zeros_like(Y)
        coef = 2 * log_ratio / (ld_distances * len(dist_pairs))
        for idx, (i, j) in enumerate(dist_pairs):
            g = coef[idx] * diff[idx] / ld_distances[idx]
            grad[i] += g
            grad[j] -= g

        return loss, grad

    def _optimize(self, Y, X, near_pairs, mid_pairs, far_pairs, dist_pairs, hd_distances):
        """Run phased optimization."""
        for t in range(self.n_iter):
            w_near, w_mid, w_far, lam = self._get_phase_weights(t)

            # Compute losses and gradients
            l_near, g_near = self._loss_near(Y, near_pairs)
            l_mid, g_mid = self._loss_mid(Y, mid_pairs)
            l_far, g_far = self._loss_far(Y, far_pairs)
            l_dist, g_dist = self._loss_distortion(Y, dist_pairs, hd_distances)

            # Combined gradient
            grad = (w_near * g_near + w_mid * g_mid +
                    w_far * g_far + lam * g_dist)

            # Gradient descent step with adaptive learning rate
            lr = self.learning_rate / (1 + t * 0.001)
            Y = Y - lr * grad

            # Center embedding
            Y = Y - Y.mean(axis=0)

        return Y

    def fit(self, X):
        """Fit the model."""
        self.fit_transform(X)
        return self


def compute_uncertainty(X, n_bootstrap=50, **hidra_kwargs):
    """
    Compute embedding uncertainty via bootstrap.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input data.
    n_bootstrap : int, default=50
        Number of bootstrap iterations.
    **hidra_kwargs
        Arguments passed to HiDRA.

    Returns
    -------
    mean_embedding : ndarray of shape (n_samples, n_components)
        Mean embedding across bootstraps.
    uncertainty : ndarray of shape (n_samples,)
        Variance of each point's position.
    """
    n_samples = X.shape[0]
    embeddings = []

    for b in range(n_bootstrap):
        # Bootstrap sample indices
        idx = np.random.choice(n_samples, size=n_samples, replace=True)
        X_boot = X[idx]

        model = HiDRA(random_state=b, **hidra_kwargs)
        Y_boot = model.fit_transform(X_boot)

        # Map back to original indices (average for duplicates)
        Y_full = np.zeros((n_samples, model.n_components))
        counts = np.zeros(n_samples)
        for orig_idx, emb in zip(idx, Y_boot):
            Y_full[orig_idx] += emb
            counts[orig_idx] += 1
        counts[counts == 0] = 1
        Y_full /= counts[:, np.newaxis]

        embeddings.append(Y_full)

    embeddings = np.array(embeddings)
    mean_embedding = embeddings.mean(axis=0)
    uncertainty = embeddings.var(axis=0).sum(axis=1)  # Total variance per point

    return mean_embedding, uncertainty
