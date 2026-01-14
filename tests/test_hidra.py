"""
Unit tests for HiDRA.
"""

import numpy as np
import pytest
import sys
sys.path.insert(0, '..')

from hidra import HiDRA, compute_uncertainty
from hidra.metrics import (
    knn_recall, spearman_correlation, distortion_ratio,
    trustworthiness, continuity, evaluate_embedding
)
from hidra.utils import (
    estimate_intrinsic_dimension, normalize_embedding,
    align_embeddings, subsample_data
)
from hidra.preprocessing import (
    HiDRAPreprocessor, remove_outliers, handle_missing_values
)


class TestHiDRA:
    """Tests for the main HiDRA class."""

    def test_basic_fit_transform(self):
        """Test basic fit_transform functionality."""
        np.random.seed(42)
        X = np.random.randn(100, 10)

        model = HiDRA(n_components=2, random_state=42)
        Y = model.fit_transform(X)

        assert Y.shape == (100, 2)
        assert model.embedding_ is not None
        assert np.allclose(Y, model.embedding_)

    def test_reproducibility(self):
        """Test that random_state ensures reproducibility."""
        np.random.seed(42)
        X = np.random.randn(100, 10)

        model1 = HiDRA(n_components=2, random_state=42)
        Y1 = model1.fit_transform(X)

        model2 = HiDRA(n_components=2, random_state=42)
        Y2 = model2.fit_transform(X)

        assert np.allclose(Y1, Y2)

    def test_different_n_components(self):
        """Test different output dimensions."""
        np.random.seed(42)
        X = np.random.randn(100, 20)

        for n_comp in [2, 3, 5]:
            model = HiDRA(n_components=n_comp, random_state=42)
            Y = model.fit_transform(X)
            assert Y.shape == (100, n_comp)

    def test_small_dataset(self):
        """Test on very small dataset."""
        np.random.seed(42)
        X = np.random.randn(20, 5)

        model = HiDRA(n_components=2, n_neighbors=5, random_state=42)
        Y = model.fit_transform(X)

        assert Y.shape == (20, 2)

    def test_parameters(self):
        """Test parameter storage."""
        model = HiDRA(
            n_components=3,
            n_neighbors=20,
            n_iter=500,
            learning_rate=0.5,
            momentum=0.8,
            min_dist=0.3,
            random_state=123
        )

        assert model.n_components == 3
        assert model.n_neighbors == 20
        assert model.n_iter == 500
        assert model.learning_rate == 0.5
        assert model.momentum == 0.8
        assert model.min_dist == 0.3
        assert model.random_state == 123

    def test_fit_method(self):
        """Test fit() returns self."""
        np.random.seed(42)
        X = np.random.randn(50, 10)

        model = HiDRA(random_state=42)
        result = model.fit(X)

        assert result is model
        assert model.embedding_ is not None


class TestMetrics:
    """Tests for evaluation metrics."""

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        X_high = np.random.randn(100, 10)
        X_low = np.random.randn(100, 2)
        return X_high, X_low

    def test_knn_recall_range(self, sample_data):
        """Test kNN recall is in [0, 1]."""
        X_high, X_low = sample_data
        recall = knn_recall(X_high, X_low)
        assert 0 <= recall <= 1

    def test_knn_recall_perfect(self):
        """Test kNN recall is 1 for identical data."""
        np.random.seed(42)
        X = np.random.randn(100, 10)
        recall = knn_recall(X, X[:, :2])  # Not perfect, but should be high
        assert recall > 0

    def test_spearman_range(self, sample_data):
        """Test Spearman correlation is in [-1, 1]."""
        X_high, X_low = sample_data
        rho = spearman_correlation(X_high, X_low)
        assert -1 <= rho <= 1

    def test_distortion_positive(self, sample_data):
        """Test distortion ratio is positive."""
        X_high, X_low = sample_data
        dist = distortion_ratio(X_high, X_low)
        assert dist > 0

    def test_trustworthiness_range(self, sample_data):
        """Test trustworthiness is in [0, 1]."""
        X_high, X_low = sample_data
        trust = trustworthiness(X_high, X_low)
        assert 0 <= trust <= 1

    def test_continuity_range(self, sample_data):
        """Test continuity is in [0, 1]."""
        X_high, X_low = sample_data
        cont = continuity(X_high, X_low)
        assert 0 <= cont <= 1

    def test_evaluate_embedding(self, sample_data):
        """Test evaluate_embedding returns all metrics."""
        X_high, X_low = sample_data
        metrics = evaluate_embedding(X_high, X_low)

        assert 'knn_recall' in metrics
        assert 'spearman_rho' in metrics
        assert 'distortion_ratio' in metrics
        assert 'trustworthiness' in metrics
        assert 'continuity' in metrics


class TestUtils:
    """Tests for utility functions."""

    def test_intrinsic_dimension(self):
        """Test intrinsic dimension estimation."""
        np.random.seed(42)
        # 2D data embedded in 10D
        t = np.random.uniform(0, 2 * np.pi, 200)
        X = np.column_stack([np.cos(t), np.sin(t)])
        X = np.hstack([X, np.zeros((200, 8))])  # Embed in 10D

        dim = estimate_intrinsic_dimension(X, k=10)
        # Should be close to 2, allow some slack
        assert 1 < dim < 5

    def test_normalize_embedding(self):
        """Test embedding normalization."""
        np.random.seed(42)
        Y = np.random.randn(100, 2) * 10 + 5

        Y_std = normalize_embedding(Y, method='standard')
        assert np.abs(Y_std.mean()) < 0.1
        assert np.abs(Y_std.std() - 1) < 0.1

        Y_mm = normalize_embedding(Y, method='minmax')
        assert Y_mm.min() >= 0
        assert Y_mm.max() <= 1

    def test_subsample_data(self):
        """Test data subsampling."""
        np.random.seed(42)
        X = np.random.randn(1000, 10)

        X_sub, idx = subsample_data(X, n_samples=100, random_state=42)
        assert X_sub.shape == (100, 10)
        assert len(idx) == 100
        assert np.allclose(X[idx], X_sub)

    def test_align_embeddings(self):
        """Test Procrustes alignment."""
        np.random.seed(42)
        Y_ref = np.random.randn(100, 2)

        # Rotate and scale
        theta = np.pi / 4
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        Y_target = Y_ref @ R * 2 + 10

        Y_aligned = align_embeddings(Y_ref, Y_target)

        # After alignment, should be close to reference
        # (up to reflection)
        diff1 = np.mean((Y_aligned - Y_ref) ** 2)
        diff2 = np.mean((Y_aligned + Y_ref) ** 2)  # Check reflection
        assert min(diff1, diff2) < 1


class TestPreprocessing:
    """Tests for preprocessing utilities."""

    def test_preprocessor_basic(self):
        """Test basic preprocessor functionality."""
        np.random.seed(42)
        X = np.random.randn(100, 20)

        prep = HiDRAPreprocessor(scaling='standard')
        X_trans = prep.fit_transform(X)

        assert X_trans.shape == X.shape
        assert np.abs(X_trans.mean()) < 0.1

    def test_preprocessor_with_pca(self):
        """Test preprocessor with PCA."""
        np.random.seed(42)
        X = np.random.randn(100, 50)

        prep = HiDRAPreprocessor(scaling='standard', pca_components=10)
        X_trans = prep.fit_transform(X)

        assert X_trans.shape == (100, 10)

    def test_remove_outliers(self):
        """Test outlier removal."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        X[0] = 100  # Add outlier

        X_clean, mask = remove_outliers(X, method='iqr')
        assert len(X_clean) < len(X)
        assert not mask[0]  # Outlier should be removed

    def test_handle_missing_values(self):
        """Test missing value handling."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        X[0, 0] = np.nan
        X[10, 2] = np.nan

        X_filled = handle_missing_values(X, strategy='mean')
        assert not np.any(np.isnan(X_filled))


class TestUncertainty:
    """Tests for uncertainty quantification."""

    def test_compute_uncertainty(self):
        """Test uncertainty computation."""
        np.random.seed(42)
        X = np.random.randn(50, 10)

        mean_emb, uncertainty = compute_uncertainty(X, n_bootstrap=5,
                                                     n_iter=50, random_state=42)

        assert mean_emb.shape == (50, 2)
        assert uncertainty.shape == (50,)
        assert np.all(uncertainty >= 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
