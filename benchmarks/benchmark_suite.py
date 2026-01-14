"""
Comprehensive benchmark suite for HiDRA.
"""

import time
import numpy as np
from sklearn.datasets import (
    make_swiss_roll, make_blobs, make_s_curve,
    make_moons, make_circles, load_digits
)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import sys
sys.path.insert(0, '..')
from hidra import HiDRA, evaluate_embedding


class BenchmarkSuite:
    """
    Comprehensive benchmark suite for dimensionality reduction methods.

    Parameters
    ----------
    methods : dict or None
        Dictionary of {name: callable} methods. If None, uses defaults.
    metrics : list or None
        List of metric names to compute. If None, uses all.
    random_state : int, default=42
        Random seed for reproducibility.
    """

    def __init__(self, methods=None, metrics=None, random_state=42):
        self.random_state = random_state

        if methods is None:
            self.methods = {
                'PCA': lambda X: PCA(n_components=2, random_state=random_state).fit_transform(X),
                't-SNE': lambda X: TSNE(n_components=2, random_state=random_state).fit_transform(X),
                'HiDRA': lambda X: HiDRA(n_components=2, random_state=random_state).fit_transform(X)
            }
        else:
            self.methods = methods

        self.metrics = metrics
        self.results = {}

    def get_datasets(self):
        """Return dictionary of benchmark datasets."""
        np.random.seed(self.random_state)

        datasets = {}

        # Manifold datasets
        X, y = make_swiss_roll(n_samples=1500, noise=0.5, random_state=self.random_state)
        datasets['Swiss Roll'] = (X, y, 'manifold')

        X, y = make_s_curve(n_samples=1500, noise=0.1, random_state=self.random_state)
        datasets['S-Curve'] = (X, y, 'manifold')

        # Cluster datasets
        X, y = make_blobs(n_samples=1500, n_features=50, centers=5,
                          cluster_std=2.0, random_state=self.random_state)
        datasets['Blobs 50D'] = (X, y, 'cluster')

        X, y = make_blobs(n_samples=1500, n_features=100, centers=10,
                          cluster_std=1.5, random_state=self.random_state)
        datasets['Blobs 100D'] = (X, y, 'cluster')

        # 2D datasets
        X, y = make_moons(n_samples=1000, noise=0.1, random_state=self.random_state)
        datasets['Moons'] = (X, y, '2d')

        X, y = make_circles(n_samples=1000, noise=0.05, factor=0.5,
                            random_state=self.random_state)
        datasets['Circles'] = (X, y, '2d')

        # No structure
        X = np.random.randn(1000, 50)
        y = np.zeros(1000)
        datasets['Gaussian 50D'] = (X, y, 'random')

        return datasets

    def run_single(self, X, y, method_name):
        """Run a single method on a dataset."""
        method = self.methods[method_name]

        start_time = time.time()
        try:
            Y = method(X)
            elapsed = time.time() - start_time
            metrics = evaluate_embedding(X, Y)
            metrics['time'] = elapsed
            metrics['success'] = True
        except Exception as e:
            metrics = {
                'success': False,
                'error': str(e),
                'time': time.time() - start_time
            }

        return metrics

    def run_all(self, verbose=True):
        """Run all methods on all datasets."""
        datasets = self.get_datasets()
        self.results = {}

        for dataset_name, (X, y, dtype) in datasets.items():
            if verbose:
                print(f"\n[{dataset_name}] ({dtype}, n={X.shape[0]}, d={X.shape[1]})")

            self.results[dataset_name] = {'type': dtype}

            for method_name in self.methods:
                if verbose:
                    print(f"  Running {method_name}...", end=' ')

                metrics = self.run_single(X, y, method_name)
                self.results[dataset_name][method_name] = metrics

                if verbose:
                    if metrics['success']:
                        print(f"kNN={metrics['knn_recall']:.3f}, "
                              f"rho={metrics['spearman_rho']:.3f}, "
                              f"t={metrics['time']:.2f}s")
                    else:
                        print(f"FAILED: {metrics['error']}")

        return self.results

    def summarize(self):
        """Print summary of results."""
        if not self.results:
            print("No results to summarize. Run run_all() first.")
            return

        print("\n" + "=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80)

        # Count wins
        wins = {m: {'knn': 0, 'spearman': 0, 'distortion': 0}
                for m in self.methods}

        for dataset_name, data in self.results.items():
            if 'type' not in data:
                continue

            # Find best for each metric
            knn_best = max(
                [m for m in self.methods if data[m]['success']],
                key=lambda m: data[m].get('knn_recall', -1)
            )
            rho_best = max(
                [m for m in self.methods if data[m]['success']],
                key=lambda m: data[m].get('spearman_rho', -1)
            )
            dist_best = min(
                [m for m in self.methods if data[m]['success']],
                key=lambda m: data[m].get('distortion_ratio', float('inf'))
            )

            wins[knn_best]['knn'] += 1
            wins[rho_best]['spearman'] += 1
            wins[dist_best]['distortion'] += 1

        print("\nWins by metric:")
        print(f"{'Method':<10} {'kNN Recall':>12} {'Spearman':>12} {'Distortion':>12} {'Total':>12}")
        print("-" * 60)
        for method, counts in wins.items():
            total = sum(counts.values())
            print(f"{method:<10} {counts['knn']:>12} {counts['spearman']:>12} "
                  f"{counts['distortion']:>12} {total:>12}")

    def to_dataframe(self):
        """Convert results to pandas DataFrame."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for to_dataframe()")

        rows = []
        for dataset_name, data in self.results.items():
            dtype = data.get('type', 'unknown')
            for method_name in self.methods:
                if method_name in data and data[method_name]['success']:
                    row = {
                        'dataset': dataset_name,
                        'type': dtype,
                        'method': method_name,
                        **{k: v for k, v in data[method_name].items()
                           if k not in ['success', 'error']}
                    }
                    rows.append(row)

        return pd.DataFrame(rows)


def quick_benchmark(X, y=None, verbose=True):
    """
    Quick benchmark on a single dataset.

    Parameters
    ----------
    X : ndarray
        Input data.
    y : ndarray or None
        Labels (for display only).
    verbose : bool
        Whether to print results.

    Returns
    -------
    results : dict
        Results for each method.
    """
    suite = BenchmarkSuite()
    results = {}

    for name, method in suite.methods.items():
        metrics = suite.run_single(X, y, name)
        results[name] = metrics

        if verbose and metrics['success']:
            print(f"{name}: kNN={metrics['knn_recall']:.3f}, "
                  f"rho={metrics['spearman_rho']:.3f}, "
                  f"dist={metrics['distortion_ratio']:.1f}, "
                  f"t={metrics['time']:.2f}s")

    return results


if __name__ == "__main__":
    print("Running HiDRA Benchmark Suite")
    print("=" * 50)

    suite = BenchmarkSuite()
    results = suite.run_all(verbose=True)
    suite.summarize()
