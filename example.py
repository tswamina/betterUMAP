"""
Example usage of HiDRA vs UMAP, t-SNE, and PCA.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll, make_blobs
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Optional: install umap-learn for comparison
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("umap-learn not installed. Skipping UMAP comparison.")

from hidra import HiDRA, knn_recall, spearman_correlation, distortion_ratio


def run_comparison(X, y, dataset_name):
    """Run all methods and compare metrics."""
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"Shape: {X.shape}")
    print('='*60)

    results = {}
    embeddings = {}

    # PCA
    print("\nRunning PCA...")
    pca = PCA(n_components=2)
    Y_pca = pca.fit_transform(X)
    embeddings['PCA'] = Y_pca
    results['PCA'] = {
        'knn_recall': knn_recall(X, Y_pca),
        'spearman': spearman_correlation(X, Y_pca),
        'distortion': distortion_ratio(X, Y_pca)
    }

    # t-SNE
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=42)
    Y_tsne = tsne.fit_transform(X)
    embeddings['t-SNE'] = Y_tsne
    results['t-SNE'] = {
        'knn_recall': knn_recall(X, Y_tsne),
        'spearman': spearman_correlation(X, Y_tsne),
        'distortion': distortion_ratio(X, Y_tsne)
    }

    # UMAP (if available)
    if HAS_UMAP:
        print("Running UMAP...")
        reducer = umap.UMAP(n_components=2, random_state=42)
        Y_umap = reducer.fit_transform(X)
        embeddings['UMAP'] = Y_umap
        results['UMAP'] = {
            'knn_recall': knn_recall(X, Y_umap),
            'spearman': spearman_correlation(X, Y_umap),
            'distortion': distortion_ratio(X, Y_umap)
        }

    # HiDRA
    print("Running HiDRA...")
    hidra = HiDRA(n_components=2, random_state=42)
    Y_hidra = hidra.fit_transform(X)
    embeddings['HiDRA'] = Y_hidra
    results['HiDRA'] = {
        'knn_recall': knn_recall(X, Y_hidra),
        'spearman': spearman_correlation(X, Y_hidra),
        'distortion': distortion_ratio(X, Y_hidra)
    }

    # Print results
    print("\nResults:")
    print("-" * 60)
    print(f"{'Method':<10} {'k-NN Recall':>12} {'Spearman ρ':>12} {'Distortion':>12}")
    print("-" * 60)
    for method, metrics in results.items():
        print(f"{method:<10} {metrics['knn_recall']:>12.3f} {metrics['spearman']:>12.3f} {metrics['distortion']:>12.1f}")

    # Plot
    n_methods = len(embeddings)
    fig, axes = plt.subplots(1, n_methods, figsize=(4 * n_methods, 4))
    if n_methods == 1:
        axes = [axes]

    for ax, (name, Y) in zip(axes, embeddings.items()):
        ax.scatter(Y[:, 0], Y[:, 1], c=y, cmap='viridis', s=5, alpha=0.7)
        ax.set_title(f"{name}\nk-NN: {results[name]['knn_recall']:.2f}, ρ: {results[name]['spearman']:.2f}")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle(dataset_name)
    plt.tight_layout()
    plt.savefig(f"{dataset_name.lower().replace(' ', '_')}_comparison.png", dpi=150)
    plt.show()

    return results


if __name__ == "__main__":
    # Test 1: Swiss Roll (manifold structure)
    print("Generating Swiss Roll dataset...")
    X_swiss, y_swiss = make_swiss_roll(n_samples=1500, noise=0.5, random_state=42)
    run_comparison(X_swiss, y_swiss, "Swiss Roll")

    # Test 2: Blobs (cluster structure)
    print("\nGenerating Blobs dataset...")
    X_blobs, y_blobs = make_blobs(
        n_samples=1500, n_features=50, centers=5,
        cluster_std=2.0, random_state=42
    )
    run_comparison(X_blobs, y_blobs, "High-dim Blobs")

    # Test 3: Single Gaussian (should NOT create clusters)
    print("\nGenerating Single Gaussian dataset...")
    np.random.seed(42)
    X_gauss = np.random.randn(1000, 50)
    y_gauss = np.zeros(1000)  # No real clusters
    run_comparison(X_gauss, y_gauss, "Single Gaussian")

    print("\nDone! Check the saved PNG files for visualizations.")
