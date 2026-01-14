"""
Generate comparison figures for HiDRA vs other dimensionality reduction methods.
Includes UMAP for complete comparison.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll, make_blobs, make_s_curve
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import warnings
warnings.filterwarnings('ignore')

from hidra import HiDRA
from hidra.metrics import knn_recall, spearman_correlation, distortion_ratio

# Set style
plt.style.use('seaborn-v0_8-whitegrid')


def run_methods(X, random_state=42):
    """Run all dimensionality reduction methods."""
    results = {}

    # PCA
    results['PCA'] = PCA(n_components=2, random_state=random_state).fit_transform(X)

    # t-SNE
    results['t-SNE'] = TSNE(n_components=2, random_state=random_state, perplexity=30).fit_transform(X)

    # UMAP
    results['UMAP'] = umap.UMAP(n_components=2, random_state=random_state).fit_transform(X)

    # HiDRA with balanced settings
    results['HiDRA'] = HiDRA(n_components=2, n_iter=1000, distance_weight=0.15,
                             random_state=random_state).fit_transform(X)

    return results


def compute_metrics(X, embeddings):
    """Compute all metrics for each embedding."""
    metrics = {}
    for name, Y in embeddings.items():
        metrics[name] = {
            'kNN Recall': knn_recall(X, Y),
            'Spearman ρ': spearman_correlation(X, Y),
            'Distortion': distortion_ratio(X, Y)
        }
    return metrics


def plot_comparison(X, y, embeddings, metrics, title, filename):
    """Create comparison figure."""
    n_methods = len(embeddings)
    fig, axes = plt.subplots(1, n_methods, figsize=(4.5 * n_methods, 4.5))

    if n_methods == 1:
        axes = [axes]

    for ax, (name, Y) in zip(axes, embeddings.items()):
        ax.scatter(Y[:, 0], Y[:, 1], c=y, cmap='viridis', s=6, alpha=0.7)

        m = metrics[name]
        subtitle = f"kNN: {m['kNN Recall']:.3f} | ρ: {m['Spearman ρ']:.3f}"
        ax.set_title(f"{name}\n{subtitle}", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle(title, fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {filename}")


def create_metrics_table(all_metrics):
    """Create a summary metrics table figure."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')

    datasets = list(all_metrics.keys())
    methods = ['PCA', 't-SNE', 'UMAP', 'HiDRA']

    cell_text = []
    for dataset in datasets:
        for method in methods:
            m = all_metrics[dataset][method]
            row = [
                dataset if method == 'PCA' else '',
                method,
                f"{m['kNN Recall']:.3f}",
                f"{m['Spearman ρ']:.3f}",
                f"{m['Distortion']:.1f}"
            ]
            cell_text.append(row)

    col_labels = ['Dataset', 'Method', 'kNN Recall ↑', 'Spearman ρ ↑', 'Distortion ↓']

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
        colWidths=[0.22, 0.12, 0.15, 0.15, 0.15]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.6)

    # Style header
    for j in range(len(col_labels)):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    # Highlight best results
    for i, dataset in enumerate(datasets):
        base_row = i * 4 + 1

        knn_values = [all_metrics[dataset][m]['kNN Recall'] for m in methods]
        rho_values = [all_metrics[dataset][m]['Spearman ρ'] for m in methods]
        dist_values = [all_metrics[dataset][m]['Distortion'] for m in methods]

        best_knn_idx = np.argmax(knn_values)
        best_rho_idx = np.argmax(rho_values)
        best_dist_idx = np.argmin(dist_values)

        for j in range(len(methods)):
            row = base_row + j
            if j == best_knn_idx:
                table[(row, 2)].set_facecolor('#C6EFCE')
            if j == best_rho_idx:
                table[(row, 3)].set_facecolor('#C6EFCE')
            if j == best_dist_idx:
                table[(row, 4)].set_facecolor('#C6EFCE')

    plt.title('HiDRA Benchmark Results\n(Green = Best in category)',
              fontsize=14, fontweight='bold', pad=20)
    plt.savefig('benchmark_summary.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: benchmark_summary.png")


if __name__ == "__main__":
    np.random.seed(42)
    all_metrics = {}

    # Dataset 1: Swiss Roll
    print("\n[1/4] Swiss Roll dataset...")
    X, y = make_swiss_roll(n_samples=1000, noise=0.5, random_state=42)
    embeddings = run_methods(X)
    metrics = compute_metrics(X, embeddings)
    all_metrics['Swiss Roll'] = metrics
    plot_comparison(X, y, embeddings, metrics, 'Swiss Roll (3D Manifold)', 'swiss_roll_comparison.png')

    # Dataset 2: S-Curve
    print("[2/4] S-Curve dataset...")
    X, y = make_s_curve(n_samples=1000, noise=0.1, random_state=42)
    embeddings = run_methods(X)
    metrics = compute_metrics(X, embeddings)
    all_metrics['S-Curve'] = metrics
    plot_comparison(X, y, embeddings, metrics, 'S-Curve (3D Manifold)', 's_curve_comparison.png')

    # Dataset 3: High-dim Blobs
    print("[3/4] High-dim Blobs dataset...")
    X, y = make_blobs(n_samples=1000, n_features=50, centers=5, cluster_std=2.0, random_state=42)
    embeddings = run_methods(X)
    metrics = compute_metrics(X, embeddings)
    all_metrics['Blobs 50D'] = metrics
    plot_comparison(X, y, embeddings, metrics, 'High-dim Blobs (50D, 5 clusters)', 'blobs_comparison.png')

    # Dataset 4: Gaussian
    print("[4/4] Gaussian dataset...")
    X = np.random.randn(800, 50)
    y = np.zeros(800)
    embeddings = run_methods(X)
    metrics = compute_metrics(X, embeddings)
    all_metrics['Gaussian 50D'] = metrics
    plot_comparison(X, y, embeddings, metrics, 'Random Gaussian (50D)', 'gaussian_comparison.png')

    # Create summary table
    print("\nCreating summary table...")
    create_metrics_table(all_metrics)

    # Print summary
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    print(f"{'Dataset':<15} {'Method':<8} {'kNN':>10} {'Spearman':>12} {'Distortion':>12}")
    print("-"*80)
    for dataset, methods_metrics in all_metrics.items():
        for i, (method, m) in enumerate(methods_metrics.items()):
            ds_name = dataset if i == 0 else ''
            print(f"{ds_name:<15} {method:<8} {m['kNN Recall']:>10.3f} {m['Spearman ρ']:>12.3f} {m['Distortion']:>12.1f}")
        print()

    print("All figures saved successfully!")
