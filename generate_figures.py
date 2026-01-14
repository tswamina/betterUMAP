"""
Generate comparison figures for HiDRA vs other dimensionality reduction methods.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll, make_blobs, make_s_curve
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

from hidra import HiDRA, knn_recall, spearman_correlation, distortion_ratio

# Set style
plt.style.use('seaborn-v0_8-whitegrid')


def run_methods(X, random_state=42):
    """Run all dimensionality reduction methods."""
    results = {}

    # PCA
    pca = PCA(n_components=2, random_state=random_state)
    results['PCA'] = pca.fit_transform(X)

    # t-SNE
    tsne = TSNE(n_components=2, random_state=random_state, perplexity=30)
    results['t-SNE'] = tsne.fit_transform(X)

    # HiDRA
    hidra = HiDRA(n_components=2, random_state=random_state)
    results['HiDRA'] = hidra.fit_transform(X)

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
    fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 5))

    if n_methods == 1:
        axes = [axes]

    colors = plt.cm.viridis(y / y.max()) if y.max() > 0 else plt.cm.viridis(np.zeros_like(y))

    for ax, (name, Y) in zip(axes, embeddings.items()):
        ax.scatter(Y[:, 0], Y[:, 1], c=y, cmap='viridis', s=8, alpha=0.7)

        m = metrics[name]
        subtitle = f"kNN: {m['kNN Recall']:.3f} | ρ: {m['Spearman ρ']:.3f} | dist: {m['Distortion']:.1f}"
        ax.set_title(f"{name}\n{subtitle}", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

        # Add border color based on best metric
        for spine in ax.spines.values():
            spine.set_linewidth(2)

    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {filename}")


def create_metrics_table(all_metrics):
    """Create a summary metrics table figure."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')

    # Prepare data for table
    datasets = list(all_metrics.keys())
    methods = ['PCA', 't-SNE', 'HiDRA']

    # Create table data
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
        colWidths=[0.25, 0.15, 0.15, 0.15, 0.15]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # Style header
    for j, label in enumerate(col_labels):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    # Highlight best results
    for i, dataset in enumerate(datasets):
        base_row = i * 3 + 1

        # Find best for each metric
        knn_values = [all_metrics[dataset][m]['kNN Recall'] for m in methods]
        rho_values = [all_metrics[dataset][m]['Spearman ρ'] for m in methods]
        dist_values = [all_metrics[dataset][m]['Distortion'] for m in methods]

        best_knn = methods[np.argmax(knn_values)]
        best_rho = methods[np.argmax(rho_values)]
        best_dist = methods[np.argmin(dist_values)]

        for j, method in enumerate(methods):
            row = base_row + j
            if method == best_knn:
                table[(row, 2)].set_facecolor('#C6EFCE')
            if method == best_rho:
                table[(row, 3)].set_facecolor('#C6EFCE')
            if method == best_dist:
                table[(row, 4)].set_facecolor('#C6EFCE')

    plt.title('Benchmark Results Summary\n(Green = Best)', fontsize=14, fontweight='bold', pad=20)
    plt.savefig('benchmark_summary.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: benchmark_summary.png")


if __name__ == "__main__":
    np.random.seed(42)
    all_metrics = {}

    # Dataset 1: Swiss Roll (manifold)
    print("\n[1/4] Swiss Roll dataset...")
    X_swiss, y_swiss = make_swiss_roll(n_samples=1500, noise=0.5, random_state=42)
    embeddings = run_methods(X_swiss)
    metrics = compute_metrics(X_swiss, embeddings)
    all_metrics['Swiss Roll'] = metrics
    plot_comparison(X_swiss, y_swiss, embeddings, metrics,
                    'Swiss Roll (3D Manifold, n=1500)', 'swiss_roll_comparison.png')

    # Dataset 2: S-Curve (manifold)
    print("[2/4] S-Curve dataset...")
    X_scurve, y_scurve = make_s_curve(n_samples=1500, noise=0.1, random_state=42)
    embeddings = run_methods(X_scurve)
    metrics = compute_metrics(X_scurve, embeddings)
    all_metrics['S-Curve'] = metrics
    plot_comparison(X_scurve, y_scurve, embeddings, metrics,
                    'S-Curve (3D Manifold, n=1500)', 's_curve_comparison.png')

    # Dataset 3: High-dim Blobs (clusters)
    print("[3/4] High-dim Blobs dataset...")
    X_blobs, y_blobs = make_blobs(n_samples=1500, n_features=50, centers=5,
                                   cluster_std=2.0, random_state=42)
    embeddings = run_methods(X_blobs)
    metrics = compute_metrics(X_blobs, embeddings)
    all_metrics['Blobs 50D'] = metrics
    plot_comparison(X_blobs, y_blobs, embeddings, metrics,
                    'High-dim Blobs (50D, 5 clusters, n=1500)', 'blobs_comparison.png')

    # Dataset 4: Gaussian (no structure)
    print("[4/4] Gaussian dataset...")
    X_gauss = np.random.randn(1000, 50)
    y_gauss = np.zeros(1000)
    embeddings = run_methods(X_gauss)
    metrics = compute_metrics(X_gauss, embeddings)
    all_metrics['Gaussian 50D'] = metrics
    plot_comparison(X_gauss, y_gauss, embeddings, metrics,
                    'Single Gaussian (50D, n=1000)', 'gaussian_comparison.png')

    # Create summary table
    print("\nCreating summary table...")
    create_metrics_table(all_metrics)

    # Print summary
    print("\n" + "="*70)
    print("BENCHMARK SUMMARY")
    print("="*70)
    print(f"{'Dataset':<15} {'Method':<8} {'kNN':>8} {'Spearman':>10} {'Distortion':>12}")
    print("-"*70)
    for dataset, methods_metrics in all_metrics.items():
        for i, (method, m) in enumerate(methods_metrics.items()):
            ds_name = dataset if i == 0 else ''
            print(f"{ds_name:<15} {method:<8} {m['kNN Recall']:>8.3f} {m['Spearman ρ']:>10.3f} {m['Distortion']:>12.1f}")
        print()

    print("All figures saved successfully!")
