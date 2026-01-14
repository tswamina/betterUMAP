Check out: https://garden.tanushswaminathan.com/writings/science/A-Better-UMAP

# betterUMAP

HiDRA: **H**ierarchical **D**istance-**R**egularized **A**pproximation

An improved dimensionality reduction method that addresses UMAP's core failures:
- Explicit distance preservation via stratified distance pair sampling
- Multi-phase optimization (global → local) with early exaggeration
- Uncertainty quantification via bootstrap resampling

Test on whatever genome dataset you'd like

## Installation

```bash
git clone https://github.com/tswamina/betterUMAP.git
cd betterUMAP
pip install -r requirements.txt
```

## Usage

```python
from hidra import HiDRA

# Fit embedding
model = HiDRA(n_components=2, perplexity=30.0, n_iter=1000)
Y = model.fit_transform(X)

# Evaluate
from hidra import evaluate_embedding
metrics = evaluate_embedding(X, Y)
print(f"k-NN Recall: {metrics['knn_recall']:.3f}")
print(f"Spearman ρ: {metrics['spearman_correlation']:.3f}")
print(f"Trustworthiness: {metrics['trustworthiness']:.3f}")
```

### Uncertainty Quantification

```python
from hidra import compute_uncertainty

Y, uncertainty = compute_uncertainty(X, n_bootstrap=30)
# uncertainty[i] = std of embeddings for point i across bootstrap samples
```

### Preprocessing

```python
from hidra import HiDRAPreprocessor

preprocessor = HiDRAPreprocessor(n_components=50, scale=True)
X_processed = preprocessor.fit_transform(X)
```

### Visualization

```python
from hidra import plot_embedding, plot_embedding_with_uncertainty

plot_embedding(Y, labels=y, title="HiDRA Embedding")
plot_embedding_with_uncertainty(Y, uncertainty, labels=y)
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_components` | 2 | Output embedding dimension |
| `perplexity` | 30.0 | Controls neighborhood size for affinity computation |
| `n_iter` | 1000 | Total optimization iterations |
| `distance_weight` | 0.1 | Weight for global distance preservation |
| `early_exaggeration` | 12.0 | P matrix exaggeration factor (first 250 iterations) |
| `learning_rate` | 'auto' | Step size (auto = n_samples / 12) |
| `init` | 'random' | Initialization: 'random' or 'pca' |

## Evaluation Metrics

```python
from hidra import (
    knn_recall,           # Neighborhood preservation [0,1] ↑
    spearman_correlation, # Distance rank correlation [-1,1] ↑
    distortion_ratio,     # Distance scaling consistency [1,∞) ↓
    trustworthiness,      # False neighbor penalty [0,1] ↑
    continuity,           # Missing neighbor penalty [0,1] ↑
)
```

## Run Examples

```bash
python example.py           # Basic usage with Swiss Roll, Blobs, Gaussian
python generate_figures.py  # Generate benchmark comparison figures
```

## Run Tests

```bash
python -m pytest tests/ -v
```

## Benchmark Results

Generated figures in the repo root:
- `swiss_roll_comparison.png` - 3D manifold unfolding
- `s_curve_comparison.png` - S-curve manifold
- `blobs_comparison.png` - High-dimensional clusters
- `benchmark_summary.png` - Metrics comparison table

## Utilities

```python
from hidra import (
    # Data utilities
    estimate_intrinsic_dimension,
    normalize_embedding,
    align_embeddings,      # Procrustes alignment
    subsample_data,

    # Preprocessing
    remove_outliers,
    handle_missing_values,
    compute_distance_matrix,
    batch_generator,
)
```

## Why?

See: [The specious art of single-cell genomics](https://pmc.ncbi.nlm.nih.gov/articles/PMC10434946/)
