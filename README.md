Check out: https://garden.tanushswaminathan.com/writings/science/A-Better-UMAP

# betterUMAP

HiDRA: **H**ierarchical **D**istance-**R**egularized **A**pproximation

An improved dimensionality reduction method that addresses UMAP's core failures:
- Explicit distance preservation
- Multi-phase optimization (global → local)
- Uncertainty quantification

Test on whatever genome dataset you'd like

## Installation

```bash
git clone https://github.com/tswamina/betterUMAP.git
cd betterUMAP
pip install -r requirements.txt
```

## Usage

```python
from hidra import HiDRA, evaluate_embedding

# Fit embedding
model = HiDRA(n_components=2, n_neighbors=15)
Y = model.fit_transform(X)

# Evaluate
from hidra import knn_recall, spearman_correlation
print(f"k-NN Recall: {knn_recall(X, Y):.3f}")
print(f"Spearman ρ: {spearman_correlation(X, Y):.3f}")
```

## Run Example

```bash
python example.py
```

## Why?

See: [The specious art of single-cell genomics](https://pmc.ncbi.nlm.nih.gov/articles/PMC10434946/)
