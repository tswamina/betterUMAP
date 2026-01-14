from .core import HiDRA, compute_uncertainty
from .metrics import (
    knn_recall,
    spearman_correlation,
    distortion_ratio,
    trustworthiness,
    continuity,
    evaluate_embedding
)
from .utils import (
    estimate_intrinsic_dimension,
    compute_neighborhood_overlap,
    normalize_embedding,
    align_embeddings,
    subsample_data
)
from .preprocessing import (
    HiDRAPreprocessor,
    remove_outliers,
    handle_missing_values,
    compute_distance_matrix
)
from .visualization import (
    plot_embedding,
    plot_embedding_with_uncertainty,
    plot_comparison,
    plot_loss_history,
    plot_metrics_comparison
)

__version__ = "0.2.0"
__all__ = [
    # Core
    "HiDRA",
    "compute_uncertainty",
    # Metrics
    "knn_recall",
    "spearman_correlation",
    "distortion_ratio",
    "trustworthiness",
    "continuity",
    "evaluate_embedding",
    # Utils
    "estimate_intrinsic_dimension",
    "compute_neighborhood_overlap",
    "normalize_embedding",
    "align_embeddings",
    "subsample_data",
    # Preprocessing
    "HiDRAPreprocessor",
    "remove_outliers",
    "handle_missing_values",
    "compute_distance_matrix",
    # Visualization
    "plot_embedding",
    "plot_embedding_with_uncertainty",
    "plot_comparison",
    "plot_loss_history",
    "plot_metrics_comparison"
]
