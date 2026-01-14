from .core import HiDRA, compute_uncertainty
from .metrics import (
    knn_recall,
    spearman_correlation,
    distortion_ratio,
    trustworthiness,
    continuity,
    evaluate_embedding
)

__version__ = "0.2.0"
__all__ = [
    "HiDRA",
    "compute_uncertainty",
    "knn_recall",
    "spearman_correlation",
    "distortion_ratio",
    "trustworthiness",
    "continuity",
    "evaluate_embedding"
]
