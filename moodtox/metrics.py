from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)


def binary_metrics(labels, probabilities, threshold: float = 0.5) -> dict[str, float]:
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    pred = (p >= threshold).astype(int)
    two_classes = len(np.unique(y)) == 2
    return {
        "auc": float(roc_auc_score(y, p)) if two_classes else float("nan"),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, pred)),
    }
