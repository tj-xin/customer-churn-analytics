
import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    brier_score_loss,
)


def evaluate_classifier(y_true, y_prob, threshold=0.5):
    """Evaluate binary classification probabilities."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "brier_score": brier_score_loss(y_true, y_prob),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        ),
    }


def evaluate_top_k(y_true, y_prob, fractions=(0.1, 0.2, 0.3)):
    """Evaluate churn capture among highest-risk customers."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if len(y_true) != len(y_prob):
        raise ValueError("Labels and probabilities must have equal length.")

    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("y_true must contain binary labels.")

    if not np.isfinite(y_prob).all():
        raise ValueError("Probabilities must be finite.")

    order = np.argsort(-y_prob, kind="stable")
    sorted_labels = y_true[order]

    total_positive = sorted_labels.sum()
    overall_rate = y_true.mean()
    rows = []

    for fraction in fractions:
        if not 0 < fraction <= 1:
            raise ValueError("Each fraction must be in (0, 1].")

        k = max(1, int(len(y_true) * fraction))
        captured = int(sorted_labels[:k].sum())

        precision_at_k = captured / k
        recall_at_k = (
            captured / total_positive if total_positive > 0 else 0.0
        )
        lift_at_k = (
            precision_at_k / overall_rate if overall_rate > 0 else 0.0
        )

        rows.append({
            "fraction": fraction,
            "contacted": k,
            "captured_churn": captured,
            "precision_at_k": precision_at_k,
            "recall_at_k": recall_at_k,
            "lift_at_k": lift_at_k,
        })

    return pd.DataFrame(rows)
