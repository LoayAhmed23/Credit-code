"""
Evaluation metrics and report generation.
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score, roc_curve, f1_score, recall_score,
    precision_score, accuracy_score, confusion_matrix, classification_report,
)

import config


def ks_statistic(y_true, y_proba) -> float:
    """Kolmogorov-Smirnov statistic: max |TPR - FPR|."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    return float(np.max(np.abs(tpr - fpr)))


def evaluate(y_true, y_pred, y_proba) -> dict:
    """Compute a full set of binary-classification metrics."""
    return {
        "AUC": roc_auc_score(y_true, y_proba),
        "KS": ks_statistic(y_true, y_proba),
        "F1": f1_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Accuracy": accuracy_score(y_true, y_pred),
    }


def generate_report(metrics: dict, y_true, y_pred,
                     output_path: str = None) -> str:
    """Build a plain-text evaluation report and optionally write it to disk."""
    lines = []
    lines.append("=" * 60)
    lines.append("CREDIT RISK MODEL — EVALUATION REPORT")
    lines.append("=" * 60)

    # Scalar metrics
    lines.append("")
    lines.append("METRICS")
    lines.append("-" * 40)
    for name, value in metrics.items():
        lines.append(f"  {name:<12s}: {value:.4f}")

    # Class distribution
    lines.append("")
    lines.append("CLASS DISTRIBUTION (test set)")
    lines.append("-" * 40)
    total = len(y_true)
    n_default = int(np.sum(y_true == 1))
    n_normal = total - n_default
    lines.append(f"  Non-default (0): {n_normal:>7,}  ({n_normal/total*100:.1f}%)")
    lines.append(f"  Default     (1): {n_default:>7,}  ({n_default/total*100:.1f}%)")

    # Confusion matrix
    lines.append("")
    lines.append("CONFUSION MATRIX")
    lines.append("-" * 40)
    cm = confusion_matrix(y_true, y_pred)
    lines.append(f"  TN={cm[0,0]:>7,}   FP={cm[0,1]:>7,}")
    lines.append(f"  FN={cm[1,0]:>7,}   TP={cm[1,1]:>7,}")

    # Classification report
    lines.append("")
    lines.append("CLASSIFICATION REPORT")
    lines.append("-" * 40)
    lines.append(classification_report(y_true, y_pred, target_names=["Non-default", "Default"]))

    lines.append("=" * 60)

    report = "\n".join(lines)

    if output_path:
        with open(output_path, "w") as f:
            f.write(report)
        print(f"[evaluation] Report saved to {output_path}")

    return report
