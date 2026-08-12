"""3-class (CN/MCI/AD) metrics, additive alongside metrics.py's binary-only
functions. metrics.py's binary_metrics/cluster_bootstrap/paired_bootstrap_difference
are untouched -- BSNIP2 and SCZ (both 2-class) keep using those unchanged.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np


def multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray, probability: np.ndarray | None,
                       classes: Sequence[str]) -> dict:
    """`probability`, if given, is `[n, len(classes)]`. Balanced accuracy is
    macro-averaged per-class recall (the standard multiclass generalization of
    the binary (sensitivity+specificity)/2 definition metrics.py uses)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if len(y_true) != len(y_pred) or len(y_true) == 0:
        raise ValueError("Metric arrays must have equal non-zero length")
    per_class = {}
    recalls = []
    for cls in classes:
        tp = int(np.sum((y_true == cls) & (y_pred == cls)))
        fn = int(np.sum((y_true == cls) & (y_pred != cls)))
        fp = int(np.sum((y_true != cls) & (y_pred == cls)))
        recall = tp / (tp + fn) if tp + fn else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        recalls.append(recall)
        per_class[cls] = {"recall": recall, "precision": precision, "f1": f1,
                          "tp": tp, "fn": fn, "fp": fp}
    auc_macro_ovr = None
    if probability is not None and len(np.unique(y_true)) == len(classes):
        from sklearn.metrics import roc_auc_score

        try:
            auc_macro_ovr = float(roc_auc_score(y_true, np.asarray(probability),
                                                multi_class="ovr", average="macro",
                                                labels=list(classes)))
        except ValueError:
            auc_macro_ovr = None
    return {"n": len(y_true), "balanced_accuracy": float(np.mean(recalls)),
           "roc_auc_ovr_macro": auc_macro_ovr, "per_class": per_class}


def aggregate_subjects_multiclass(
    y_true: np.ndarray, probability: np.ndarray, subject_ids: np.ndarray, classes: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """`probability` is `[n, len(classes)]`; subject-level probability is the
    per-class mean across that subject's volumes, prediction is its argmax."""
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, subject in enumerate(subject_ids.astype(str)):
        grouped[subject].append(index)
    subjects, labels, probabilities = [], [], []
    for subject in sorted(grouped):
        indices = grouped[subject]
        unique = set(y_true[indices].tolist())
        if len(unique) != 1:
            raise ValueError(f"Subject {subject} has inconsistent labels")
        subjects.append(subject)
        labels.append(next(iter(unique)))
        probabilities.append(np.mean(probability[indices], axis=0))
    probs = np.asarray(probabilities)
    class_array = np.asarray(classes)
    predictions = class_array[np.argmax(probs, axis=1)]
    return np.asarray(labels), predictions, probs, np.asarray(subjects)


def cluster_bootstrap_multiclass(
    y_true: np.ndarray, y_pred: np.ndarray, probability: np.ndarray | None,
    subject_ids: np.ndarray, classes: Sequence[str], samples: int = 2000, seed: int = 0,
) -> dict[str, tuple[float, float]]:
    unique_subjects = np.unique(subject_ids.astype(str))
    if len(unique_subjects) < 2:
        raise ValueError("Cluster bootstrap requires at least two subjects")
    grouped = {subject: np.flatnonzero(subject_ids.astype(str) == subject) for subject in unique_subjects}
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = defaultdict(list)
    for _ in range(samples):
        chosen = rng.choice(unique_subjects, size=len(unique_subjects), replace=True)
        indices = np.concatenate([grouped[subject] for subject in chosen])
        result = multiclass_metrics(y_true[indices], y_pred[indices],
                                    None if probability is None else probability[indices], classes)
        values["balanced_accuracy"].append(result["balanced_accuracy"])
        if result["roc_auc_ovr_macro"] is not None:
            values["roc_auc_ovr_macro"].append(result["roc_auc_ovr_macro"])
        for cls in classes:
            values[f"{cls}_recall"].append(result["per_class"][cls]["recall"])
    return {name: (float(np.percentile(series, 2.5)), float(np.percentile(series, 97.5)))
           for name, series in values.items() if series}


def class_weights_multiclass(labels: np.ndarray, num_classes: int) -> "np.ndarray":
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    if np.any(counts == 0):
        raise ValueError(f"Training split must contain all {num_classes} classes, got counts {counts}")
    return counts.sum() / (num_classes * counts)
