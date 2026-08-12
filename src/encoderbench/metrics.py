"""Volume/subject metrics, clustered confidence intervals, and paired differences."""

from __future__ import annotations

from collections import defaultdict
from typing import Callable

import numpy as np


def force_invalid_wrong(y_true: np.ndarray, y_pred: np.ndarray, positive: str,
                        negative: str = "CN") -> tuple[np.ndarray, int]:
    output = y_pred.astype(str).copy()
    invalid = 0
    for index, prediction in enumerate(output):
        if prediction not in (positive, negative):
            output[index] = negative if y_true[index] == positive else positive
            invalid += 1
    return output, invalid


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, probability: np.ndarray | None,
                   positive: str, negative: str = "CN") -> dict[str, float | int | None]:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if len(y_true) != len(y_pred) or len(y_true) == 0:
        raise ValueError("Metric arrays must have equal non-zero length")
    tp = int(np.sum((y_true == positive) & (y_pred == positive)))
    tn = int(np.sum((y_true == negative) & (y_pred == negative)))
    fp = int(np.sum((y_true == negative) & (y_pred == positive)))
    fn = int(np.sum((y_true == positive) & (y_pred == negative)))
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if precision + sensitivity else 0.0
    auc: float | None = None
    if probability is not None and len(np.unique(y_true)) == 2:
        from sklearn.metrics import roc_auc_score

        auc = float(roc_auc_score((y_true == positive).astype(int), np.asarray(probability)))
    return {"n": len(y_true), "balanced_accuracy": (sensitivity + specificity) / 2,
            "roc_auc": auc, "sensitivity": sensitivity, "specificity": specificity,
            "f1": f1, "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def aggregate_subjects(y_true: np.ndarray, probability: np.ndarray, subject_ids: np.ndarray,
                       positive: str, negative: str = "CN",
                       threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        probabilities.append(float(np.mean(probability[indices])))
    probs = np.asarray(probabilities)
    predictions = np.where(probs >= threshold, positive, negative)
    return np.asarray(labels), predictions, probs, np.asarray(subjects)


def cluster_bootstrap(
    y_true: np.ndarray, y_pred: np.ndarray, probability: np.ndarray | None,
    subject_ids: np.ndarray, positive: str, samples: int = 2000, seed: int = 0,
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
        result = binary_metrics(y_true[indices], y_pred[indices],
                                None if probability is None else probability[indices], positive)
        for name in ("balanced_accuracy", "roc_auc", "sensitivity", "specificity", "f1"):
            if result[name] is not None:
                values[name].append(float(result[name]))
    return {name: (float(np.percentile(series, 2.5)), float(np.percentile(series, 97.5)))
            for name, series in values.items() if series}


def paired_bootstrap_difference(
    truth: dict[str, str], prediction_a: dict[str, str], prediction_b: dict[str, str],
    positive: str, samples: int = 2000, seed: int = 0,
    metric: Callable[..., dict[str, float | int | None]] = binary_metrics,
) -> dict[str, float]:
    subjects = sorted(set(truth) & set(prediction_a) & set(prediction_b))
    if len(subjects) < 2:
        raise ValueError("Paired comparison requires at least two common subjects")
    labels = np.asarray([truth[s] for s in subjects])
    first = np.asarray([prediction_a[s] for s in subjects])
    second = np.asarray([prediction_b[s] for s in subjects])
    point = float(metric(labels, first, None, positive)["balanced_accuracy"]) - float(
        metric(labels, second, None, positive)["balanced_accuracy"]
    )
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(samples):
        indices = rng.integers(0, len(subjects), len(subjects))
        a = float(metric(labels[indices], first[indices], None, positive)["balanced_accuracy"])
        b = float(metric(labels[indices], second[indices], None, positive)["balanced_accuracy"])
        differences.append(a - b)
    return {"difference": point, "ci_low": float(np.percentile(differences, 2.5)),
            "ci_high": float(np.percentile(differences, 97.5)), "n_subjects": len(subjects)}
