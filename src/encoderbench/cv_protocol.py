"""Repeated stratified random splits, matching bsnip2-smri-classification's
evaluation protocol (`classification/classify.py:run_task`) instead of
encoderbench's single frozen train/validation/test split.

Why: the frozen protocol carves one fixed validation subset out of the training
pool (`validation_fraction: 0.15` of a pool that is itself half the cohort ->
38 BSNIP2 volumes). Selecting an epoch or a weight decay by scoring 38 samples
is noisy enough that the selected model's test score sits 0.05-0.08 BA below
its validation score -- selection noise, not encoder quality. smri avoids this
by (a) re-drawing the train/test split N times and reporting mean +/- SD over
repeats, and (b) choosing its one hyperparameter by k-fold CV *inside* the
training fold, so every training sample serves as validation once.

smri's exact protocol: `--n-repeats 5 --test-frac 0.30 --inner-folds 5`,
stratified, test scored once per repeat.

Two protocol shapes are produced here, because not every model can afford
k-fold CV over its selection axis:

  * `two_way` (train/test only, no validation rows): for closed-form models
    whose single hyperparameter can be chosen by inner k-fold CV over the whole
    training fold -- the exact smri protocol. Used by the exact-linear head.
  * `three_way` (train/validation/test): for gradient-trained models, where
    selecting the epoch by k-fold CV would mean re-training k times per repeat.
    The training fold is subdivided once into fit/validation. This is still a
    strict improvement over the frozen protocol: validation is drawn fresh per
    repeat (so selection noise averages out across repeats instead of being
    baked in once), and it is larger, because it is carved from a 70% training
    fold instead of a 50% one.

Splitting is at subject level in both cases: a subject's volumes never straddle
train and test.
"""
from __future__ import annotations

from typing import Iterator, Literal

import numpy as np

TEST_FRACTION = 0.30
INNER_FOLDS = 5
N_REPEATS = 5
VALIDATION_FRACTION_OF_TRAIN = 0.20
BASE_SEED = 42


def _stratified_subject_draw(subject_ids: np.ndarray, labels: np.ndarray,
                             fraction: float, rng: np.random.Generator) -> set[str]:
    """Draw `fraction` of subjects per class, so class balance is preserved."""
    subject_label: dict[str, str] = {}
    for subject, label in zip(subject_ids.astype(str), labels.astype(str)):
        if subject in subject_label and subject_label[subject] != label:
            raise ValueError(f"Subject {subject} has conflicting labels")
        subject_label[subject] = label
    drawn: set[str] = set()
    for label in sorted(set(subject_label.values())):
        members = sorted(s for s, value in subject_label.items() if value == label)
        rng.shuffle(members)
        drawn.update(members[: round(len(members) * fraction)])
    return drawn


def repeated_splits(
    subject_ids: np.ndarray,
    labels: np.ndarray,
    shape: Literal["two_way", "three_way"] = "three_way",
    n_repeats: int = N_REPEATS,
    test_fraction: float = TEST_FRACTION,
    validation_fraction_of_train: float = VALIDATION_FRACTION_OF_TRAIN,
    base_seed: int = BASE_SEED,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield `(repeat_index, splits)` where `splits` is a per-row array of
    "train"/"validation"/"test".

    `two_way` still emits a nominal validation block (a single row per class,
    the smallest that keeps downstream cache validation happy) because
    FeatureCache.validate() requires all three split names to be present; the
    exact-linear head folds those rows back into its fitting pool via
    `inner_folds`, so they are not actually held out.
    """
    if shape not in ("two_way", "three_way"):
        raise ValueError(f"Unknown split shape {shape!r}")
    subject_ids = np.asarray(subject_ids).astype(str)
    labels = np.asarray(labels).astype(str)

    for repeat in range(n_repeats):
        rng = np.random.default_rng(base_seed + repeat)
        test_subjects = _stratified_subject_draw(subject_ids, labels, test_fraction, rng)
        splits = np.where(np.isin(subject_ids, list(test_subjects)), "test", "train").astype(object)

        train_mask = splits == "train"
        if shape == "three_way":
            validation_subjects = _stratified_subject_draw(
                subject_ids[train_mask], labels[train_mask], validation_fraction_of_train, rng
            )
        else:
            # One subject per class: satisfies the "all three splits present"
            # invariant without meaningfully shrinking the fitting pool, which
            # inner k-fold CV reunites anyway.
            validation_subjects = set()
            for label in sorted(set(labels[train_mask])):
                candidates = sorted(set(subject_ids[train_mask & (labels == label)]))
                validation_subjects.add(candidates[0])
        splits[np.isin(subject_ids, list(validation_subjects)) & train_mask] = "validation"

        yield repeat, splits.astype(str)


def describe(splits: np.ndarray, labels: np.ndarray) -> dict[str, int]:
    return {f"{split}:{label}": int(np.sum((splits == split) & (labels == label)))
            for split in ("train", "validation", "test")
            for label in sorted(set(np.asarray(labels).astype(str)))}
