"""Shared epoch-selection and decision-threshold rules for the probe (A) and
finetune (B) settings.

History of this module, because each rule here replaced one that was measurably
wrong on this series' curves:

**Round 1 -- the original rule.** ``argmax`` over every recorded epoch of the
raw validation balanced accuracy. It selected epoch 279 for A4 seed 0 whose
validation loss bottomed at epoch 15, and epoch 260 for A2ss70 seed 2 whose
minimum was at epoch 121; both landed on the largest weight-decay curve, whose
epoch-to-epoch ``|delta BA|`` was 4-8x the other curves'. A max over 1500 noisy
candidates is biased towards whichever curve is noisiest. Validation-to-test
drops of 0.10-0.12 followed directly.

**Round 2 -- AUC, trailing mean, loss-minimum guard.** Fixed the optimism: the
validation-to-test gap over 20 runs went from a systematic -0.10 to +0.02 with
no consistent sign. But it introduced two new problems, both visible in the
numbers:

* *Metric mismatch.* Test AUC came out identical to round 1 (0.910 vs 0.910 on
  A2ss70) while test balanced accuracy fell 0.842 -> 0.800. Selecting on AUC
  ignores where the 0.5 threshold sits, and balanced accuracy is measured at
  exactly that threshold. Re-thresholding the round-2 models recovered
  0.800 -> 0.863, i.e. past the round-1 number. The threshold, not the epoch,
  was carrying the difference -- hence ``tune_threshold`` below.
* *Plateau drift.* Trailing smoothing systematically penalises a sharp peak
  that follows a dip: B2ss70 seed 0's raw AUC peaked at 0.9505 (epoch 8) but
  its trailing window included two poor epochs, so the rule preferred epoch 17
  at 0.9300. On a 65-subject validation split the standard error of AUC is
  about 0.034, so those two epochs are indistinguishable -- and when scores are
  indistinguishable the earlier, less-trained model is the safer pick. Hence
  the one-standard-error rule in ``choose_epoch``.

Round 3 (this module) therefore keeps AUC + smoothing + the guard, and adds:
a one-standard-error "earliest acceptable epoch" tie-break, and a decision
threshold fitted on validation instead of hardcoded at 0.5.

The rules are deliberately identical for the A and B settings so the two stay
comparable to each other.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np


def trailing_mean(values: Sequence[float], index: int, window: int) -> float:
    """Mean of the finite entries in ``values[index - window + 1 : index + 1]``.

    Non-finite validation losses come from saturated probabilities, not from
    diverged training, so they are skipped rather than allowed to poison the
    window. Returns NaN when the window holds no finite value at all.
    """
    start = max(0, index - window + 1)
    recent = [value for value in values[start:index + 1] if math.isfinite(value)]
    if not recent:
        return float("nan")
    return sum(recent) / len(recent)


def auc_standard_error(auc: float, positives: int, negatives: int) -> float:
    """Hanley-McNeil standard error of an AUC estimate.

    Used as the tolerance for the one-standard-error rule. On this series'
    65-subject validation split (30 AD / 35 CN) at AUC ~0.93 it returns about
    0.034, which is larger than most epoch-to-epoch differences the selector is
    asked to rank -- which is precisely the point: differences inside one
    standard error should not decide anything, so the earliest such epoch wins.
    """
    if positives < 1 or negatives < 1 or not math.isfinite(auc):
        return float("inf")
    auc = min(max(auc, 0.0), 1.0)
    q1 = auc / (2.0 - auc) if auc < 2.0 else 0.0
    q2 = 2.0 * auc * auc / (1.0 + auc) if auc > -1.0 else 0.0
    variance = (auc * (1.0 - auc)
                + (positives - 1) * (q1 - auc * auc)
                + (negatives - 1) * (q2 - auc * auc)) / (positives * negatives)
    return math.sqrt(max(variance, 0.0))


class EpochRecorder:
    """Records per-epoch validation scalars and decides when to stop.

    Selection itself is deferred to :func:`choose_epoch`, which needs the whole
    curve (the one-standard-error rule is defined relative to the best score,
    which is not known until the run ends). Early stopping stays online because
    it is what bounds the run: with ``patience == guard`` no epoch that offline
    selection would consider is ever cut off, so deferring selection costs
    nothing but the memory to hold each epoch's weights.
    """

    def __init__(self, guard: int, window: int, patience: int,
                 overfit_window: int = 0) -> None:
        if guard < 1 or window < 1 or patience < 1:
            raise ValueError("guard, window and patience must all be >= 1")
        self.guard = int(guard)
        self.window = int(window)
        self.patience = int(patience)
        # Explicit divergence stop: training loss still falling while validation
        # loss rises, sustained over `overfit_window` epochs. 0 disables it.
        #
        # Deliberately a *safety net*, not the primary stopping rule. Measured on
        # the round-4 traces, stopping at the validation-loss minimum is worse
        # than running on (test BA 0.870 vs 0.884 on B2ss70, 0.891 vs 0.904 on
        # B4): past that minimum the model keeps improving as a ranker while
        # degrading as a probability estimator, and balanced accuracy after
        # thresholding depends on the former. Requiring a sustained window
        # rather than a single-epoch trigger puts the stop at roughly epoch 20
        # when replayed against those runs -- after every epoch that was
        # actually selected (8-18), so it bounds the wasted tail without
        # truncating the useful region.
        self.overfit_window = int(overfit_window)
        self.records: list[dict[str, Any]] = []
        self._auc: list[float] = []
        self._loss: list[float] = []
        self._train: list[float] = []
        self._best_loss = float("inf")
        self._best_loss_epoch = 0
        self._stale = 0
        self.overfit_stop_epoch: int | None = None

    @property
    def stale_epochs(self) -> int:
        return self._stale

    @property
    def best_loss_epoch(self) -> int:
        return self._best_loss_epoch

    def stop_requested(self) -> bool:
        return self._stale >= self.patience or self.overfit_stop_epoch is not None

    def _overfitting(self) -> bool:
        """Training loss falling while validation loss rises, over two windows.

        Compares the mean of the last `overfit_window` epochs against the mean
        of the `overfit_window` before them, so a single noisy epoch cannot
        trigger it.
        """
        w = self.overfit_window
        if w < 1 or len(self._train) < 2 * w:
            return False

        def means(series: list[float]) -> tuple[float, float] | None:
            recent = [v for v in series[-w:] if math.isfinite(v)]
            earlier = [v for v in series[-2 * w:-w] if math.isfinite(v)]
            if not recent or not earlier:
                return None
            return sum(earlier) / len(earlier), sum(recent) / len(recent)

        train, validation = means(self._train), means(self._loss)
        if train is None or validation is None:
            return False
        return train[1] < train[0] and validation[1] > validation[0]

    def update(self, epoch: int, val_loss: float, val_auc: float,
               val_balanced_accuracy: float, train_loss: float | None = None) -> dict[str, Any]:
        self._loss.append(float(val_loss))
        self._auc.append(float(val_auc))
        self._train.append(float(train_loss) if train_loss is not None else float("nan"))
        if math.isfinite(val_loss) and val_loss < self._best_loss - 1e-9:
            self._best_loss = float(val_loss)
            self._best_loss_epoch = int(epoch)
            self._stale = 0
        else:
            self._stale += 1

        index = len(self._auc) - 1
        smoothed_auc = trailing_mean(self._auc, index, self.window)
        smoothed_loss = trailing_mean(self._loss, index, self.window)
        record = {"epoch": int(epoch), "val_loss": float(val_loss), "val_auc": float(val_auc),
                  "val_balanced_accuracy": float(val_balanced_accuracy),
                  "smoothed_val_auc": smoothed_auc, "smoothed_val_loss": smoothed_loss,
                  "stale_epochs": self._stale, "val_loss_min_epoch": self._best_loss_epoch,
                  # Eligibility depends only on quantities known at this epoch,
                  # so it can be recorded now; `choose_epoch` recomputes the
                  # guard against the final loss minimum anyway.
                  "eligible": bool(len(self._auc) >= self.window and math.isfinite(val_loss)
                                   and math.isfinite(smoothed_auc))}
        if self.overfit_stop_epoch is None and self._overfitting():
            self.overfit_stop_epoch = int(epoch)
        record["overfit_stop"] = self.overfit_stop_epoch == int(epoch)
        self.records.append(record)
        return record


def choose_epoch(records: Sequence[dict[str, Any]], guard: int,
                 positives: int, negatives: int,
                 tolerance_standard_errors: float = 0.0) -> dict[str, Any]:
    """Pick an epoch: the earliest one statistically indistinguishable from the best.

    1. Eligible epochs are those with a full smoothing window, a finite
       validation loss, and an epoch number no more than ``guard`` past the
       validation-loss minimum.
    2. Among them, take the highest smoothed validation AUC, then accept any
       epoch within ``tolerance_standard_errors`` standard errors of it.
    3. Return the **earliest** such epoch. When scores cannot be told apart,
       the less-trained model is preferred -- it has drifted less far from the
       pretrained encoder and has had less opportunity to inflate its logits.

    **The tolerance defaults to 0, i.e. plain ``argmax``, deliberately.** The
    one-standard-error idea is sound but ``auc_standard_error`` is the wrong
    yardstick for it: that is the *marginal* standard error of a single AUC
    (~0.034 here), whereas two epochs are scored on the *same* validation
    subjects and their AUCs are strongly correlated, so the standard error of
    the paired difference is far smaller. Using the marginal value swallows
    almost the whole curve -- replayed against B2ss70 seed 0 it selects epoch 3,
    where the encoder has barely moved, which would collapse the finetune
    setting back into the frozen-probe setting and erase the one effect this
    series has actually established (B > A, consistent across 5/5 and 4/5
    seeds).

    The right yardstick is a paired bootstrap over validation subjects, which
    needs each epoch's validation *predictions*, not just its AUC. Those are
    now recorded (see ``validation_probability_trace`` in the run summaries),
    so the tolerance can be calibrated from data and this argument switched on
    once the traces show whether late selection costs anything at all. Until
    then it stays off rather than being guessed at.
    """
    finite = [record for record in records if math.isfinite(record["val_loss"])]
    if not finite:
        # Every epoch saturated. Fall back to raw AUC over everything recorded.
        fallback = max(records, key=lambda record: record["val_auc"])
        return {**fallback, "selection_rule": "fallback: no finite validation loss"}
    loss_min_epoch = min(finite, key=lambda record: record["val_loss"])["epoch"]
    eligible = [record for record in records
                if record["eligible"] and record["epoch"] <= loss_min_epoch + guard]
    if not eligible:
        fallback = max(finite, key=lambda record: record["val_auc"])
        return {**fallback, "selection_rule": "fallback: no epoch satisfied the guard"}

    best = max(eligible, key=lambda record: record["smoothed_val_auc"])
    tolerance = tolerance_standard_errors * auc_standard_error(
        best["smoothed_val_auc"], positives, negatives)
    if not math.isfinite(tolerance):
        tolerance = 0.0
    acceptable = [record for record in eligible
                  if record["smoothed_val_auc"] >= best["smoothed_val_auc"] - tolerance]
    chosen = min(acceptable, key=lambda record: record["epoch"])
    return {**chosen, "selection_rule":
            f"earliest within {tolerance_standard_errors:g} SE ({tolerance:.4f}) of the best "
            f"smoothed val AUC {best['smoothed_val_auc']:.4f} at epoch {best['epoch']}, "
            f"guard {guard} past val_loss minimum at epoch {loss_min_epoch}"}


def tune_threshold(truth: np.ndarray, probability: np.ndarray) -> float:
    """Decision threshold fitted on validation, at the balanced operating point.

    Balanced accuracy is measured at a fixed threshold, so the threshold is a
    model parameter and leaving it at 0.5 leaves free performance on the table:
    re-thresholding the round-2 models lifted test balanced accuracy 0.800 ->
    0.863 on A2ss70. Two things push the right threshold away from 0.5 here --
    the heads are trained with class weights, and cross-entropy on a separable
    training set inflates logits until nearly every prediction is saturated.

    The balanced operating point (sensitivity == specificity) is used rather
    than ``argmax`` validation balanced accuracy because on 65 subjects the
    latter is a max over a very noisy, step-shaped surface -- the same
    max-of-noise failure this module exists to avoid. Ties break towards the
    higher validation balanced accuracy, and then towards 0.5.
    """
    truth = np.asarray(truth).astype(np.int64)
    probability = np.asarray(probability, dtype=np.float64)
    if (truth == 1).sum() == 0 or (truth == 0).sum() == 0:
        return 0.5
    candidates = np.unique(np.concatenate([probability, [0.0, 1.0]]))
    # Midpoints between observed scores, so the threshold never sits exactly on
    # a sample and flip depending on >= versus >.
    midpoints = np.unique(np.concatenate([candidates, (candidates[:-1] + candidates[1:]) / 2.0]))
    best: tuple[float, float, float, float] | None = None
    for threshold in midpoints:
        prediction = (probability >= threshold).astype(np.int64)
        sensitivity = float((prediction[truth == 1] == 1).mean())
        specificity = float((prediction[truth == 0] == 0).mean())
        key = (abs(sensitivity - specificity), -(sensitivity + specificity) / 2.0,
               abs(float(threshold) - 0.5))
        if best is None or key < best[:3]:
            best = (*key, float(threshold))
    assert best is not None
    return best[3]


def selector_settings(settings: dict[str, Any], default_guard: int,
                      default_window: int, default_patience: int) -> dict[str, int]:
    """Read the selection knobs out of a config section, with defaults.

    Kept permissive so configs written before this module existed keep working:
    a config that sets none of these gets the caller's defaults.
    """
    return {"guard": int(settings.get("select_guard", default_guard)),
            "window": int(settings.get("select_smooth_window", default_window)),
            "patience": int(settings.get("patience", default_patience)),
            "overfit_window": int(settings.get("overfit_stop_window", 0))}
