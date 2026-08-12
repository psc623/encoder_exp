"""Deterministic attention-probe training."""

from __future__ import annotations

import csv
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from encoderbench.cache import FeatureCache, load_cache
from encoderbench.metrics import aggregate_subjects, binary_metrics, cluster_bootstrap
from encoderbench.selection import EpochRecorder, choose_epoch, selector_settings, tune_threshold
from encoderbench.utils import ensure_parent, set_seed, write_json


def _binary_auc(truth: np.ndarray, probability: np.ndarray) -> float:
    """Rank-based AUC, returning 0.5 when one class is absent from the split.

    Used as the per-epoch selection signal (see `encoderbench.selection`), so it
    has to be cheap enough to call every epoch and must never raise -- a
    validation split that momentarily holds one class would otherwise abort a
    300-epoch run.
    """
    truth = np.asarray(truth).astype(np.int64)
    positives = int((truth == 1).sum())
    negatives = int((truth == 0).sum())
    if positives == 0 or negatives == 0:
        return 0.5
    order = np.argsort(np.asarray(probability, dtype=np.float64), kind="mergesort")
    ranks = np.empty(len(truth), dtype=np.float64)
    ranks[order] = np.arange(1, len(truth) + 1, dtype=np.float64)
    # Average ranks within ties so identical probabilities cannot be ordered by
    # array position -- a saturated head emits many exactly-equal scores.
    sorted_probability = np.asarray(probability, dtype=np.float64)[order]
    start = 0
    while start < len(sorted_probability):
        stop = start + 1
        while stop < len(sorted_probability) and sorted_probability[stop] == sorted_probability[start]:
            stop += 1
        if stop - start > 1:
            ranks[order[start:stop]] = ranks[order[start:stop]].mean()
        start = stop
    return float((ranks[truth == 1].sum() - positives * (positives + 1) / 2)
                 / (positives * negatives))


def _selection_key(entry: dict[str, Any]) -> tuple[float, float]:
    """Cross-weight-decay ranking key -- the same one `choose_epoch` uses within a decay."""
    loss = entry.get("smoothed_val_loss", float("nan"))
    return (entry["smoothed_val_auc"], -loss if math.isfinite(loss) else float("-inf"))


def _labels(cache: FeatureCache, positive: str) -> np.ndarray:
    allowed = {"CN", positive}
    if not set(cache.labels.tolist()) <= allowed:
        raise ValueError(f"Cache labels are not a subset of {sorted(allowed)}")
    return (cache.labels == positive).astype(np.int64)


def token_normalization(train_features: np.ndarray) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Per-channel statistics over every training *token*, with a relative floor.

    The head standardizes individual tokens, so the statistics are taken over
    individual tokens rather than over each volume's token-mean. Averaging 64
    tokens first collapses any channel whose only across-token content is the
    fixed position encoding: such a channel has identical values in every
    volume, so its token-mean standard deviation is exactly 0. AnatCL's ResNet18
    layer4 is post-ReLU and has two permanently dead channels of that kind;
    dividing them by the old 1e-6 clamp scaled them to ~1.2e5, which saturated
    every tanh unit in the attention MLP and left the attention weights
    identical for all subjects. The relative floor keeps any remaining
    low-variance channel from dominating the same way.
    """
    import torch

    flat = train_features.reshape(-1, train_features.shape[-1])
    mean = torch.from_numpy(flat.mean(axis=0).astype(np.float32))
    std = torch.from_numpy(flat.std(axis=0).astype(np.float32))
    floor = torch.clamp(0.01 * std.median(), min=1e-6)
    return mean, std.clamp_min(floor)


def _class_weights(labels: np.ndarray) -> "torch.Tensor":
    import torch

    counts = np.bincount(labels, minlength=2).astype(np.float32)
    if np.any(counts == 0):
        raise ValueError("Training split must contain both classes")
    return torch.tensor(counts.sum() / (2.0 * counts), dtype=torch.float32)


def _summary(cache: FeatureCache, indices: np.ndarray, probability: np.ndarray,
             positive: str, bootstrap_samples: int, bootstrap_seed: int,
             threshold: float = 0.5) -> dict[str, Any]:
    truth = cache.labels[indices]
    subjects = cache.subject_ids[indices]
    prediction = np.where(probability >= threshold, positive, "CN")
    volume = binary_metrics(truth, prediction, probability, positive)
    volume_ci = cluster_bootstrap(truth, prediction, probability, subjects, positive,
                                  bootstrap_samples, bootstrap_seed)
    st, sp, sprob, sids = aggregate_subjects(truth, probability, subjects, positive,
                                             threshold=threshold)
    subject = binary_metrics(st, sp, sprob, positive)
    subject_ci = cluster_bootstrap(st, sp, sprob, sids, positive,
                                   bootstrap_samples, bootstrap_seed)
    return {"volume_level": volume, "volume_level_subject_cluster_ci": volume_ci,
            "subject_level": subject, "subject_level_ci": subject_ci,
            "decision_threshold": float(threshold)}


def _write_predictions(path: str | Path, cache: FeatureCache, indices: np.ndarray,
                       probability: np.ndarray, positive: str, threshold: float = 0.5) -> Path:
    output = ensure_parent(path)
    with output.open("w", newline="", encoding="utf-8") as handle:
        # `pred` follows the run's fitted threshold; `pred_at_half` keeps the old
        # fixed-0.5 call alongside it so every downstream reader can reproduce
        # either convention from the same file.
        fields = ("file_id", "subject_id", "true", "pred", "pred_at_half",
                  "positive_probability", "decision_threshold")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, prob in zip(indices, probability):
            writer.writerow({"file_id": cache.file_ids[index], "subject_id": cache.subject_ids[index],
                             "true": cache.labels[index],
                             "pred": positive if prob >= threshold else "CN",
                             "pred_at_half": positive if prob >= 0.5 else "CN",
                             "positive_probability": f"{float(prob):.10g}",
                             "decision_threshold": f"{float(threshold):.10g}"})
    return output


def run_probe(cache_path: str | Path, output_dir: str | Path, positive: str, seed: int,
              settings: dict[str, Any], evaluation: dict[str, Any], shuffled: bool = False,
              device: str = "auto", splits_override: np.ndarray | None = None,
              tag_suffix: str = "") -> dict[str, Any]:
    import torch
    from torch import nn

    from encoderbench.models import AttentionPoolHead, trainable_parameter_count

    cache = load_cache(cache_path)
    if splits_override is not None:
        # Repeated-random-split (smri-style) protocol: the cache's own frozen
        # train/validation/test assignment is replaced per repeat. Additive and
        # opt-in -- every existing caller passes None and is byte-identical.
        if len(splits_override) != len(cache.splits):
            raise ValueError(f"splits_override has {len(splits_override)} rows, "
                             f"cache has {len(cache.splits)}")
        cache.splits = np.asarray(splits_override)
        cache.validate()
    set_seed(seed)
    resolved_device = torch.device("cuda" if device == "auto" and torch.cuda.is_available()
                                   else ("cpu" if device == "auto" else device))
    y = _labels(cache, positive)
    split_indices = {split: np.flatnonzero(cache.splits == split)
                     for split in ("train", "validation", "test")}
    train_y = y[split_indices["train"]].copy()
    if shuffled:
        train_y = train_y[np.random.default_rng(seed).permutation(len(train_y))]
    train_features = cache.features[split_indices["train"]]
    mean, std = token_normalization(train_features)
    weights = _class_weights(train_y).to(resolved_device)
    width = cache.features.shape[-1]
    knobs = selector_settings(settings, default_guard=50, default_window=3, default_patience=50)
    # Cross-entropy has no minimum on a training split the model can separate:
    # the gradient keeps inflating the logits forever, so `train_loss -> 0`
    # reports ||w|| -> inf rather than "finished learning". That is what drove
    # the round-2 finetune curves, where between epochs 13 and 23 the training
    # loss fell three orders of magnitude while validation AUC and balanced
    # accuracy did not move and validation loss rose 26%. Label smoothing gives
    # the objective a finite minimiser, which keeps validation loss meaningful
    # (and therefore keeps the loss-minimum guard meaningful).
    label_smoothing = float(settings.get("label_smoothing", 0.0))
    tolerance = float(settings.get("select_tolerance_standard_errors", 0.0))
    best: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    val_index = split_indices["validation"]
    val_truth = y[val_index]
    val_positives, val_negatives = int((val_truth == 1).sum()), int((val_truth == 0).sum())
    for weight_decay in settings["weight_decays"]:
        set_seed(seed)
        head = AttentionPoolHead(width, mean, std, settings["hidden_size"]).to(resolved_device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=settings["learning_rate"],
                                      weight_decay=float(weight_decay))
        loss_function = nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)
        recorder = EpochRecorder(**knobs)
        # The head is 33,539 parameters, so holding every epoch's weights costs
        # ~134 KB each and lets selection run offline over the whole curve.
        states: dict[int, dict[str, Any]] = {}
        for epoch in range(int(settings["max_epochs"])):
            head.train()
            generator = torch.Generator().manual_seed(seed * 1_000_003 + epoch)
            order = torch.randperm(len(train_y), generator=generator).numpy()
            # Training loss was previously not recorded at all for the probe,
            # which made it impossible to tell "still fitting" from "memorised
            # the training split ten epochs ago" on the A curves. Accumulate it
            # per sample so it is on the same footing as finetune's.
            batch_losses: list[tuple[float, int]] = []
            for start in range(0, len(order), int(settings["batch_size"])):
                batch = order[start:start + int(settings["batch_size"])]
                x = torch.from_numpy(train_features[batch]).to(resolved_device)
                target = torch.from_numpy(train_y[batch]).to(resolved_device)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_function(head(x), target)
                loss.backward()
                optimizer.step()
                batch_losses.append((float(loss.detach()), len(batch)))
            train_loss = (sum(value * count for value, count in batch_losses)
                          / max(1, sum(count for _, count in batch_losses)))
            head.eval()
            with torch.inference_mode():
                val_x = torch.from_numpy(cache.features[val_index]).to(resolved_device)
                logits = head(val_x)
                val_loss = float(nn.functional.cross_entropy(
                    logits, torch.from_numpy(val_truth).to(resolved_device), weight=weights
                ).item())
                probability = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()
                prediction = logits.argmax(dim=-1).cpu().numpy()
            sensitivity = np.sum((val_truth == 1) & (prediction == 1)) / max(1, np.sum(val_truth == 1))
            specificity = np.sum((val_truth == 0) & (prediction == 0)) / max(1, np.sum(val_truth == 0))
            balanced = float((sensitivity + specificity) / 2)
            val_auc = _binary_auc(val_truth, probability)

            record = recorder.update(epoch + 1, val_loss, val_auc, balanced, train_loss)
            record["weight_decay"] = float(weight_decay)
            record["train_loss"] = float(train_loss)
            # Kept so an alternative selection rule can be evaluated offline
            # without retraining -- see the module docstring in
            # encoderbench.selection on why the tolerance is not yet calibrated.
            record["val_probability"] = [round(float(value), 6) for value in probability]
            history.append(dict(record))
            states[epoch + 1] = deepcopy({key: value.detach().cpu()
                                          for key, value in head.state_dict().items()})
            print(f"[probe] seed={seed} wd={weight_decay} epoch={epoch + 1:3d} "
                  f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_auc={val_auc:.4f} "
                  f"val_ba={balanced:.4f} smooth_auc={record['smoothed_val_auc']:.4f} "
                  f"eligible={int(record['eligible'])} stale={record['stale_epochs']} "
                  f"loss_min_ep={record['val_loss_min_epoch']}", flush=True)
            if recorder.stop_requested():
                reason = ((f"overfitting detected over {recorder.overfit_window} epochs "
                           f"(first flagged at epoch {recorder.overfit_stop_epoch})")
                          if recorder.overfit_stop_epoch is not None else
                          (f"{recorder.stale_epochs} epochs without a validation-loss "
                           f"improvement; minimum was epoch {recorder.best_loss_epoch}"))
                print(f"[probe] seed={seed} wd={weight_decay} early stop at epoch {epoch + 1} "
                      f"-- {reason}", flush=True)
                break
        best_for_decay = dict(choose_epoch(recorder.records, knobs["guard"],
                                           val_positives, val_negatives, tolerance))
        best_for_decay["weight_decay"] = float(weight_decay)
        best_for_decay["state"] = states[best_for_decay["epoch"]]
        print(f"[probe] seed={seed} wd={weight_decay} selected epoch {best_for_decay['epoch']} "
              f"({best_for_decay['selection_rule']})", flush=True)
        if best is None or _selection_key(best_for_decay) > _selection_key(best):
            best = best_for_decay
    assert best is not None
    # Downstream readers (report generator, RESULTS tables) expect these two.
    best["balanced_accuracy"] = best["val_balanced_accuracy"]
    best["loss"] = best["val_loss"]
    best.pop("val_probability", None)
    head = AttentionPoolHead(width, mean, std, settings["hidden_size"]).to(resolved_device)
    head.load_state_dict(best.pop("state"))
    head.eval()
    test_index = split_indices["test"]
    with torch.inference_mode():
        val_probability = torch.softmax(
            head(torch.from_numpy(cache.features[val_index]).to(resolved_device)), dim=-1
        )[:, 1].cpu().numpy()
        probability = torch.softmax(
            head(torch.from_numpy(cache.features[test_index]).to(resolved_device)), dim=-1
        )[:, 1].cpu().numpy()
    # Fit the decision threshold on validation. Balanced accuracy is measured at
    # a fixed threshold, so leaving it at 0.5 was throwing away real performance:
    # re-thresholding the round-2 models lifted test balanced accuracy from 0.800
    # to 0.863 on A2ss70, which is past what the old validation-BA-argmax rule
    # reached. Both numbers are reported below so the threshold's contribution
    # stays separately attributable from the other round-3 changes.
    threshold = tune_threshold(val_truth, val_probability)
    print(f"[probe] seed={seed} decision threshold fitted on validation: {threshold:.4f}",
          flush=True)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    tag = f"seed_{seed}" + ("_shuffled" if shuffled else "") + tag_suffix
    checkpoint = output / f"probe_{tag}.pt"
    torch.save({"model_state": head.state_dict(), "mean": mean, "std": std,
                "cache_metadata": cache.metadata, "seed": seed, "shuffled": shuffled,
                "selection": best, "decision_threshold": float(threshold)}, checkpoint)
    predictions = _write_predictions(output / f"probe_{tag}_predictions.csv", cache, test_index,
                                      probability, positive, threshold)
    result = {"kind": "attention_probe", "disease": cache.metadata.get("disease"),
              "encoder": cache.metadata.get("encoder"), "seed": seed, "shuffled_labels": shuffled,
              "positive_label": positive, "selection": best, "history": history,
              "decision_threshold": float(threshold),
              "trainable_parameters": trainable_parameter_count(head),
              "token_shape": list(cache.features.shape[1:]), "checkpoint": str(checkpoint),
              "predictions": str(predictions),
              "metrics": _summary(cache, test_index, probability, positive,
                                  int(evaluation["bootstrap_samples"]),
                                  int(evaluation["bootstrap_seed"]), threshold),
              "metrics_at_half": _summary(cache, test_index, probability, positive,
                                          int(evaluation["bootstrap_samples"]),
                                          int(evaluation["bootstrap_seed"]), 0.5)}
    write_json(output / f"probe_{tag}_summary.json", result)
    return result
