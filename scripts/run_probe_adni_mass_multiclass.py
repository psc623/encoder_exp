#!/usr/bin/env python
"""3-class (CN/MCI/AD) frozen-encoder attention probe: the "A" setting from
the scaling comparison, extended to 3 classes. Standalone script parallel to
encoderbench.training.run_probe for the same reason
run_finetune_adni_mass_multiclass.py is standalone rather than a modification
of encoderbench.finetune.run_finetune: metrics.py/AttentionPoolHead's old
callers/manifest.LABELS are hard-wired for binary classification and
BSNIP2/SCZ depend on that path unchanged. Reuses the genuinely arity-agnostic
pieces (token_normalization) and pairs them with encoderbench.metrics_multiclass.

Same weight-decay grid search / 300-epoch budget / selection-by-validation-
balanced-accuracy protocol as run_probe, just with 3-class labels/loss/metrics.
Saves probe_seed_{seed}.pt with a "model_state" key compatible with
run_finetune_adni_mass_multiclass.py's --warm-start-dir (the head's shape,
including the classifier's num_classes output, is identical either way).
"""
from __future__ import annotations

import argparse
import csv
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from encoderbench.cache import load_cache
from encoderbench.config import load_config
from encoderbench.metrics_multiclass import (aggregate_subjects_multiclass, class_weights_multiclass,
                                             cluster_bootstrap_multiclass, multiclass_metrics)
from encoderbench.training import token_normalization
from encoderbench.utils import ensure_parent, set_seed, write_json

CLASSES = ("CN", "MCI", "AD")


def _write_predictions(path, cache, indices: np.ndarray, probability: np.ndarray,
                       classes: tuple[str, ...]) -> Path:
    output = ensure_parent(path)
    predicted = [classes[int(np.argmax(p))] for p in probability]
    fieldnames = ["file_id", "subject_id", "true", "pred"] + [f"prob_{cls}" for cls in classes]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for local_index, index in enumerate(indices):
            entry = {"file_id": cache.file_ids[index], "subject_id": cache.subject_ids[index],
                     "true": cache.labels[index], "pred": predicted[local_index]}
            for class_index, cls in enumerate(classes):
                entry[f"prob_{cls}"] = float(probability[local_index, class_index])
            writer.writerow(entry)
    return output


def _summary_multiclass(cache, indices: np.ndarray, probability: np.ndarray, classes: tuple[str, ...],
                        bootstrap_samples: int, bootstrap_seed: int) -> dict[str, Any]:
    truth = cache.labels[indices]
    subjects = cache.subject_ids[indices]
    prediction = np.asarray([classes[int(np.argmax(p))] for p in probability])
    volume = multiclass_metrics(truth, prediction, probability, classes)
    volume_ci = cluster_bootstrap_multiclass(truth, prediction, probability, subjects, classes,
                                             bootstrap_samples, bootstrap_seed)
    subject_truth, subject_pred, subject_prob, subject_ids = aggregate_subjects_multiclass(
        truth, probability, subjects, classes)
    subject = multiclass_metrics(subject_truth, subject_pred, subject_prob, classes)
    subject_ci = cluster_bootstrap_multiclass(subject_truth, subject_pred, subject_prob, subject_ids,
                                              classes, bootstrap_samples, bootstrap_seed)
    return {"volume_level": volume, "volume_level_subject_cluster_ci": volume_ci,
           "subject_level": subject, "subject_level_ci": subject_ci}


def run_probe_multiclass(cache_path: str, output_dir: str, seed: int, settings: dict,
                         evaluation: dict, device: str = "auto",
                         classes: tuple[str, ...] = CLASSES) -> dict:
    import torch
    from torch import nn

    from encoderbench.models import AttentionPoolHead, trainable_parameter_count

    cache = load_cache(cache_path)
    class_to_idx = {cls: index for index, cls in enumerate(classes)}
    if not set(cache.labels.tolist()) <= set(classes):
        raise ValueError(f"Cache labels are not a subset of {classes}")
    y = np.asarray([class_to_idx[label] for label in cache.labels], dtype=np.int64)

    set_seed(seed)
    resolved_device = torch.device("cuda" if device == "auto" and torch.cuda.is_available()
                                   else ("cpu" if device == "auto" else device))
    split_indices = {split: np.flatnonzero(cache.splits == split)
                     for split in ("train", "validation", "test")}
    train_y = y[split_indices["train"]].copy()
    train_features = cache.features[split_indices["train"]]
    mean, std = token_normalization(train_features)
    weights = torch.from_numpy(class_weights_multiclass(train_y, len(classes))).to(resolved_device)
    width = cache.features.shape[-1]
    best: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    for weight_decay in settings["weight_decays"]:
        set_seed(seed)
        head = AttentionPoolHead(width, mean, std, settings["hidden_size"],
                                 num_classes=len(classes)).to(resolved_device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=settings["learning_rate"],
                                      weight_decay=float(weight_decay))
        loss_function = nn.CrossEntropyLoss(weight=weights)
        best_for_decay: dict[str, Any] | None = None
        for epoch in range(int(settings["max_epochs"])):
            head.train()
            generator = torch.Generator().manual_seed(seed * 1_000_003 + epoch)
            order = torch.randperm(len(train_y), generator=generator).numpy()
            for start in range(0, len(order), int(settings["batch_size"])):
                batch = order[start:start + int(settings["batch_size"])]
                x = torch.from_numpy(train_features[batch]).to(resolved_device)
                target = torch.from_numpy(train_y[batch]).to(resolved_device)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_function(head(x), target)
                loss.backward()
                optimizer.step()
            val_index = split_indices["validation"]
            head.eval()
            with torch.inference_mode():
                val_x = torch.from_numpy(cache.features[val_index]).to(resolved_device)
                logits = head(val_x)
                val_loss = float(nn.functional.cross_entropy(
                    logits, torch.from_numpy(y[val_index]).to(resolved_device), weight=weights
                ).item())
                probability = torch.softmax(logits.float(), dim=-1).cpu().numpy()
            truth = np.asarray([classes[label] for label in y[val_index]])
            prediction = np.asarray([classes[int(np.argmax(p))] for p in probability])
            metrics = multiclass_metrics(truth, prediction, probability, classes)
            balanced = metrics["balanced_accuracy"]
            history.append({"weight_decay": float(weight_decay), "epoch": epoch + 1,
                            "val_loss": val_loss, "val_balanced_accuracy": balanced})
            candidate = {"balanced_accuracy": balanced, "loss": val_loss, "epoch": epoch + 1,
                        "state": deepcopy({key: value.detach().cpu()
                                           for key, value in head.state_dict().items()})}
            if best_for_decay is None or (balanced, -val_loss) > (
                    best_for_decay["balanced_accuracy"], -best_for_decay["loss"]):
                best_for_decay = candidate
        assert best_for_decay is not None
        best_for_decay["weight_decay"] = float(weight_decay)
        if best is None or (best_for_decay["balanced_accuracy"], -best_for_decay["loss"]) > (
                best["balanced_accuracy"], -best["loss"]):
            best = best_for_decay
    assert best is not None
    head = AttentionPoolHead(width, mean, std, settings["hidden_size"],
                             num_classes=len(classes)).to(resolved_device)
    head.load_state_dict(best.pop("state"))
    head.eval()
    test_index = split_indices["test"]
    with torch.inference_mode():
        probability = torch.softmax(
            head(torch.from_numpy(cache.features[test_index]).to(resolved_device)), dim=-1
        ).cpu().numpy()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    tag = f"seed_{seed}"
    checkpoint = output / f"probe_{tag}.pt"
    torch.save({"model_state": head.state_dict(), "mean": mean, "std": std,
               "cache_metadata": cache.metadata, "seed": seed, "selection": best,
               "classes": list(classes)}, checkpoint)
    predictions = _write_predictions(output / f"probe_{tag}_predictions.csv", cache, test_index,
                                     probability, classes)
    result = {"kind": "attention_probe_multiclass", "disease": cache.metadata.get("disease"),
             "encoder": cache.metadata.get("encoder"), "seed": seed, "classes": list(classes),
             "selection": best, "history": history,
             "trainable_parameters": trainable_parameter_count(head),
             "token_shape": list(cache.features.shape[1:]), "checkpoint": str(checkpoint),
             "predictions": str(predictions),
             "metrics": _summary_multiclass(cache, test_index, probability, classes,
                                            int(evaluation["bootstrap_samples"]),
                                            int(evaluation["bootstrap_seed"]))}
    write_json(output / f"probe_{tag}_summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/adni_mass_scaling_exp.yaml")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    seeds = [int(value) for value in args.seeds.split(",")]
    for seed in seeds:
        result = run_probe_multiclass(args.cache, args.out_dir, seed, config.raw["probe"],
                                      config.raw["evaluation"], args.device)
        print(f"seed {seed} done: volume_level={result['metrics']['volume_level']}", flush=True)


if __name__ == "__main__":
    main()
