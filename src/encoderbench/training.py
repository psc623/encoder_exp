"""Deterministic attention-probe and frozen-LLM bridge training."""

from __future__ import annotations

import csv
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from encoderbench.cache import FeatureCache, load_cache
from encoderbench.metrics import aggregate_subjects, binary_metrics, cluster_bootstrap
from encoderbench.utils import ensure_parent, set_seed, write_json


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
             positive: str, bootstrap_samples: int, bootstrap_seed: int) -> dict[str, Any]:
    truth = cache.labels[indices]
    subjects = cache.subject_ids[indices]
    prediction = np.where(probability >= 0.5, positive, "CN")
    volume = binary_metrics(truth, prediction, probability, positive)
    volume_ci = cluster_bootstrap(truth, prediction, probability, subjects, positive,
                                  bootstrap_samples, bootstrap_seed)
    st, sp, sprob, sids = aggregate_subjects(truth, probability, subjects, positive)
    subject = binary_metrics(st, sp, sprob, positive)
    subject_ci = cluster_bootstrap(st, sp, sprob, sids, positive,
                                   bootstrap_samples, bootstrap_seed)
    return {"volume_level": volume, "volume_level_subject_cluster_ci": volume_ci,
            "subject_level": subject, "subject_level_ci": subject_ci}


def _write_predictions(path: str | Path, cache: FeatureCache, indices: np.ndarray,
                       probability: np.ndarray, positive: str) -> Path:
    output = ensure_parent(path)
    with output.open("w", newline="", encoding="utf-8") as handle:
        fields = ("file_id", "subject_id", "true", "pred", "positive_probability")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, prob in zip(indices, probability):
            writer.writerow({"file_id": cache.file_ids[index], "subject_id": cache.subject_ids[index],
                             "true": cache.labels[index], "pred": positive if prob >= 0.5 else "CN",
                             "positive_probability": f"{float(prob):.10g}"})
    return output


def run_probe(cache_path: str | Path, output_dir: str | Path, positive: str, seed: int,
              settings: dict[str, Any], evaluation: dict[str, Any], shuffled: bool = False,
              device: str = "auto") -> dict[str, Any]:
    import torch
    from torch import nn

    from encoderbench.models import AttentionPoolHead, trainable_parameter_count

    cache = load_cache(cache_path)
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
    best: dict[str, Any] | None = None
    for weight_decay in settings["weight_decays"]:
        set_seed(seed)
        head = AttentionPoolHead(width, mean, std, settings["hidden_size"]).to(resolved_device)
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
                prediction = logits.argmax(dim=-1).cpu().numpy()
            truth = y[val_index]
            sensitivity = np.sum((truth == 1) & (prediction == 1)) / max(1, np.sum(truth == 1))
            specificity = np.sum((truth == 0) & (prediction == 0)) / max(1, np.sum(truth == 0))
            balanced = float((sensitivity + specificity) / 2)
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
    head = AttentionPoolHead(width, mean, std, settings["hidden_size"]).to(resolved_device)
    head.load_state_dict(best.pop("state"))
    head.eval()
    test_index = split_indices["test"]
    with torch.inference_mode():
        probability = torch.softmax(
            head(torch.from_numpy(cache.features[test_index]).to(resolved_device)), dim=-1
        )[:, 1].cpu().numpy()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    tag = f"seed_{seed}" + ("_shuffled" if shuffled else "")
    checkpoint = output / f"probe_{tag}.pt"
    torch.save({"model_state": head.state_dict(), "mean": mean, "std": std,
                "cache_metadata": cache.metadata, "seed": seed, "shuffled": shuffled,
                "selection": best}, checkpoint)
    predictions = _write_predictions(output / f"probe_{tag}_predictions.csv", cache, test_index,
                                      probability, positive)
    result = {"kind": "attention_probe", "disease": cache.metadata.get("disease"),
              "encoder": cache.metadata.get("encoder"), "seed": seed, "shuffled_labels": shuffled,
              "positive_label": positive, "selection": best,
              "trainable_parameters": trainable_parameter_count(head),
              "token_shape": list(cache.features.shape[1:]), "checkpoint": str(checkpoint),
              "predictions": str(predictions),
              "metrics": _summary(cache, test_index, probability, positive,
                                  int(evaluation["bootstrap_samples"]),
                                  int(evaluation["bootstrap_seed"]))}
    write_json(output / f"probe_{tag}_summary.json", result)
    return result


def run_bridge(cache_path: str | Path, output_dir: str | Path, positive: str, disease: str,
               kind: str, seed: int, settings: dict[str, Any], evaluation: dict[str, Any],
               medgemma_checkpoint: str | Path, device: str = "cuda") -> dict[str, Any]:
    import torch

    from encoderbench.llm import FrozenMedGemma, assert_gradient_isolation
    from encoderbench.models import build_bridge, trainable_parameter_count

    cache = load_cache(cache_path)
    set_seed(seed)
    resolved_device = torch.device(device)
    y = _labels(cache, positive)
    indices = {split: np.flatnonzero(cache.splits == split)
               for split in ("train", "validation", "test")}
    train_y = y[indices["train"]]
    class_weights = _class_weights(train_y).to(resolved_device)
    backbone = FrozenMedGemma(medgemma_checkpoint, str(resolved_device), settings["precision"])
    if backbone.hidden_size != int(settings["llm_hidden_size"]):
        raise ValueError(f"Configured LLM width {settings['llm_hidden_size']} != {backbone.hidden_size}")
    audit = backbone.token_audit(disease)
    bridge = build_bridge(kind, cache.features.shape[-1], settings).to(resolved_device)
    optimizer = torch.optim.AdamW(bridge.parameters(), lr=float(settings["learning_rate"]),
                                  weight_decay=float(settings["weight_decay"]))
    micro = int(settings["micro_batch_size"])
    accumulation = int(settings["gradient_accumulation"])
    steps_per_epoch = math.ceil(len(train_y) / (micro * accumulation))
    total_steps = steps_per_epoch * int(settings["max_epochs"])
    warmup = max(1, round(total_steps * float(settings["warmup_fraction"])))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / warmup)
    )
    best: dict[str, Any] | None = None
    best_state: dict[str, Any] | None = None
    stale_epochs = 0
    gradient_audit: dict[str, object] | None = None
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(int(settings["max_epochs"])):
        bridge.train()
        generator = torch.Generator().manual_seed(seed * 1_000_003 + epoch)
        order = torch.randperm(len(train_y), generator=generator).numpy()
        pending = 0
        for start in range(0, len(order), micro):
            local = order[start:start + micro]
            global_indices = indices["train"][local]
            features = torch.from_numpy(cache.features[global_indices]).to(resolved_device)
            labels = torch.from_numpy(train_y[local]).to(resolved_device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=settings["precision"] == "bf16"):
                visual = bridge(features)
                loss = backbone.weighted_answer_loss(visual, labels, class_weights, audit)
                scaled_loss = loss / accumulation
            scaled_loss.backward()
            pending += 1
            if gradient_audit is None:
                gradient_audit = assert_gradient_isolation(bridge, backbone)
            if pending == accumulation or start + micro >= len(order):
                torch.nn.utils.clip_grad_norm_(bridge.parameters(), float(settings["gradient_clip"]))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                pending = 0
        probability, validation_loss = _bridge_evaluate(
            bridge, backbone, cache.features[indices["validation"]],
            y[indices["validation"]], audit, micro
        )
        prediction = probability >= 0.5
        truth = y[indices["validation"]]
        sensitivity = np.sum(prediction & (truth == 1)) / max(1, np.sum(truth == 1))
        specificity = np.sum((~prediction) & (truth == 0)) / max(1, np.sum(truth == 0))
        candidate = {"epoch": epoch + 1, "balanced_accuracy": float((sensitivity + specificity) / 2),
                     "answer_loss": validation_loss}
        if best is None or (candidate["balanced_accuracy"], -candidate["answer_loss"]) > (
                best["balanced_accuracy"], -best["answer_loss"]):
            best, best_state, stale_epochs = candidate, deepcopy(
                {key: value.detach().cpu() for key, value in bridge.state_dict().items()}), 0
        else:
            stale_epochs += 1
        if stale_epochs >= int(settings["patience"]):
            break
    if best_state is None or best is None or gradient_audit is None:
        raise RuntimeError("Bridge training did not complete one valid epoch")
    bridge.load_state_dict(best_state)
    probability, _ = _bridge_evaluate(bridge, backbone, cache.features[indices["test"]],
                                      y[indices["test"]], audit, micro)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    tag = f"{kind}_seed_{seed}"
    checkpoint = output / f"bridge_{tag}.pt"
    torch.save({"bridge_state": bridge.state_dict(), "bridge_kind": kind, "seed": seed,
                "selection": best, "token_audit": audit.to_dict(),
                "cache_metadata": cache.metadata}, checkpoint)
    predictions = _write_predictions(output / f"bridge_{tag}_predictions.csv", cache,
                                      indices["test"], probability, positive)
    result = {"kind": f"{kind}_bridge", "disease": cache.metadata.get("disease"),
              "encoder": cache.metadata.get("encoder"), "seed": seed, "positive_label": positive,
              "selection": best, "trainable_parameters": trainable_parameter_count(bridge),
              "token_shape": list(cache.features.shape[1:]), "tokenizer_audit": audit.to_dict(),
              "gradient_audit": gradient_audit, "llm_checkpoint": backbone.checkpoint_audit,
              "checkpoint": str(checkpoint), "predictions": str(predictions),
              "metrics": _summary(cache, indices["test"], probability, positive,
                                  int(evaluation["bootstrap_samples"]),
                                  int(evaluation["bootstrap_seed"]))}
    write_json(output / f"bridge_{tag}_summary.json", result)
    return result


def _bridge_evaluate(bridge, backbone, features: np.ndarray, labels: np.ndarray, audit,
                     batch_size: int) -> tuple[np.ndarray, float]:
    import torch

    bridge.eval()
    probabilities, losses = [], []
    with torch.inference_mode():
        for start in range(0, len(labels), batch_size):
            batch_features = torch.from_numpy(features[start:start + batch_size]).to(backbone.device)
            visual = bridge(batch_features)
            probabilities.append(backbone.class_probabilities(visual, audit).cpu().numpy())
            batch_labels = labels[start:start + batch_size]
            for row, label in enumerate(batch_labels):
                answer = audit.answer_a_ids if label == 1 else audit.answer_b_ids
                score = backbone.sequence_scores(visual[row:row + 1], audit.context_ids, answer)[0]
                losses.append(float(-score.item()))
    return np.concatenate(probabilities), float(np.mean(losses))
