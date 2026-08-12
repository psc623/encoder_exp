#!/usr/bin/env python
"""3-class (CN/MCI/AD) full-parameter MASS finetune: native tokens (no 4x4x4
pooling), no parameter budget cap, attention head trained from scratch.

Standalone script deliberately parallel to encoderbench.finetune.run_finetune
rather than a modification of it: this repo's shared metrics/head/loss
machinery (metrics.py, AttentionPoolHead's old callers, manifest.LABELS,
cli.py's --positive logic) is hard-wired for binary classification, and
BSNIP2/SCZ both depend on that binary path unchanged. Rather than threading a
3rd disease-arity concept through all of that, this script reuses only the
genuinely arity-agnostic pieces (configure_parameter_budget, checkpoint
resume/atomic-save helpers, token_normalization, _aligned_cache) and pairs them
with encoderbench.metrics_multiclass (a new, independent module) for the
multiclass loss/metrics/prediction path. Nothing under encoderbench.metrics,
encoderbench.cli, or encoderbench.manifest.LABELS is touched.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from encoderbench.cache import load_cache
from encoderbench.config import load_config
from encoderbench.finetune import (_aligned_cache, _atomic_torch_save, _encoder_delta,
                                    _load_encoder_delta, configure_parameter_budget)
from encoderbench.manifest import read_manifest
from encoderbench.metrics_multiclass import (aggregate_subjects_multiclass, class_weights_multiclass,
                                             cluster_bootstrap_multiclass, multiclass_metrics)
from encoderbench.utils import checkpoint_identifier, ensure_parent, set_seed, sha256_file, write_json

CLASSES = ("CN", "MCI", "AD")


def _protocol_digest(manifest_path, encoder: str, seed: int, settings: dict,
                     config: dict, classes: tuple[str, ...]) -> str:
    checkpoint = config["checkpoints"][encoder]
    payload = {"schema_version": 1, "manifest_sha256": sha256_file(manifest_path),
              "encoder": encoder, "seed": seed, "classes": list(classes), "settings": settings,
              "checkpoint": checkpoint_identifier(checkpoint)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _infer(extractor, head, rows: list[dict], device, precision: str) -> np.ndarray:
    import torch

    extractor.model.eval()
    head.eval()
    probabilities = []
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                              enabled=device.type == "cuda" and precision == "bf16")
    with torch.inference_mode(), autocast:
        for row in rows:
            logits = head(extractor.forward_tokens(row["path"]))
            probabilities.append(torch.softmax(logits.float(), dim=-1)[0].cpu().numpy())
    return np.asarray(probabilities, dtype=np.float32)


def _write_predictions(path, rows: list[dict], indices: np.ndarray, probability: np.ndarray,
                       classes: tuple[str, ...]) -> Path:
    output = ensure_parent(path)
    predicted = [classes[int(np.argmax(p))] for p in probability]
    fieldnames = ["file_id", "subject_id", "true", "pred"] + [f"prob_{cls}" for cls in classes]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for local_index, index in enumerate(indices):
            row = rows[index]
            entry = {"file_id": row["file_id"], "subject_id": row["subject_id"],
                     "true": row["group"], "pred": predicted[local_index]}
            for class_index, cls in enumerate(classes):
                entry[f"prob_{cls}"] = float(probability[local_index, class_index])
            writer.writerow(entry)
    return output


def _summary_multiclass(rows: list[dict], indices: np.ndarray, probability: np.ndarray,
                        classes: tuple[str, ...], bootstrap_samples: int, bootstrap_seed: int) -> dict:
    truth = np.asarray([rows[i]["group"] for i in indices])
    subjects = np.asarray([rows[i]["subject_id"] for i in indices])
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


def _warm_start_head_multiclass(head, warm_start_dir: str | None, seed: int) -> str | None:
    """Load a probe_seed_{seed}.pt checkpoint (written by
    run_probe_adni_mass_multiclass.py) as the head's starting point. The
    3-class AttentionPoolHead's shape (including the classifier's num_classes
    output) matches the probe's exactly, so this is a plain state_dict load,
    no reshaping. Returns the checkpoint path used, or None if not given /
    not found (proceeds with random init in that case)."""
    import torch

    if warm_start_dir is None:
        return None
    checkpoint = Path(warm_start_dir) / f"probe_seed_{seed}.pt"
    if not checkpoint.is_file():
        return None
    probe = torch.load(checkpoint, map_location="cpu", weights_only=False)
    head.load_state_dict(probe["model_state"])
    return str(checkpoint)


def run_finetune_multiclass(manifest_path: str, cache_path: str, encoder: str, config: dict,
                            output_dir: str, seed: int, device: str = "cuda",
                            restart: bool = False, layer: int | None = None,
                            native_tokens: bool = False, classes: tuple[str, ...] = CLASSES,
                            warm_start_dir: str | None = None) -> dict:
    import torch
    from torch import nn
    from tqdm.auto import tqdm

    from encoderbench.extractors import build_extractor
    from encoderbench.models import AttentionPoolHead, trainable_parameter_count

    settings = config["finetune"]
    set_seed(seed)
    resolved_device = torch.device(device)
    rows = read_manifest(manifest_path)
    cache = load_cache(cache_path)
    mean_np, std_np = _aligned_cache(rows, cache)
    width = int(cache.features.shape[-1])
    mean, std = torch.from_numpy(mean_np), torch.from_numpy(std_np)

    class_to_idx = {cls: index for index, cls in enumerate(classes)}
    labels = np.asarray([class_to_idx[row["group"]] for row in rows], dtype=np.int64)
    indices = {split: np.asarray([index for index, row in enumerate(rows) if row["split"] == split])
              for split in ("train", "validation", "test")}
    train_targets = labels[indices["train"]]

    extractor = build_extractor(encoder, config, str(resolved_device), layer, native_tokens)
    budget_audit = configure_parameter_budget(extractor, int(settings["parameter_budget"]))
    head = AttentionPoolHead(width, mean, std, int(settings["hidden_size"]), num_classes=len(classes))
    head_warm_start = _warm_start_head_multiclass(head, warm_start_dir, seed)
    head = head.to(resolved_device)
    class_weights = torch.from_numpy(class_weights_multiclass(train_targets, len(classes))).to(resolved_device)
    loss_function = nn.CrossEntropyLoss(weight=class_weights)
    encoder_parameters = [parameter for parameter in extractor.model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": encoder_parameters, "lr": float(settings["encoder_learning_rate"])},
        {"params": head.parameters(), "lr": float(settings["head_learning_rate"])},
    ], weight_decay=float(settings["weight_decay"]))
    accumulation = int(settings["gradient_accumulation"])
    steps_per_epoch = math.ceil(len(indices["train"]) / accumulation)
    total_steps = int(settings["max_epochs"]) * steps_per_epoch
    warmup = max(1, round(0.05 * total_steps))

    def schedule(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    tag = f"seed_{seed}"
    best_path = output / f"finetune_{tag}.pt"
    latest_path = output / f"finetune_{tag}_latest.pt"
    prediction_path = output / f"finetune_{tag}_predictions.csv"
    summary_path = output / f"finetune_{tag}_summary.json"
    protocol = _protocol_digest(manifest_path, encoder, seed, settings, config, classes)
    best: dict | None = None
    stale_epochs = 0
    first_epoch = 0
    if latest_path.is_file() and not restart:
        latest = torch.load(latest_path, map_location=resolved_device, weights_only=False)
        if latest.get("protocol_sha256") != protocol:
            raise ValueError(f"Existing latest checkpoint uses another protocol: {latest_path}")
        _load_encoder_delta(extractor.model, latest["encoder_delta"])
        head.load_state_dict(latest["head_state"])
        optimizer.load_state_dict(latest["optimizer_state"])
        scheduler.load_state_dict(latest["scheduler_state"])
        best = latest["best"]
        stale_epochs = int(latest["stale_epochs"])
        first_epoch = int(latest["epoch"])
    elif restart:
        for stale_file in (latest_path, best_path, prediction_path, summary_path):
            stale_file.unlink(missing_ok=True)

    use_amp = resolved_device.type == "cuda" and settings["precision"] == "bf16"
    train_positions = indices["train"]
    gradient_audit: dict | None = None
    for epoch in range(first_epoch, int(settings["max_epochs"])):
        extractor.model.train()
        head.train()
        generator = torch.Generator().manual_seed(seed * 1_000_003 + epoch)
        order = torch.randperm(len(train_positions), generator=generator).numpy()
        progress = tqdm(range(0, len(order), accumulation),
                        desc=f"finetune-mc {encoder} seed={seed} epoch={epoch + 1}", unit="step")
        for start in progress:
            batch = order[start:start + accumulation]
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for local_index in batch:
                position = int(train_positions[local_index])
                row = rows[position]
                target = torch.tensor([labels[position]], device=resolved_device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                    logits = head(extractor.forward_tokens(row["path"]))
                    loss = loss_function(logits.float(), target) / len(batch)
                loss.backward()
                losses.append(float(loss.detach()) * len(batch))
            if gradient_audit is None:
                missing_gradients = [name for name, parameter in extractor.model.named_parameters()
                                     if parameter.requires_grad and parameter.grad is None]
                if missing_gradients:
                    raise RuntimeError(
                        f"Fine-tune gradient audit failed for {encoder}: {missing_gradients[:8]}"
                    )
                gradient_audit = {
                    "encoder_parameters_with_gradients": sum(
                        parameter.numel() for parameter in encoder_parameters if parameter.grad is not None
                    ), "missing_gradient_parameters": [], "passed": True,
                }
            torch.nn.utils.clip_grad_norm_(encoder_parameters + list(head.parameters()),
                                           float(settings["gradient_clip"]))
            optimizer.step()
            scheduler.step()
            progress.set_postfix(loss=f"{np.mean(losses):.4f}")

        val_rows = [rows[index] for index in indices["validation"]]
        val_probability = _infer(extractor, head, val_rows, resolved_device, settings["precision"])
        val_truth = np.asarray([classes[label] for label in labels[indices["validation"]]])
        val_prediction = np.asarray([classes[int(np.argmax(p))] for p in val_probability])
        # nll_loss directly on log-probabilities (val_probability already sums to 1 per
        # row from softmax in _infer), unlike passing log-probabilities through
        # cross_entropy (which would re-apply log_softmax on top of an already-log
        # input -- not the same quantity).
        val_loss = float(nn.functional.nll_loss(
            torch.from_numpy(val_probability).clamp_min(1e-8).log(),
            torch.from_numpy(labels[indices["validation"]]), weight=class_weights.cpu(),
        ).item())
        val_metrics = multiclass_metrics(val_truth, val_prediction, val_probability, classes)
        candidate = {"epoch": epoch + 1, "balanced_accuracy": val_metrics["balanced_accuracy"],
                    "loss": val_loss}
        improved = best is None or (candidate["balanced_accuracy"], -candidate["loss"]) > (
            best["balanced_accuracy"], -best["loss"]
        )
        if improved:
            best = candidate
            stale_epochs = 0
            _atomic_torch_save({
                "protocol_sha256": protocol, "encoder": encoder, "seed": seed,
                "encoder_delta": _encoder_delta(extractor.model),
                "head_state": {name: value.detach().cpu() for name, value in head.state_dict().items()},
                "selection": best, "parameter_budget": budget_audit, "gradient_audit": gradient_audit,
            }, best_path)
        else:
            stale_epochs += 1
        assert best is not None
        _atomic_torch_save({
            "protocol_sha256": protocol, "epoch": epoch + 1,
            "encoder_delta": _encoder_delta(extractor.model),
            "head_state": {name: value.detach().cpu() for name, value in head.state_dict().items()},
            "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
            "best": best, "stale_epochs": stale_epochs,
        }, latest_path)
        if stale_epochs >= int(settings["patience"]):
            break

    if not best_path.is_file():
        raise RuntimeError(f"Fine-tuning produced no best checkpoint: {best_path}")
    selected = torch.load(best_path, map_location=resolved_device, weights_only=False)
    _load_encoder_delta(extractor.model, selected["encoder_delta"])
    head.load_state_dict(selected["head_state"])
    best = selected["selection"]
    test_rows = [rows[index] for index in indices["test"]]
    probability = _infer(extractor, head, test_rows, resolved_device, settings["precision"])
    predictions = _write_predictions(prediction_path, rows, indices["test"], probability, classes)
    result = {
        "kind": "encoder_finetune_multiclass", "disease": "ad3", "encoder": encoder, "seed": seed,
        "classes": list(classes), "selection": best, "head_warm_started_from": head_warm_start,
        "parameter_budget": budget_audit, "gradient_audit": selected["gradient_audit"],
        "trainable_parameters": budget_audit["trainable_encoder_parameters"]
                                + trainable_parameter_count(head),
        "head_trainable_parameters": trainable_parameter_count(head),
        "native_tokens": native_tokens, "layer": layer, "channel_width": width,
        "checkpoint": str(best_path),
        "base_checkpoint": checkpoint_identifier(config["checkpoints"][encoder]),
        "predictions": str(predictions),
        "metrics": _summary_multiclass(rows, indices["test"], probability, classes,
                                       int(config["evaluation"]["bootstrap_samples"]),
                                       int(config["evaluation"]["bootstrap_seed"])),
    }
    write_json(summary_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/adni_full_mass_scaling.yaml")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--encoder", default="mass")
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--native-tokens", dest="native_tokens", action="store_true", default=True)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--warm-start-dir", default=None,
                        help="Directory containing probe_seed_N.pt (from "
                             "run_probe_adni_mass_multiclass.py) to warm-start the head from")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    seeds = [int(value) for value in args.seeds.split(",")]
    for seed in seeds:
        result = run_finetune_multiclass(args.manifest, args.cache, args.encoder, config.raw,
                                         args.out_dir, seed, args.device, args.restart,
                                         args.layer, args.native_tokens,
                                         warm_start_dir=args.warm_start_dir)
        print(f"seed {seed} done: volume_level={result['metrics']['volume_level']}", flush=True)


if __name__ == "__main__":
    main()
