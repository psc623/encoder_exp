"""Budgeted end-to-end encoder fine-tuning with the common attention head."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from encoderbench.cache import FeatureCache, load_cache
from encoderbench.manifest import read_manifest
from encoderbench.training import _class_weights, _summary, _write_predictions
from encoderbench.utils import checkpoint_identifier, ensure_parent, set_seed, sha256_file, write_json


def _module_parameter_count(module: "torch.nn.Module", excluded: set[int] | None = None) -> int:
    excluded = excluded or set()
    return sum(parameter.numel() for parameter in module.parameters()
               if id(parameter) not in excluded)


def configure_parameter_budget(extractor: Any, budget: int) -> dict[str, Any]:
    """Unfreeze complete output-side groups without splitting a block or stage."""
    if budget <= 0:
        raise ValueError("Fine-tune parameter budget must be positive")
    extractor.model.requires_grad_(False)
    primary, always = extractor.finetune_groups()
    selected_ids: set[int] = set()
    selected_groups: list[dict[str, Any]] = []

    def select(name: str, module: "torch.nn.Module") -> int:
        parameters = [parameter for parameter in module.parameters() if id(parameter) not in selected_ids]
        for parameter in parameters:
            parameter.requires_grad = True
            selected_ids.add(id(parameter))
        count = sum(parameter.numel() for parameter in parameters)
        selected_groups.append({"name": name, "parameters": count})
        return count

    auxiliary_count = sum(select(name, module) for name, module in always)
    primary_count = 0
    primary_selected = 0
    for name, module in primary:
        count = _module_parameter_count(module, selected_ids)
        if count == 0:
            continue
        if primary_selected == 0 or auxiliary_count + primary_count + count <= budget:
            primary_count += select(name, module)
            primary_selected += 1
        else:
            break
    if primary_selected == 0:
        raise RuntimeError(f"{extractor.name} exposes no trainable fine-tune group")

    actual = sum(parameter.numel() for parameter in extractor.model.parameters()
                 if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in extractor.model.parameters())
    if actual != auxiliary_count + primary_count:
        raise RuntimeError("Fine-tune parameter accounting is inconsistent")
    return {
        "target_parameters": int(budget), "trainable_encoder_parameters": int(actual),
        "total_model_parameters": int(total), "trainable_fraction": actual / total,
        "selected_groups": selected_groups,
    }


def _aligned_cache(rows: list[dict[str, str]], cache: FeatureCache) -> tuple[np.ndarray, np.ndarray]:
    by_key = {(str(file_id), str(subject_id)): index for index, (file_id, subject_id) in
              enumerate(zip(cache.file_ids, cache.subject_ids))}
    order = []
    for row in rows:
        key = (row["file_id"], row["subject_id"])
        if key not in by_key:
            raise ValueError(f"Feature cache is missing manifest row {key}")
        index = by_key[key]
        if cache.labels[index] != row["group"] or cache.splits[index] != row["split"]:
            raise ValueError(f"Feature cache metadata differs from manifest for {key}")
        order.append(index)
    if len(order) != len(cache.file_ids):
        raise ValueError("Feature cache and manifest contain different row counts")
    train = np.asarray([position for position, row in enumerate(rows) if row["split"] == "train"])
    aligned = cache.features[np.asarray(order)]
    pooled = aligned[train].mean(axis=1)
    return pooled.mean(axis=0).astype(np.float32), (pooled.std(axis=0) + 1e-6).astype(np.float32)


def _evaluation_cache(rows: list[dict[str, str]], encoder: str, width: int,
                      disease: str) -> FeatureCache:
    count = len(rows)
    return FeatureCache(
        features=np.zeros((count, 64, 1), dtype=np.float32),
        file_ids=np.asarray([row["file_id"] for row in rows]),
        subject_ids=np.asarray([row["subject_id"] for row in rows]),
        labels=np.asarray([row["group"] for row in rows]),
        splits=np.asarray([row["split"] for row in rows]),
        metadata={"disease": disease, "encoder": encoder, "token_shape": [64, width]},
    )


def _atomic_torch_save(value: Any, path: Path) -> Path:
    import torch

    output = ensure_parent(path)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, output)
    return output


def _encoder_delta(model: "torch.nn.Module") -> dict[str, "torch.Tensor"]:
    return {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()
            if parameter.requires_grad}


def _load_encoder_delta(model: "torch.nn.Module", state: dict[str, "torch.Tensor"]) -> None:
    parameters = dict(model.named_parameters())
    if set(state) != {name for name, parameter in parameters.items() if parameter.requires_grad}:
        raise ValueError("Fine-tuned encoder delta keys do not match the selected parameter groups")
    with __import__("torch").no_grad():
        for name, value in state.items():
            parameters[name].copy_(value.to(parameters[name].device, dtype=parameters[name].dtype))


def _protocol_digest(manifest_path: str | Path, encoder: str, seed: int,
                     shuffled: bool, settings: dict[str, Any], config: dict[str, Any]) -> str:
    checkpoint = config["checkpoints"][encoder]
    payload = {
        "schema_version": 1, "manifest_sha256": sha256_file(manifest_path),
        "encoder": encoder, "seed": seed, "shuffled": shuffled, "settings": settings,
        "checkpoint": checkpoint_identifier(checkpoint),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _balanced_accuracy(truth: np.ndarray, prediction: np.ndarray) -> float:
    sensitivity = np.sum((truth == 1) & (prediction == 1)) / max(1, np.sum(truth == 1))
    specificity = np.sum((truth == 0) & (prediction == 0)) / max(1, np.sum(truth == 0))
    return float((sensitivity + specificity) / 2)


def _infer(extractor: Any, head: "torch.nn.Module", rows: list[dict[str, str]],
           device: "torch.device", precision: str) -> np.ndarray:
    import torch

    extractor.model.eval()
    head.eval()
    probabilities = []
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                              enabled=device.type == "cuda" and precision == "bf16")
    with torch.inference_mode(), autocast:
        for row in rows:
            logits = head(extractor.forward_tokens(row["path"]))
            probabilities.append(float(torch.softmax(logits.float(), dim=-1)[0, 1].cpu()))
    return np.asarray(probabilities, dtype=np.float32)


def run_finetune(manifest_path: str | Path, cache_path: str | Path, disease: str,
                 encoder: str, config: dict[str, Any], output_dir: str | Path,
                 seed: int, shuffled: bool = False, device: str = "cuda",
                 restart: bool = False) -> dict[str, Any]:
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
    eval_cache = _evaluation_cache(rows, encoder, width, disease)
    positive = "AD" if disease == "ad" else "SCZ"
    labels = np.asarray([1 if row["group"] == positive else 0 for row in rows], dtype=np.int64)
    indices = {split: np.asarray([index for index, row in enumerate(rows) if row["split"] == split])
               for split in ("train", "validation", "test")}
    train_targets = labels[indices["train"]].copy()
    if shuffled:
        train_targets = train_targets[np.random.default_rng(seed).permutation(len(train_targets))]

    extractor = build_extractor(encoder, config, str(resolved_device))
    budget_audit = configure_parameter_budget(extractor, int(settings["parameter_budget"]))
    head = AttentionPoolHead(width, mean, std, int(settings["hidden_size"])).to(resolved_device)
    class_weights = _class_weights(train_targets).to(resolved_device)
    loss_function = nn.CrossEntropyLoss(weight=class_weights)
    encoder_parameters = [parameter for parameter in extractor.model.parameters()
                          if parameter.requires_grad]
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
    tag = f"seed_{seed}" + ("_shuffled" if shuffled else "")
    best_path = output / f"finetune_{tag}.pt"
    latest_path = output / f"finetune_{tag}_latest.pt"
    prediction_path = output / f"finetune_{tag}_predictions.csv"
    summary_path = output / f"finetune_{tag}_summary.json"
    protocol = _protocol_digest(manifest_path, encoder, seed, shuffled, settings, config)
    best: dict[str, Any] | None = None
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
        latest_path.unlink(missing_ok=True)
        best_path.unlink(missing_ok=True)
        prediction_path.unlink(missing_ok=True)
        summary_path.unlink(missing_ok=True)

    use_amp = resolved_device.type == "cuda" and settings["precision"] == "bf16"
    train_positions = indices["train"]
    gradient_audit: dict[str, Any] | None = None
    for epoch in range(first_epoch, int(settings["max_epochs"])):
        extractor.model.train()
        head.train()
        generator = torch.Generator().manual_seed(seed * 1_000_003 + epoch)
        order = torch.randperm(len(train_positions), generator=generator).numpy()
        progress = tqdm(range(0, len(order), accumulation),
                        desc=f"finetune {encoder} seed={seed} epoch={epoch + 1}", unit="step")
        for start in progress:
            batch = order[start:start + accumulation]
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for local_index in batch:
                row = rows[int(train_positions[local_index])]
                target = torch.tensor([train_targets[local_index]], device=resolved_device)
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
                        f"Fine-tune gradient audit failed for {encoder}: "
                        f"{missing_gradients[:8]}"
                    )
                gradient_audit = {
                    "encoder_parameters_with_gradients": sum(
                        parameter.numel() for parameter in encoder_parameters
                        if parameter.grad is not None
                    ),
                    "missing_gradient_parameters": [],
                    "passed": True,
                }
            torch.nn.utils.clip_grad_norm_(encoder_parameters + list(head.parameters()),
                                           float(settings["gradient_clip"]))
            optimizer.step()
            scheduler.step()
            progress.set_postfix(loss=f"{np.mean(losses):.4f}")

        val_rows = [rows[index] for index in indices["validation"]]
        val_probability = _infer(extractor, head, val_rows, resolved_device, settings["precision"])
        val_truth = labels[indices["validation"]]
        val_prediction = (val_probability >= 0.5).astype(np.int64)
        val_loss = float(nn.functional.cross_entropy(
            torch.from_numpy(np.stack((1.0 - val_probability, val_probability), axis=1)).log(),
            torch.from_numpy(val_truth), weight=class_weights.cpu(),
        ).item())
        candidate = {"epoch": epoch + 1,
                     "balanced_accuracy": _balanced_accuracy(val_truth, val_prediction),
                     "loss": val_loss}
        improved = best is None or (candidate["balanced_accuracy"], -candidate["loss"]) > (
            best["balanced_accuracy"], -best["loss"]
        )
        if improved:
            best = candidate
            stale_epochs = 0
            _atomic_torch_save({
                "protocol_sha256": protocol, "encoder": encoder, "seed": seed,
                "shuffled": shuffled, "encoder_delta": _encoder_delta(extractor.model),
                "head_state": {name: value.detach().cpu() for name, value in head.state_dict().items()},
                "selection": best, "parameter_budget": budget_audit,
                "gradient_audit": gradient_audit,
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
    predictions = _write_predictions(prediction_path, eval_cache, indices["test"],
                                      probability, positive)
    result = {
        "kind": "encoder_finetune", "disease": disease, "encoder": encoder, "seed": seed,
        "shuffled_labels": shuffled, "positive_label": positive, "selection": best,
        "parameter_budget": budget_audit,
        "gradient_audit": selected["gradient_audit"],
        "trainable_parameters": budget_audit["trainable_encoder_parameters"]
                                + trainable_parameter_count(head),
        "head_trainable_parameters": trainable_parameter_count(head),
        "token_shape": [64, width], "checkpoint": str(best_path),
        "base_checkpoint": checkpoint_identifier(config["checkpoints"][encoder]),
        "predictions": str(predictions),
        "metrics": _summary(eval_cache, indices["test"], probability, positive,
                            int(config["evaluation"]["bootstrap_samples"]),
                            int(config["evaluation"]["bootstrap_seed"])),
    }
    write_json(summary_path, result)
    return result
