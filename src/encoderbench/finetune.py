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
from encoderbench.selection import (EpochRecorder, choose_epoch, selector_settings,
                                    tune_threshold)
from encoderbench.training import (_binary_auc, _class_weights, _summary, _write_predictions,
                                   token_normalization)
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
    # Same statistics the attention probe uses: over individual tokens, with a
    # relative floor. Taking them over each volume's token-mean (as this did)
    # gives std exactly 0 for any channel whose only across-token content is the
    # fixed position encoding, which then blows up to ~1e5 after division.
    mean, std = token_normalization(aligned[train])
    return mean.numpy().astype(np.float32), std.numpy().astype(np.float32)


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


def _warm_start_head(head: "torch.nn.Module", config: dict[str, Any], disease: str,
                     encoder: str, seed: int, directory_override: str | Path | None = None) -> str | None:
    """Load the probe's already-tuned head as finetune's starting point.

    Without this, finetune trains a randomly-initialized head from scratch under
    a far smaller budget than probe used (12 epochs/one weight decay vs up to 300
    epochs/a 5-way weight-decay grid search), which confounds "did unfreezing the
    encoder help" with "did this head, trained worse, do worse". Warm-starting
    means finetune measures the encoder-adaptation effect starting from the same
    point probe already found, instead of from a fresh coin flip.

    The default checkpoint directory is keyed only by (disease, encoder) -- fine
    for the single frozen `ad`/`mass` probe this was designed around, but a
    collision risk once multiple probes exist for the same (disease, encoder)
    pair trained on different manifests/protocols (e.g. a native-token probe on
    a different dataset than the original 4x4x4 one). `directory_override`
    swaps in a different base directory (still expects `probe_seed_{seed}.pt`
    inside it, same as the default -- this is what lets one CLI invocation warm
    -start every seed from its own matching probe seed).

    Returns the checkpoint path used, or None if no matching probe checkpoint
    exists (finetune still proceeds with the random initialization in that case).
    """
    import torch

    directory = Path(directory_override) if directory_override is not None else (
        Path(config["output_root"]) / "attention" / disease / encoder)
    checkpoint = directory / f"probe_seed_{seed}.pt"
    if not checkpoint.is_file():
        return None
    probe = torch.load(checkpoint, map_location="cpu", weights_only=False)
    head.load_state_dict(probe["model_state"])
    return str(checkpoint)


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
                 restart: bool = False, layer: int | None = None,
                 native_tokens: bool = False, warm_start: bool = True,
                 warm_start_dir: str | Path | None = None) -> dict[str, Any]:
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
    positive = "AD" if disease == "ad" else ("SZ" if disease == "bsnip2" else "SCZ")
    labels = np.asarray([1 if row["group"] == positive else 0 for row in rows], dtype=np.int64)
    indices = {split: np.asarray([index for index, row in enumerate(rows) if row["split"] == split])
               for split in ("train", "validation", "test")}
    train_targets = labels[indices["train"]].copy()
    if shuffled:
        train_targets = train_targets[np.random.default_rng(seed).permutation(len(train_targets))]

    extractor = build_extractor(encoder, config, str(resolved_device), layer, native_tokens)
    budget_audit = configure_parameter_budget(extractor, int(settings["parameter_budget"]))
    head = AttentionPoolHead(width, mean, std, int(settings["hidden_size"]))
    head_warm_start = (_warm_start_head(head, config, disease, encoder, seed, warm_start_dir)
                       if warm_start else None)
    head = head.to(resolved_device)
    class_weights = _class_weights(train_targets).to(resolved_device)
    # See the same comment in training.run_probe: cross-entropy has no minimum
    # once the training split is separable, so the round-2 finetune runs drove
    # train_loss to 0.0000 by epoch 14-18 while validation AUC stopped moving --
    # logit inflation, not learning. Label smoothing gives the objective a
    # finite minimiser so validation loss (and hence the guard) stays meaningful.
    loss_function = nn.CrossEntropyLoss(weight=class_weights,
                                        label_smoothing=float(settings.get("label_smoothing", 0.0)))
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
    # Head-only warm-up. Measured on B2ss70/B4: at encoder_lr=5e-4 the first
    # epoch alone drove validation balanced accuracy from the warm-started
    # probe's ~0.85 down to 0.50-0.61, i.e. the warm start was destroyed before
    # it could do anything. AdamW's per-element step is ~lr regardless of
    # gradient scale, and MASS's encoder.down3 weights have std ~0.017, so one
    # 5e-4 step moves each weight by ~3% of its own scale and a 72-step epoch
    # can rewrite the stage outright. Holding the encoder at lr=0 for the first
    # `encoder_freeze_epochs` lets the head re-settle against the live encoder
    # first; the encoder's Adam moments still accumulate during those steps, so
    # it is warm rather than cold when its learning rate turns on.
    freeze_steps = int(settings.get("encoder_freeze_epochs", 0)) * steps_per_epoch

    def schedule(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    def encoder_schedule(step: int) -> float:
        return 0.0 if step < freeze_steps else schedule(step)

    # Per-group lambdas: group 0 is the encoder, group 1 is the head. Freezing
    # via lr=0 rather than requires_grad=False keeps the gradient audit below
    # meaningful and keeps AdamW's decoupled weight decay off the encoder too
    # (decay is scaled by the same lr).
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [encoder_schedule, schedule])
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    tag = f"seed_{seed}" + ("_shuffled" if shuffled else "")
    best_path = output / f"finetune_{tag}.pt"
    latest_path = output / f"finetune_{tag}_latest.pt"
    prediction_path = output / f"finetune_{tag}_predictions.csv"
    summary_path = output / f"finetune_{tag}_summary.json"
    protocol = _protocol_digest(manifest_path, encoder, seed, shuffled, settings, config)
    knobs = selector_settings(settings, default_guard=15, default_window=3, default_patience=15)
    tolerance = float(settings.get("select_tolerance_standard_errors", 0.0))
    record_test_trace = bool(settings.get("record_test_trace", False))
    recorder = EpochRecorder(**knobs)
    # Selection is deferred to the end of the run (`choose_epoch` ranks against
    # the best score, which is not known until then), so every epoch's weights
    # have to survive until selection. 9.3M parameters is ~37 MB per epoch and
    # a run stops after ~20-25 epochs, so they go to disk beside the run rather
    # than into RAM -- that also means a resubmitted job resumes with the
    # earlier epochs' weights still available. All but the selected one are
    # deleted once selection is made.
    epoch_state_path = lambda epoch: output / f"finetune_{tag}_epoch{epoch:03d}.pt"  # noqa: E731
    best: dict[str, Any] | None = None
    first_epoch = 0
    history: list[dict[str, Any]] = []
    if latest_path.is_file() and not restart:
        latest = torch.load(latest_path, map_location=resolved_device, weights_only=False)
        if latest.get("protocol_sha256") != protocol:
            raise ValueError(f"Existing latest checkpoint uses another protocol: {latest_path}")
        _load_encoder_delta(extractor.model, latest["encoder_delta"])
        head.load_state_dict(latest["head_state"])
        optimizer.load_state_dict(latest["optimizer_state"])
        scheduler.load_state_dict(latest["scheduler_state"])
        best = latest["best"]
        first_epoch = int(latest["epoch"])
        history = list(latest.get("history", []))
        # The recorder holds no model weights, only per-epoch scalars, so a
        # resumed job rebuilds it exactly by replaying the recorded history
        # rather than needing it serialised alongside the optimizer state. The
        # weights it will select among are the per-epoch files already on disk.
        for entry in history:
            recorder.update(entry["epoch"], entry["val_loss"], entry["val_auc"],
                            entry["val_balanced_accuracy"])
    elif restart:
        latest_path.unlink(missing_ok=True)
        best_path.unlink(missing_ok=True)
        prediction_path.unlink(missing_ok=True)
        summary_path.unlink(missing_ok=True)
        for stale in output.glob(f"finetune_{tag}_epoch*.pt"):
            stale.unlink(missing_ok=True)

    use_amp = resolved_device.type == "cuda" and settings["precision"] == "bf16"
    train_positions = indices["train"]
    gradient_audit: dict[str, Any] | None = None
    # Snapshot the pretrained encoder so the run can report how far finetuning
    # actually moved it. The original B cells were described as "warm-started
    # joint finetune" but at encoder_lr=5e-4 the encoder ended up 28.8-38.7%
    # away from the MASS checkpoint in relative L2 -- retraining, not adapting.
    # Recording it makes that visible in the summary instead of needing a
    # separate offline diff.
    base_encoder = {name: parameter.detach().cpu().clone().float()
                    for name, parameter in extractor.model.named_parameters()
                    if parameter.requires_grad}

    def encoder_drift() -> float:
        numerator = denominator = 0.0
        for name, parameter in extractor.model.named_parameters():
            if name not in base_encoder:
                continue
            base = base_encoder[name]
            current = parameter.detach().cpu().float()
            numerator += float(((current - base) ** 2).sum())
            denominator += float((base ** 2).sum())
        return math.sqrt(numerator / denominator) if denominator > 0 else 0.0

    # Validation metrics of the warm-started model *before any optimizer step*.
    # This is the direct check on whether warm-starting survives: the old runs
    # went from the probe's ~0.85 to 0.50-0.61 within epoch 1 and there was no
    # recorded epoch-0 value to compare against, so the collapse was only
    # visible by inference.
    warm_start_baseline: dict[str, Any] | None = None
    if first_epoch == 0:
        baseline_rows = [rows[index] for index in indices["validation"]]
        baseline_probability = _infer(extractor, head, baseline_rows, resolved_device,
                                      settings["precision"])
        baseline_truth = labels[indices["validation"]]
        warm_start_baseline = {
            "head_warm_started_from": head_warm_start,
            "val_balanced_accuracy": _balanced_accuracy(
                baseline_truth, (baseline_probability >= 0.5).astype(np.int64)),
            "val_auc": _binary_auc(baseline_truth, baseline_probability),
        }
        print(f"[finetune] seed={seed} epoch=  0 (warm-started head, encoder untouched) "
              f"val_auc={warm_start_baseline['val_auc']:.4f} "
              f"val_ba={warm_start_baseline['val_balanced_accuracy']:.4f} "
              f"warm_start={head_warm_start}", flush=True)
    for epoch in range(first_epoch, int(settings["max_epochs"])):
        extractor.model.train()
        head.train()
        generator = torch.Generator().manual_seed(seed * 1_000_003 + epoch)
        order = torch.randperm(len(train_positions), generator=generator).numpy()
        progress = tqdm(range(0, len(order), accumulation),
                        desc=f"finetune {encoder} seed={seed} epoch={epoch + 1}", unit="step")
        epoch_losses: list[float] = []
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
            epoch_losses.append(float(np.mean(losses)))
            progress.set_postfix(loss=f"{np.mean(losses):.4f}")

        val_rows = [rows[index] for index in indices["validation"]]
        val_probability = _infer(extractor, head, val_rows, resolved_device, settings["precision"])
        val_truth = labels[indices["validation"]]
        val_prediction = (val_probability >= 0.5).astype(np.int64)
        # Previously this took log() of the probabilities directly, so a
        # saturated head (probability exactly 0.0 or 1.0, which happens as soon
        # as the training split is memorised) produced val_loss = inf. Those
        # were being read as "training diverged" when nothing had diverged at
        # all -- the same epochs still scored val BA 0.83-0.86. Clamping keeps
        # the loss finite and monotone in the same direction, and puts it on
        # comparable footing with the probe's logit-space cross-entropy.
        clamped = np.clip(val_probability.astype(np.float64), 1e-7, 1.0 - 1e-7)
        val_loss = float(nn.functional.cross_entropy(
            torch.from_numpy(np.stack((1.0 - clamped, clamped), axis=1)).log(),
            torch.from_numpy(val_truth), weight=class_weights.cpu().double(),
        ).item())
        val_balanced = _balanced_accuracy(val_truth, val_prediction)
        val_auc = _binary_auc(val_truth, val_probability)
        epoch_train_loss = float(np.mean(epoch_losses))
        record = recorder.update(epoch + 1, val_loss, val_auc, val_balanced, epoch_train_loss)
        record["train_loss"] = epoch_train_loss
        record["encoder_lr"] = float(optimizer.param_groups[0]["lr"])
        record["head_lr"] = float(optimizer.param_groups[1]["lr"])
        # Kept so an alternative selection rule can be evaluated offline without
        # retraining -- the paired bootstrap that would calibrate the one-
        # standard-error tolerance needs each epoch's predictions, not just its
        # AUC. See the module docstring in encoderbench.selection.
        record["val_probability"] = [round(float(value), 6) for value in val_probability]
        if record_test_trace:
            # DIAGNOSTIC ONLY. This is the test split, evaluated every epoch so
            # that the epoch-selection rule can finally be settled offline: the
            # round-3 traces established that the paired standard error of the
            # AUC difference is ~0.018 (half the marginal 0.034) and that a
            # one-paired-SE rule would move selection 3-6 epochs earlier, but
            # with no per-epoch test scores there was still no way to say
            # whether earlier is better. Nothing in training, early stopping,
            # threshold fitting or epoch selection reads this array -- it is
            # written to the summary and never back into the run. Turn
            # `record_test_trace` off for numbers meant to be quoted.
            trace = _infer(extractor, head, [rows[index] for index in indices["test"]],
                           resolved_device, settings["precision"])
            record["test_probability_diagnostic_only"] = [round(float(v), 6) for v in trace]
        history.append(dict(record))
        print(f"[finetune] seed={seed} epoch={epoch + 1:3d} "
              f"train_loss={record['train_loss']:.4f} val_loss={val_loss:.4f} "
              f"val_auc={val_auc:.4f} val_ba={val_balanced:.4f} "
              f"smooth_auc={record['smoothed_val_auc']:.4f} eligible={int(record['eligible'])} "
              f"stale={record['stale_epochs']} loss_min_ep={record['val_loss_min_epoch']} "
              f"enc_lr={record['encoder_lr']:.2e} head_lr={record['head_lr']:.2e}", flush=True)
        _atomic_torch_save({
            "protocol_sha256": protocol, "encoder": encoder, "seed": seed, "shuffled": shuffled,
            "epoch": epoch + 1,
            "encoder_delta": _encoder_delta(extractor.model),
            "head_state": {name: value.detach().cpu() for name, value in head.state_dict().items()},
        }, epoch_state_path(epoch + 1))
        stale_epochs = recorder.stale_epochs
        _atomic_torch_save({
            "protocol_sha256": protocol, "epoch": epoch + 1,
            "encoder_delta": _encoder_delta(extractor.model),
            "head_state": {name: value.detach().cpu() for name, value in head.state_dict().items()},
            "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
            "best": best, "stale_epochs": stale_epochs, "history": history,
        }, latest_path)
        if recorder.stop_requested():
            if recorder.overfit_stop_epoch is not None:
                reason = (f"overfitting detected: training loss falling while validation loss "
                          f"rises, sustained over {recorder.overfit_window} epochs "
                          f"(first flagged at epoch {recorder.overfit_stop_epoch})")
            else:
                reason = (f"{recorder.stale_epochs} epochs without a validation-loss "
                          f"improvement; minimum was epoch {recorder.best_loss_epoch}")
            print(f"[finetune] seed={seed} early stop at epoch {epoch + 1} -- {reason}", flush=True)
            break

    if not recorder.records:
        raise RuntimeError(f"Fine-tuning recorded no epoch for {encoder} seed {seed}")
    val_truth = labels[indices["validation"]]
    best = dict(choose_epoch(recorder.records, knobs["guard"],
                             int((val_truth == 1).sum()), int((val_truth == 0).sum()), tolerance))
    best["balanced_accuracy"] = best["val_balanced_accuracy"]
    best["loss"] = best["val_loss"]
    selected_val_probability = np.asarray(best.pop("val_probability"), dtype=np.float64)
    print(f"[finetune] seed={seed} selected epoch {best['epoch']} ({best['selection_rule']})",
          flush=True)
    selected_path = epoch_state_path(best["epoch"])
    if not selected_path.is_file():
        raise RuntimeError(f"Selected epoch {best['epoch']} has no saved weights: {selected_path}")
    selected = torch.load(selected_path, map_location=resolved_device, weights_only=False)
    _load_encoder_delta(extractor.model, selected["encoder_delta"])
    head.load_state_dict(selected["head_state"])
    _atomic_torch_save({
        "protocol_sha256": protocol, "encoder": encoder, "seed": seed, "shuffled": shuffled,
        "encoder_delta": selected["encoder_delta"], "head_state": selected["head_state"],
        "selection": best, "parameter_budget": budget_audit,
        "gradient_audit": gradient_audit, "history": history,
    }, best_path)
    # The per-epoch weights existed only so selection could range over the whole
    # curve; keeping ~800 MB per seed after the fact serves nothing, and the
    # per-epoch validation predictions needed to re-run selection offline are
    # already in `history`.
    for stale in output.glob(f"finetune_{tag}_epoch*.pt"):
        stale.unlink(missing_ok=True)

    # Threshold fitted on validation using the selected weights -- see
    # training.run_probe for the measurement that motivated this.
    threshold = tune_threshold(val_truth, selected_val_probability)
    print(f"[finetune] seed={seed} decision threshold fitted on validation: {threshold:.4f}",
          flush=True)
    test_rows = [rows[index] for index in indices["test"]]
    probability = _infer(extractor, head, test_rows, resolved_device, settings["precision"])
    predictions = _write_predictions(prediction_path, eval_cache, indices["test"],
                                      probability, positive, threshold)
    result = {
        "kind": "encoder_finetune", "disease": disease, "encoder": encoder, "seed": seed,
        "shuffled_labels": shuffled, "positive_label": positive, "selection": best,
        "head_warm_started_from": head_warm_start,
        "warm_start_baseline": warm_start_baseline,
        # Measured on the *selected* weights, which are loaded above.
        "encoder_relative_l2_change": encoder_drift(),
        "overfit_stop_epoch": recorder.overfit_stop_epoch,
        "decision_threshold": float(threshold),
        "parameter_budget": budget_audit,
        "gradient_audit": gradient_audit,
        "history": history,
        "trainable_parameters": budget_audit["trainable_encoder_parameters"]
                                + trainable_parameter_count(head),
        "head_trainable_parameters": trainable_parameter_count(head),
        "token_shape": [64, width], "checkpoint": str(best_path),
        "base_checkpoint": checkpoint_identifier(config["checkpoints"][encoder]),
        "predictions": str(predictions),
        "metrics": _summary(eval_cache, indices["test"], probability, positive,
                            int(config["evaluation"]["bootstrap_samples"]),
                            int(config["evaluation"]["bootstrap_seed"]), threshold),
        "metrics_at_half": _summary(eval_cache, indices["test"], probability, positive,
                                    int(config["evaluation"]["bootstrap_samples"]),
                                    int(config["evaluation"]["bootstrap_seed"]), 0.5),
    }
    write_json(summary_path, result)
    return result
