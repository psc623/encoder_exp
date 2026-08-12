"""Four fine-tune modes sharing run_finetune's training loop, differing only in
which head is used, whether it is warm-started, and whether it (and how much
of the encoder) is trainable:

  mode 3: attention head, warm-started from run_probe's checkpoint, FROZEN -- encoder only trains
  mode 4: linear head,    warm-started from run_linear_probe's fit,   FROZEN -- encoder only trains
  mode 5: attention head, randomly initialized ("from zero"), trained jointly with a fully-unfrozen encoder
  mode 6: linear head,    randomly initialized ("from zero"), trained jointly with a fully-unfrozen encoder

Modes 3/4 reuse run_finetune's existing 8M parameter_budget (configure_parameter_budget)
so they stay comparable to the original bsnip2:mass finetune numbers already in
final_report.md. Modes 5/6 are explicitly NOT parameter-budgeted per instruction
("不要限制算力"): the whole encoder is unfrozen.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Literal

import numpy as np

from encoderbench.cache import load_cache
from encoderbench.finetune import (
    _aligned_cache, _atomic_torch_save, _balanced_accuracy, _encoder_delta,
    _evaluation_cache, _infer, _load_encoder_delta, _protocol_digest, configure_parameter_budget,
)
from encoderbench.manifest import read_manifest
from encoderbench.training import _class_weights, _summary, _write_predictions
from encoderbench.utils import checkpoint_identifier, ensure_parent, set_seed, write_json

HeadKind = Literal["attention", "linear"]


_EFFECTIVELY_UNLIMITED_BUDGET = 10**15


def configure_full_finetune(extractor: Any) -> dict[str, Any]:
    """Unfreeze every fine-tune-eligible group with no budget cap -- used by
    modes 5/6 ("don't limit compute"). Deliberately reuses
    configure_parameter_budget rather than `extractor.model.requires_grad_(True)`:
    MASS (and the other encoders) are wrapped segmentation/backbone nets whose
    `forward_tokens` only runs the path up to the configured tap layer, so a
    blanket unfreeze also marks downstream-of-tap modules (e.g. MASS's decoder)
    trainable even though they never appear in the forward graph -- those
    parameters get no gradient and trip run_finetune_variant's gradient audit.
    extractor.finetune_groups() already enumerates only the modules that are
    actually upstream of the tap; passing a budget far above any real
    parameter count makes configure_parameter_budget select all of them.
    """
    return configure_parameter_budget(extractor, _EFFECTIVELY_UNLIMITED_BUDGET)


def _build_head(head_kind: HeadKind, n_tokens: int, width: int, mean: "torch.Tensor",
                std: "torch.Tensor", hidden_size: int) -> "torch.nn.Module":
    if head_kind == "attention":
        from encoderbench.models import AttentionPoolHead
        # AttentionPoolHead's Linear(width,hidden) is applied per-token and
        # broadcasts over however many tokens there are -- no n_tokens needed.
        return AttentionPoolHead(width, mean, std, hidden_size)
    from encoderbench.linear_head import LinearPoolHead
    return LinearPoolHead(n_tokens, width, mean, std).to_module()


def _warm_start(head: "torch.nn.Module", head_kind: HeadKind, config: dict[str, Any],
                disease: str, encoder: str, seed: int,
                warm_start_dir: "Path | str | None" = None,
                warm_start_tag: str | None = None) -> str | None:
    import torch

    # warm_start_dir/tag let a caller point at a specific frozen-probe run --
    # needed by the repeated-split protocol, where mode3/4 must warm-start from
    # the mode1/2 fit for the *same repeat*, not from the canonical single-split
    # checkpoint. Unset == the original canonical locations.
    tag = warm_start_tag if warm_start_tag is not None else f"seed_{seed}"
    if head_kind == "attention":
        directory = (Path(warm_start_dir) if warm_start_dir is not None
                     else Path(config["output_root"]) / "attention" / disease / encoder)
        checkpoint = directory / f"probe_{tag}.pt"
        if not checkpoint.is_file():
            return None
        probe = torch.load(checkpoint, map_location="cpu", weights_only=False)
        head.load_state_dict(probe["model_state"])
        return str(checkpoint)
    directory = (Path(warm_start_dir) if warm_start_dir is not None
                 else Path(config["output_root"]) / "linear" / disease / encoder)
    checkpoint = directory / f"linear_{tag}.pt"
    if not checkpoint.is_file():
        return None
    fit = torch.load(checkpoint, map_location="cpu", weights_only=False)
    with torch.no_grad():
        head.linear.weight[1].copy_(torch.from_numpy(fit["w"]))
        head.linear.bias[1].fill_(float(fit["b"]))
        head.linear.weight[0].zero_()
        head.linear.bias[0].zero_()
    return str(checkpoint)


def run_finetune_variant(manifest_path: str | Path, cache_path: str | Path, disease: str,
                         encoder: str, config: dict[str, Any], output_dir: str | Path, seed: int,
                         head_kind: HeadKind, train_head: bool, warm_start_head: bool,
                         parameter_budget: int | None, device: str = "cuda",
                         restart: bool = False, sample_log_path: str | Path | None = None,
                         native_tokens: bool = False,
                         pooled_grid_override: "Sequence[int] | None" = None,
                         attention_hidden_size: int | None = None,
                         splits_override: "np.ndarray | None" = None,
                         tag_suffix: str = "",
                         settings_override: dict[str, Any] | None = None,
                         warm_start_dir: "Path | str | None" = None,
                         warm_start_tag: str | None = None,
                         ) -> dict[str, Any]:
    import torch
    from torch import nn
    from tqdm.auto import tqdm

    from encoderbench.extractors import build_extractor

    settings = dict(config["finetune"])
    if settings_override:
        # Lets the caller retune finetune hyperparameters (learning rates,
        # weight decays, epoch budget, ...) per run without editing the shared
        # config every other experiment reads. Recorded in the summary JSON.
        settings.update(settings_override)
    set_seed(seed)
    resolved_device = torch.device(device)
    rows = read_manifest(manifest_path)
    cache = load_cache(cache_path)
    if splits_override is not None:
        # Repeated-random-split protocol: overwrite BOTH the manifest rows' and
        # the cache's split labels, in the same row order, so _aligned_cache's
        # consistency check still means something instead of being bypassed.
        if len(splits_override) != len(rows) or len(splits_override) != len(cache.splits):
            raise ValueError(f"splits_override has {len(splits_override)} rows; manifest has "
                             f"{len(rows)} and cache has {len(cache.splits)}")
        by_key = {(str(f), str(s)): i for i, (f, s) in
                  enumerate(zip(cache.file_ids, cache.subject_ids))}
        for row, split in zip(rows, splits_override):
            row["split"] = str(split)
            cache.splits[by_key[(row["file_id"], row["subject_id"])]] = str(split)
        cache.validate()
    mean_np, std_np = _aligned_cache(rows, cache)
    n_tokens = int(cache.features.shape[1])
    width = int(cache.features.shape[-1])
    mean, std = torch.from_numpy(mean_np), torch.from_numpy(std_np)
    eval_cache = _evaluation_cache(rows, encoder, width, disease)
    positive = "AD" if disease == "ad" else ("SZ" if disease == "bsnip2" else "SCZ")
    labels = np.asarray([1 if row["group"] == positive else 0 for row in rows], dtype=np.int64)
    indices = {split: np.asarray([index for index, row in enumerate(rows) if row["split"] == split])
               for split in ("train", "validation", "test")}
    train_targets = labels[indices["train"]].copy()

    # SynthSeg's pooling has no trainable parameters (only the post-pooling
    # adapter does, see finetune_groups), so re-loading+resizing+pooling the
    # raw posterior on every forward pass during finetune is pure waste --
    # ~6s/sample measured, which would make a 12-epoch/216-sample run take
    # hours. The cache we already loaded above has exactly these pooled
    # tokens; hand them to the extractor keyed by path so forward_tokens can
    # skip straight to the (cheap, trainable) adapter. No effect on MASS/other
    # encoders' extractors, which don't accept this kwarg.
    pooled_cache = None
    if encoder == "synthseg":
        file_id_to_row = {index: file_id for index, file_id in enumerate(cache.file_ids)}
        row_index_by_file_id = {file_id: index for index, file_id in file_id_to_row.items()}
        pooled_cache = {
            row["path"]: torch.from_numpy(cache.features[row_index_by_file_id[row["file_id"]]])
            for row in rows if row["file_id"] in row_index_by_file_id
        }

    # native_tokens must match how cache_path's features were extracted (the
    # extractor's forward_tokens has to reproduce the exact token layout the
    # frozen mean/std/warm-start weights were fit against) -- caller's job to
    # pass the same value used when the cache was built.
    extractor = build_extractor(encoder, config, str(resolved_device), native_tokens=native_tokens,
                                pooled_grid_override=pooled_grid_override, pooled_cache=pooled_cache)
    budget_audit = (configure_parameter_budget(extractor, int(parameter_budget))
                    if parameter_budget is not None else configure_full_finetune(extractor))

    # AttentionPoolHead's hidden_size must match whatever mode1's probe checkpoint
    # actually used (it can differ from config["finetune"]["hidden_size"] when a
    # hyperparameter sweep changed the probe's hidden_size but finetune's config
    # section was never touched) -- otherwise warm-starting a differently-shaped
    # head raises a state_dict shape-mismatch error. Caller passes the value
    # mode1 actually used; None falls back to the (unswept) finetune config value,
    # which matches historical behavior for every run this bug didn't affect.
    resolved_hidden_size = (int(attention_hidden_size) if attention_hidden_size is not None
                            else int(settings["hidden_size"]))
    head = _build_head(head_kind, n_tokens, width, mean, std, resolved_hidden_size)
    head_warm_start = (_warm_start(head, head_kind, config, disease, encoder, seed,
                               warm_start_dir, warm_start_tag) if warm_start_head else None)
    if warm_start_head and head_warm_start is None:
        raise RuntimeError(
            f"warm_start_head=True but no {head_kind} checkpoint was found for "
            f"disease={disease} encoder={encoder} seed={seed} -- refusing to silently "
            f"fall back to a randomly/zero-initialized frozen head, since that would "
            f"produce a chance-level result that looks like a completed run. Run "
            f"mode1 (attention) or mode2 (linear) for this seed first."
        )
    head = head.to(resolved_device)
    for parameter in head.parameters():
        parameter.requires_grad = bool(train_head)

    class_weights = _class_weights(train_targets).to(resolved_device)
    loss_function = nn.CrossEntropyLoss(weight=class_weights)
    encoder_parameters = [p for p in extractor.model.parameters() if p.requires_grad]
    param_groups = [{"params": encoder_parameters, "lr": float(settings["encoder_learning_rate"]),
                     "weight_decay": float(settings["weight_decay"])}]
    if train_head:
        # The head gets its own weight_decay knob. The linear head is ~1e6
        # parameters fit on a few hundred volumes, so it needs far stronger L2
        # than the encoder does; sharing one decay across both (the old
        # behaviour) is what let it diverge to an infinite validation loss.
        # Defaults to the shared value, so unset == previous behaviour.
        head_decay = float(settings.get("head_weight_decay", settings["weight_decay"]))
        param_groups.append({"params": list(head.parameters()),
                             "lr": float(settings["head_learning_rate"]),
                             "weight_decay": head_decay})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=float(settings["weight_decay"]))
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
    mode_tag = f"{head_kind}_{'trainhead' if train_head else 'headfrozen'}_{'warm' if warm_start_head else 'cold'}"
    tag = f"{mode_tag}_seed_{seed}{tag_suffix}"
    best_path = output / f"finetune_{tag}.pt"
    latest_path = output / f"finetune_{tag}_latest.pt"
    prediction_path = output / f"finetune_{tag}_predictions.csv"
    summary_path = output / f"finetune_{tag}_summary.json"
    protocol = _protocol_digest(manifest_path, encoder, seed, False,
                                {**settings, "mode_tag": mode_tag, "parameter_budget": parameter_budget}, config)
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
        for p in (latest_path, best_path, prediction_path, summary_path):
            p.unlink(missing_ok=True)

    use_amp = resolved_device.type == "cuda" and settings["precision"] == "bf16"
    train_positions = indices["train"]
    gradient_audit: dict[str, Any] | None = None
    log_handle = open(ensure_parent(sample_log_path), "a", encoding="utf-8") if sample_log_path else None
    try:
        for epoch in range(first_epoch, int(settings["max_epochs"])):
            extractor.model.train()
            head.train()
            generator = torch.Generator().manual_seed(seed * 1_000_003 + epoch)
            order = torch.randperm(len(train_positions), generator=generator).numpy()
            progress = tqdm(range(0, len(order), accumulation),
                            desc=f"finetune {encoder} {mode_tag} seed={seed} epoch={epoch + 1}", unit="step")
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
                    if log_handle:
                        prob1 = float(torch.softmax(logits.float(), dim=-1)[0, 1].detach().cpu())
                        log_handle.write(f"{mode_tag},{seed},{epoch+1},train,{row['file_id']},"
                                         f"{row['subject_id']},{row['group']},{prob1:.6f}\n")
                if gradient_audit is None:
                    missing = [n for n, p in extractor.model.named_parameters()
                              if p.requires_grad and p.grad is None]
                    if missing:
                        raise RuntimeError(f"Fine-tune gradient audit failed for {encoder}: {missing[:8]}")
                    gradient_audit = {"encoder_parameters_with_gradients": sum(
                        p.numel() for p in encoder_parameters if p.grad is not None),
                        "missing_gradient_parameters": [], "passed": True}
                clip_params = encoder_parameters + (list(head.parameters()) if train_head else [])
                torch.nn.utils.clip_grad_norm_(clip_params, float(settings["gradient_clip"]))
                optimizer.step()
                scheduler.step()
                progress.set_postfix(loss=f"{np.mean(losses):.4f}")

            val_rows = [rows[i] for i in indices["validation"]]
            val_probability = _infer(extractor, head, val_rows, resolved_device, settings["precision"])
            val_truth = labels[indices["validation"]]
            val_prediction = (val_probability >= 0.5).astype(np.int64)
            val_loss = float(nn.functional.cross_entropy(
                torch.from_numpy(np.stack((1.0 - val_probability, val_probability), axis=1)).log(),
                torch.from_numpy(val_truth), weight=class_weights.cpu()).item())
            if log_handle:
                for row, truth, prob in zip(val_rows, val_truth, val_probability):
                    log_handle.write(f"{mode_tag},{seed},{epoch+1},validation,{row['file_id']},"
                                     f"{row['subject_id']},{row['group']},{float(prob):.6f}\n")
                log_handle.flush()
            candidate = {"epoch": epoch + 1, "balanced_accuracy": _balanced_accuracy(val_truth, val_prediction),
                        "loss": val_loss}
            improved = best is None or (candidate["balanced_accuracy"], -candidate["loss"]) > (
                best["balanced_accuracy"], -best["loss"])
            if improved:
                best = candidate
                stale_epochs = 0
                _atomic_torch_save({
                    "protocol_sha256": protocol, "encoder": encoder, "seed": seed,
                    "encoder_delta": _encoder_delta(extractor.model),
                    "head_state": {k: v.detach().cpu() for k, v in head.state_dict().items()},
                    "selection": best, "parameter_budget": budget_audit, "gradient_audit": gradient_audit,
                }, best_path)
            else:
                stale_epochs += 1
            assert best is not None
            _atomic_torch_save({
                "protocol_sha256": protocol, "epoch": epoch + 1,
                "encoder_delta": _encoder_delta(extractor.model),
                "head_state": {k: v.detach().cpu() for k, v in head.state_dict().items()},
                "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
                "best": best, "stale_epochs": stale_epochs,
            }, latest_path)
            if stale_epochs >= int(settings["patience"]):
                break
    finally:
        if log_handle:
            log_handle.close()

    if not best_path.is_file():
        raise RuntimeError(f"Fine-tuning produced no best checkpoint: {best_path}")
    selected = torch.load(best_path, map_location=resolved_device, weights_only=False)
    _load_encoder_delta(extractor.model, selected["encoder_delta"])
    head.load_state_dict(selected["head_state"])
    best = selected["selection"]
    test_rows = [rows[i] for i in indices["test"]]
    probability = _infer(extractor, head, test_rows, resolved_device, settings["precision"])
    if log_handle_path := sample_log_path:
        with open(ensure_parent(log_handle_path), "a", encoding="utf-8") as handle:
            for row, prob in zip(test_rows, probability):
                handle.write(f"{mode_tag},{seed},FINAL,test,{row['file_id']},"
                             f"{row['subject_id']},{row['group']},{float(prob):.6f}\n")
    predictions = _write_predictions(prediction_path, eval_cache, indices["test"], probability, positive)
    result = {
        "kind": f"finetune_variant_{mode_tag}", "disease": disease, "encoder": encoder, "seed": seed,
        "head_kind": head_kind, "train_head": train_head, "warm_start_head": warm_start_head,
        "positive_label": positive, "selection": best, "head_warm_started_from": head_warm_start,
        "parameter_budget": budget_audit, "gradient_audit": selected["gradient_audit"],
        "trainable_parameters": budget_audit["trainable_encoder_parameters"]
                                + (sum(p.numel() for p in head.parameters()) if train_head else 0),
        "head_trainable_parameters": sum(p.numel() for p in head.parameters()) if train_head else 0,
        "token_shape": [64, width], "checkpoint": str(best_path),
        "base_checkpoint": checkpoint_identifier(config["checkpoints"][encoder]),
        "predictions": str(predictions),
        "settings_used": {key: settings[key] for key in sorted(settings)},
        "settings_override": settings_override or {},
        "metrics": _summary(eval_cache, indices["test"], probability, positive,
                            int(config["evaluation"]["bootstrap_samples"]), int(config["evaluation"]["bootstrap_seed"])),
    }
    write_json(summary_path, result)
    return result
