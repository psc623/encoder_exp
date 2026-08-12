#!/usr/bin/env python3
"""Subject-level Alzheimer-vs-control evaluation with pretrained BIOT."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch import nn
from torch.utils.data import DataLoader

from biot_model import (
    BIOTACClassifier,
    BIOTEncoder,
    configure_encoder_trainability,
    load_pretrained_encoder,
)
from eeg_ad_data import (
    CachedSubject,
    PreprocessingConfig,
    SubjectWindowDataset,
    load_ac_subjects,
    prepare_all_caches,
)


DEFAULT_DATASET = Path("/net/projects2/litian-lab/scpan/dataset/EEG_AD")
DEFAULT_CHECKPOINT = Path(
    "/net/projects2/litian-lab/scpan/github_repo/BIOT/pretrained-models/"
    "EEG-six-datasets-18-channels.ckpt"
)
DEFAULT_OUTPUT = Path("/net/projects2/litian-lab/scpan/encoders/EEG_exp/outputs/biot_ac")
DEFAULT_CACHE = Path("/net/projects2/litian-lab/scpan/encoders/EEG_exp/cache")
REGIMES = ("pretrained_frozen", "pretrained_last2", "pretrained_full", "scratch")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def select_balanced_accuracy_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    unique = np.unique(probabilities)
    if unique.size == 1:
        return 0.5
    midpoints = (unique[:-1] + unique[1:]) / 2.0
    candidates = np.concatenate(
        ([0.0], midpoints, [float(np.nextafter(1.0, 2.0))])
    )
    scores = np.asarray(
        [balanced_accuracy_score(labels, probabilities >= threshold) for threshold in candidates]
    )
    best_score = scores.max()
    best = candidates[np.isclose(scores, best_score)]
    return float(best[np.argmin(np.abs(best - 0.5))])


def subject_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    subject_ids: list[str],
    threshold: float,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    grouped: dict[str, dict[str, Any]] = {}
    for label, probability, subject_id in zip(labels, probabilities, subject_ids, strict=True):
        entry = grouped.setdefault(subject_id, {"labels": set(), "probabilities": []})
        entry["labels"].add(int(label))
        entry["probabilities"].append(float(probability))

    rows: list[dict[str, Any]] = []
    for subject_id in sorted(grouped):
        entry = grouped[subject_id]
        if len(entry["labels"]) != 1:
            raise ValueError(f"Subject {subject_id} has inconsistent labels: {entry['labels']}")
        label = next(iter(entry["labels"]))
        probability = float(np.mean(entry["probabilities"]))
        rows.append(
            {
                "subject_id": subject_id,
                "label": label,
                "probability": probability,
                "prediction": int(probability >= threshold),
                "n_windows": len(entry["probabilities"]),
            }
        )

    subject_labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
    subject_probabilities = np.asarray([row["probability"] for row in rows])
    predictions = subject_probabilities >= threshold
    metrics = {
        "balanced_accuracy": float(balanced_accuracy_score(subject_labels, predictions)),
        "roc_auc": float(roc_auc_score(subject_labels, subject_probabilities)),
        "threshold": float(threshold),
        "n_subjects": int(len(rows)),
        "n_windows": int(len(labels)),
    }
    return metrics, rows


@torch.no_grad()
def evaluate(
    model: BIOTACClassifier,
    dataset: SubjectWindowDataset,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    threshold: float,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    all_labels: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []
    all_subject_ids: list[str] = []
    for windows, labels, subject_ids in loader:
        logits = model(windows.to(device, non_blocking=True))
        all_labels.append(labels.numpy())
        all_probabilities.append(torch.sigmoid(logits).cpu().numpy())
        all_subject_ids.extend(subject_ids)
    return subject_metrics(
        np.concatenate(all_labels),
        np.concatenate(all_probabilities),
        all_subject_ids,
        threshold,
    )


def make_model(
    regime: str,
    checkpoint: Path,
    device: torch.device,
) -> BIOTACClassifier:
    if regime == "scratch":
        encoder = BIOTEncoder(n_channels=18)
    else:
        encoder = load_pretrained_encoder(checkpoint)
    configure_encoder_trainability(encoder, regime)
    return BIOTACClassifier(encoder).to(device)


def optimizer_for(
    model: BIOTACClassifier,
    regime: str,
    encoder_lr: float,
    head_lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    groups: list[dict[str, Any]] = [
        {"params": list(model.classifier.parameters()), "lr": head_lr}
    ]
    encoder_parameters = [
        parameter for parameter in model.encoder.parameters() if parameter.requires_grad
    ]
    if encoder_parameters:
        lr = encoder_lr if regime != "scratch" else max(encoder_lr, 1e-4)
        groups.append({"params": encoder_parameters, "lr": lr})
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


def train_fold(
    *,
    regime: str,
    fold: int,
    train_subjects: list[CachedSubject],
    validation_subjects: list[CachedSubject],
    test_subjects: list[CachedSubject],
    checkpoint: Path,
    output_dir: Path,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    fold_seed = args.seed + 10_000 * fold + REGIMES.index(regime) * 100
    set_seed(fold_seed)
    model = make_model(regime, checkpoint, device)
    optimizer = optimizer_for(
        model,
        regime,
        args.encoder_lr,
        args.head_lr,
        args.weight_decay,
    )

    train_dataset = SubjectWindowDataset(
        train_subjects,
        windows_per_subject=args.train_windows_per_subject,
        seed=fold_seed,
    )
    validation_dataset = SubjectWindowDataset(
        validation_subjects,
        windows_per_subject=args.eval_windows_per_subject,
        seed=fold_seed,
    )
    test_dataset = SubjectWindowDataset(
        test_subjects,
        windows_per_subject=args.eval_windows_per_subject,
        seed=fold_seed,
    )

    positives = sum(subject.label == 1 for subject in train_subjects)
    negatives = len(train_subjects) - positives
    positive_weight = torch.tensor([negatives / positives], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=args.amp and device.type == "cuda",
    )

    best_auc = -float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    encoder_is_frozen = not any(
        parameter.requires_grad for parameter in model.encoder.parameters()
    )

    for epoch in range(args.epochs):
        train_dataset.set_epoch(epoch)
        loader_generator = torch.Generator().manual_seed(fold_seed + epoch)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            generator=loader_generator,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        model.train()
        if encoder_is_frozen:
            model.encoder.eval()
        total_loss = 0.0
        total_items = 0
        for windows, labels, _ in train_loader:
            windows = windows.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=args.amp and device.type == "cuda",
            ):
                logits = model(windows)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach()) * len(labels)
            total_items += len(labels)

        validation_default, validation_rows = evaluate(
            model,
            validation_dataset,
            device,
            args.eval_batch_size,
            args.num_workers,
            threshold=0.5,
        )
        validation_labels = np.asarray([row["label"] for row in validation_rows])
        validation_probabilities = np.asarray(
            [row["probability"] for row in validation_rows]
        )
        selected_threshold = select_balanced_accuracy_threshold(
            validation_labels, validation_probabilities
        )
        validation_metrics, _ = subject_metrics(
            validation_labels,
            validation_probabilities,
            [row["subject_id"] for row in validation_rows],
            selected_threshold,
        )
        row = {
            "epoch": epoch + 1,
            "train_loss": total_loss / max(total_items, 1),
            "val_auc": validation_default["roc_auc"],
            "val_ba": validation_metrics["balanced_accuracy"],
            "val_threshold": selected_threshold,
        }
        history.append(row)
        print(
            f"[{regime} fold={fold} epoch={epoch + 1:03d}] "
            f"loss={row['train_loss']:.5f} val_BA={row['val_ba']:.4f} "
            f"val_AUC={row['val_auc']:.4f} threshold={selected_threshold:.4f}",
            flush=True,
        )

        if row["val_auc"] > best_auc + args.min_delta:
            best_auc = row["val_auc"]
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= args.patience:
            break

    if best_state is None:
        raise RuntimeError("Training produced no valid checkpoint")
    model.load_state_dict(best_state, strict=True)

    validation_default, validation_rows = evaluate(
        model,
        validation_dataset,
        device,
        args.eval_batch_size,
        args.num_workers,
        threshold=0.5,
    )
    threshold = select_balanced_accuracy_threshold(
        np.asarray([row["label"] for row in validation_rows]),
        np.asarray([row["probability"] for row in validation_rows]),
    )
    validation_metrics, validation_rows = evaluate(
        model,
        validation_dataset,
        device,
        args.eval_batch_size,
        args.num_workers,
        threshold,
    )
    test_metrics, test_rows = evaluate(
        model,
        test_dataset,
        device,
        args.eval_batch_size,
        args.num_workers,
        threshold,
    )

    fold_dir = output_dir / regime / f"fold_{fold}"
    write_rows(fold_dir / "history.csv", history)
    write_rows(fold_dir / "validation_subject_predictions.csv", validation_rows)
    test_rows_with_fold = [{"fold": fold, **row} for row in test_rows]
    write_rows(fold_dir / "test_subject_predictions.csv", test_rows_with_fold)
    split = {
        "train": [subject.subject_id for subject in train_subjects],
        "validation": [subject.subject_id for subject in validation_subjects],
        "test": [subject.subject_id for subject in test_subjects],
    }
    metrics: dict[str, Any] = {
        "regime": regime,
        "fold": fold,
        "best_epoch": best_epoch,
        "validation": validation_metrics,
        "test": test_metrics,
        "split": split,
    }
    json_dump(fold_dir / "metrics.json", metrics)
    if not args.no_save_model:
        torch.save(
            {
                "state_dict": best_state,
                "regime": regime,
                "fold": fold,
                "threshold": threshold,
                "biot_channels": 16,
                "class_map": {"C": 0, "A": 1},
                "split": split,
            },
            fold_dir / "best_model.pt",
        )
    print(
        f"[{regime} fold={fold}] TEST BA={test_metrics['balanced_accuracy']:.4f} "
        f"AUC={test_metrics['roc_auc']:.4f}",
        flush=True,
    )
    return metrics, test_rows_with_fold


def summarize_regime(
    regime: str,
    fold_metrics: list[dict[str, Any]],
    oof_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    balanced_accuracies = np.asarray(
        [item["test"]["balanced_accuracy"] for item in fold_metrics]
    )
    aucs = np.asarray([item["test"]["roc_auc"] for item in fold_metrics])
    oof_labels = np.asarray([row["label"] for row in oof_rows])
    oof_probabilities = np.asarray([row["probability"] for row in oof_rows])
    return {
        "regime": regime,
        "folds": len(fold_metrics),
        "fold_balanced_accuracy_mean": float(balanced_accuracies.mean()),
        "fold_balanced_accuracy_std": float(
            balanced_accuracies.std(ddof=1) if len(balanced_accuracies) > 1 else 0.0
        ),
        "fold_roc_auc_mean": float(aucs.mean()),
        "fold_roc_auc_std": float(aucs.std(ddof=1) if len(aucs) > 1 else 0.0),
        "pooled_oof_roc_auc": float(roc_auc_score(oof_labels, oof_probabilities)),
        "pooled_oof_balanced_accuracy": float(
            balanced_accuracy_score(
                oof_labels,
                np.asarray([row["prediction"] for row in oof_rows]),
            )
        ),
        "n_oof_subjects": int(len(oof_rows)),
    }


def build_splits(
    subjects: list[CachedSubject],
    n_splits: int,
    validation_fraction: float,
    seed: int,
) -> list[tuple[list[int], list[int], list[int]]]:
    labels = np.asarray([subject.label for subject in subjects])
    indices = np.arange(len(subjects))
    outer = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits: list[tuple[list[int], list[int], list[int]]] = []
    for fold, (train_validation, test) in enumerate(outer.split(indices, labels)):
        inner = StratifiedShuffleSplit(
            n_splits=1,
            test_size=validation_fraction,
            random_state=seed + fold,
        )
        relative_train, relative_validation = next(
            inner.split(train_validation, labels[train_validation])
        )
        train = train_validation[relative_train]
        validation = train_validation[relative_validation]
        if set(train) & set(validation) or set(train) & set(test) or set(validation) & set(test):
            raise AssertionError("Subject leakage detected in generated split")
        splits.append((train.tolist(), validation.tolist(), test.tolist()))
    return splits


def run_inspect(args: argparse.Namespace) -> None:
    records = load_ac_subjects(args.dataset_root, use_derivatives=not args.use_raw)
    counts = {group: sum(record.group == group for record in records) for group in ("A", "C")}
    print(f"A/C subjects: {len(records)} ({counts})")
    config = PreprocessingConfig(
        target_rate=args.target_rate,
        window_seconds=args.window_seconds,
        stride_seconds=args.stride_seconds,
    )
    selected = records[: args.subjects]
    if not selected:
        raise ValueError("--subjects must be positive")
    cached = prepare_all_caches(
        selected,
        args.cache_dir,
        config,
        max_workers=args.cache_workers,
    )
    for subject in cached:
        array = np.load(subject.cache_path, mmap_mode="r")
        print(
            f"{subject.subject_id}: shape={array.shape}, dtype={array.dtype}, "
            f"finite={np.isfinite(array).all()}, mean={array.mean():.5f}, "
            f"std={array.std():.5f}"
        )
    if args.checkpoint:
        encoder = load_pretrained_encoder(args.checkpoint)
        sample = torch.from_numpy(np.array(np.load(cached[0].cache_path, mmap_mode="r")[:1]))
        encoder.eval()
        with torch.no_grad():
            embedding = encoder(sample)
        print(
            f"Checkpoint strict-load and forward OK: input={tuple(sample.shape)}, "
            f"embedding={tuple(embedding.shape)}, finite={torch.isfinite(embedding).all().item()}"
        )


def run_train(args: argparse.Namespace) -> None:
    device = choose_device(args.device)
    records = load_ac_subjects(args.dataset_root, use_derivatives=not args.use_raw)
    counts = {group: sum(record.group == group for record in records) for group in ("A", "C")}
    if counts != {"A": 36, "C": 29}:
        print(f"Warning: expected A=36/C=29, found {counts}", file=sys.stderr)
    config = PreprocessingConfig(
        target_rate=args.target_rate,
        window_seconds=args.window_seconds,
        stride_seconds=args.stride_seconds,
    )
    cached = prepare_all_caches(
        records,
        args.cache_dir,
        config,
        max_workers=args.cache_workers,
    )
    splits = build_splits(cached, args.n_splits, args.validation_fraction, args.seed)
    selected_folds = list(range(args.n_splits)) if args.folds is None else args.folds
    invalid_folds = sorted(set(selected_folds) - set(range(args.n_splits)))
    if invalid_folds:
        raise ValueError(f"Invalid fold indices: {invalid_folds}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    configuration = vars(args).copy()
    configuration["dataset_root"] = str(args.dataset_root)
    configuration["checkpoint"] = str(args.checkpoint)
    configuration["cache_dir"] = str(args.cache_dir)
    configuration["output_dir"] = str(args.output_dir)
    configuration["device_resolved"] = str(device)
    configuration["preprocessing"] = asdict(config)
    json_dump(output_dir / "config.json", configuration)

    summaries: list[dict[str, Any]] = []
    for regime in args.regimes:
        fold_metrics: list[dict[str, Any]] = []
        oof_rows: list[dict[str, Any]] = []
        for fold in selected_folds:
            train_indices, validation_indices, test_indices = splits[fold]
            metrics, test_rows = train_fold(
                regime=regime,
                fold=fold,
                train_subjects=[cached[index] for index in train_indices],
                validation_subjects=[cached[index] for index in validation_indices],
                test_subjects=[cached[index] for index in test_indices],
                checkpoint=Path(args.checkpoint),
                output_dir=output_dir,
                device=device,
                args=args,
            )
            fold_metrics.append(metrics)
            oof_rows.extend(test_rows)
        write_rows(output_dir / regime / "oof_subject_predictions.csv", oof_rows)
        summary = summarize_regime(regime, fold_metrics, oof_rows)
        json_dump(output_dir / regime / "summary.json", summary)
        summaries.append(summary)
    json_dump(output_dir / "summary.json", summaries)
    write_rows(output_dir / "summary.csv", summaries)


def add_data_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--use-raw", action="store_true", help="Use raw instead of denoised derivatives")
    parser.add_argument("--target-rate", type=int, default=200)
    parser.add_argument("--window-seconds", type=float, default=10.0)
    parser.add_argument("--stride-seconds", type=float, default=10.0)
    parser.add_argument(
        "--cache-workers",
        type=int,
        default=1,
        help="Parallel preprocessing workers (use 4 on the provided Slurm job)",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Preprocess a few subjects and test BIOT")
    add_data_arguments(inspect_parser)
    inspect_parser.add_argument("--subjects", type=int, default=2)
    inspect_parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)

    train_parser = subparsers.add_parser("train", help="Run subject-level nested train/val/test CV")
    add_data_arguments(train_parser)
    train_parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    train_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    train_parser.add_argument("--regimes", nargs="+", choices=REGIMES, default=["pretrained_frozen"])
    train_parser.add_argument("--n-splits", type=int, default=5)
    train_parser.add_argument("--folds", nargs="+", type=int)
    train_parser.add_argument("--validation-fraction", type=float, default=0.2)
    train_parser.add_argument("--epochs", type=int, default=30)
    train_parser.add_argument("--patience", type=int, default=7)
    train_parser.add_argument("--min-delta", type=float, default=1e-4)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--eval-batch-size", type=int, default=64)
    train_parser.add_argument("--train-windows-per-subject", type=int, default=20)
    train_parser.add_argument(
        "--eval-windows-per-subject",
        type=int,
        default=None,
        help="Debug-only cap; by default validation/test use every window",
    )
    train_parser.add_argument("--encoder-lr", type=float, default=1e-5)
    train_parser.add_argument("--head-lr", type=float, default=1e-3)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--num-workers", type=int, default=0)
    train_parser.add_argument("--device", default="auto")
    train_parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    train_parser.add_argument("--seed", type=int, default=2026)
    train_parser.add_argument("--no-save-model", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "inspect":
        run_inspect(args)
    elif args.command == "train":
        run_train(args)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
