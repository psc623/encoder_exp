"""Manifest construction and exact subject-level split rules."""

from __future__ import annotations

import csv
import random
from collections import Counter
from pathlib import Path
from typing import Iterable

from encoderbench.utils import ensure_parent


FIELDS = ("path", "file_id", "subject_id", "group", "is_repeat", "split")
LABELS = {"ad": ("CN", "AD"), "scz": ("CN", "SCZ"), "bsnip2": ("CN", "SZ")}


def read_manifest(path: str | Path, split: str | None = None) -> list[dict[str, str]]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Manifest not found: {source}")
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = set(FIELDS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Manifest {source} is missing columns: {sorted(missing)}")
        rows = [dict(row) for row in reader if split is None or row["split"] == split]
    if not rows:
        raise ValueError(f"Manifest {source} has no rows" + (f" for split={split}" if split else ""))
    return rows


def write_manifest(rows: Iterable[dict[str, object]], path: str | Path) -> Path:
    output = ensure_parent(path)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in FIELDS})
    return output


def stratified_subject_split(
    rows: list[dict[str, object]], test_fraction: float = 0.5, seed: int = 0
) -> list[dict[str, object]]:
    """Match brain_fm: one shared RNG, stable group order, sorted subjects."""
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between zero and one")
    subject_group: dict[str, str] = {}
    group_order: list[str] = []
    for row in rows:
        subject, group = str(row["subject_id"]), str(row["group"])
        if subject in subject_group and subject_group[subject] != group:
            raise ValueError(f"Subject {subject} has conflicting labels")
        subject_group[subject] = group
        if group not in group_order:
            group_order.append(group)
    rng = random.Random(seed)
    test_subjects: set[str] = set()
    for group in group_order:
        subjects = sorted(s for s, label in subject_group.items() if label == group)
        rng.shuffle(subjects)
        test_subjects.update(subjects[:round(len(subjects) * test_fraction)])
    result = []
    for row in rows:
        copy = dict(row)
        copy["split"] = "test" if str(copy["subject_id"]) in test_subjects else "train"
        result.append(copy)
    return result


def carve_validation(
    rows: list[dict[str, str]], validation_fraction: float = 0.15, seed: int = 0
) -> list[dict[str, str]]:
    """Carve validation only from manifest train using a newly seeded shared RNG."""
    train = [row for row in rows if row["split"] == "train"]
    subject_group: dict[str, str] = {}
    group_order: list[str] = []
    for row in train:
        subject_group[row["subject_id"]] = row["group"]
        if row["group"] not in group_order:
            group_order.append(row["group"])
    rng = random.Random(seed)
    validation_subjects: set[str] = set()
    for group in group_order:
        subjects = sorted(s for s, label in subject_group.items() if label == group)
        rng.shuffle(subjects)
        count = max(1, round(len(subjects) * validation_fraction))
        validation_subjects.update(subjects[:count])
    result = []
    for row in rows:
        copy = dict(row)
        if copy["split"] == "train" and copy["subject_id"] in validation_subjects:
            copy["split"] = "validation"
        result.append(copy)
    return result


def build_adni_from_source(source: str | Path, output: str | Path) -> list[dict[str, object]]:
    rows = read_manifest(source)
    kept = [dict(row) for row in rows if row["group"] in LABELS["ad"]]
    subject_groups = {(row["subject_id"], row["group"]) for row in kept}
    subject_counts = Counter(group for _, group in subject_groups)
    expected = {"volumes": 548, "subjects": 416, "AD_subjects": 188, "CN_subjects": 228}
    observed = {"volumes": len(kept), "subjects": len(subject_groups),
                "AD_subjects": subject_counts["AD"], "CN_subjects": subject_counts["CN"]}
    if observed != expected:
        raise ValueError(f"ADNI source does not match the frozen cohort: expected {expected}, got {observed}")
    for row in kept:
        image = Path(row["path"])
        if not image.is_file() or image.stat().st_size == 0:
            raise FileNotFoundError(f"Missing or empty ADNI volume: {image}")
    split = carve_validation(stratified_subject_split(kept, 0.5, 0), 0.15, 0)
    validate_manifest(split, "ad")
    write_manifest(split, output)
    return split


def build_ucla(
    bids_root: str | Path, participants_tsv: str | Path, output: str | Path
) -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    root = Path(bids_root).expanduser().resolve()
    tsv = Path(participants_tsv).expanduser().resolve()
    if not tsv.is_file():
        raise FileNotFoundError(f"participants.tsv not found: {tsv}")
    diagnosis_map = {"SCHZ": "SCZ", "CONTROL": "CN"}
    rows: list[dict[str, object]] = []
    skipped: list[dict[str, str]] = []
    with tsv.open(newline="", encoding="utf-8") as handle:
        for participant in csv.DictReader(handle, delimiter="\t"):
            diagnosis = participant.get("diagnosis", "")
            if diagnosis not in diagnosis_map:
                continue
            subject = participant.get("participant_id", "")
            image = root / subject / "anat" / f"{subject}_T1w.nii.gz"
            if not image.is_file() or image.stat().st_size == 0:
                skipped.append({"subject_id": subject, "diagnosis": diagnosis, "path": str(image)})
                continue
            rows.append({"path": str(image), "file_id": subject.removeprefix("sub-"),
                         "subject_id": subject, "group": diagnosis_map[diagnosis],
                         "is_repeat": 0, "split": ""})
    if not rows:
        raise ValueError("No usable SCHZ/CONTROL T1w files were found")
    rows = carve_validation(stratified_subject_split(rows, 0.5, 0), 0.15, 0)
    validate_manifest(rows, "scz")
    write_manifest(rows, output)
    return rows, skipped


def validate_manifest(rows: list[dict[str, object]], disease: str) -> dict[str, object]:
    if disease not in LABELS:
        raise ValueError(f"Unknown disease: {disease}")
    allowed = set(LABELS[disease])
    seen: dict[str, str] = {}
    split_subjects: dict[str, set[str]] = {s: set() for s in ("train", "validation", "test")}
    for row in rows:
        group, split, subject = str(row["group"]), str(row["split"]), str(row["subject_id"])
        if group not in allowed:
            raise ValueError(f"Unexpected label {group!r} for {disease}")
        if split not in split_subjects:
            raise ValueError(f"Unexpected split {split!r}")
        if subject in seen and seen[subject] != split:
            raise ValueError(f"Subject leakage: {subject} appears in {seen[subject]} and {split}")
        seen[subject] = split
        split_subjects[split].add(subject)
    counts = Counter((str(row["split"]), str(row["group"])) for row in rows)
    return {"volumes": len(rows), "subjects": len(seen),
            "split_label_counts": {f"{s}:{g}": counts[(s, g)]
                                   for s in split_subjects for g in LABELS[disease]}}
