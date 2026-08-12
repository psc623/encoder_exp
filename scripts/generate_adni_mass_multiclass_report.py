#!/usr/bin/env python
"""Append a 'Group 2 -- 3-class, all visits' section to the existing
artifacts/reports/ADNI_MASS_scaling_report.md (written by
generate_adni_mass_scaling_report.py for the binary Group 1 run), instead of
overwriting it -- matching this repo's RESULTS.md convention of accumulating
"Group N" sections chronologically rather than replacing prior results.
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path("/net/projects2/litian-lab/scpan/encoders")
FINETUNE_DIR = ROOT / "artifacts/finetune/ad3/mass_full_scaling_multiclass"
MANIFEST = ROOT / "data/manifests/adni_full_mass3.csv"
DEDUP_LOG = ROOT / "data/manifests/adni_full_mass3_raw.dedup_log.csv"
PREPROCESS_AUDIT = ROOT / "artifacts/audits/ad/adni_full_mass3_preprocess_audit.json"
OUT = ROOT / "artifacts/reports/ADNI_MASS_scaling_report.md"
SEEDS = (0, 1, 2)
CLASSES = ("CN", "MCI", "AD")


def manifest_counts() -> dict:
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    counts: dict[str, dict[str, int]] = {}
    subjects: dict[str, set[str]] = {}
    for row in rows:
        split = row["split"]
        counts.setdefault(split, {}).setdefault(row["group"], 0)
        counts[split][row["group"]] += 1
        subjects.setdefault(split, set()).add(row["subject_id"])
    return {"volumes": len(rows), "by_split_group": counts,
           "subjects_by_split": {split: len(ids) for split, ids in subjects.items()}}


def fmt(values: list[float]) -> str:
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{mean:.3f} ± {sd:.3f}"


def main() -> None:
    summaries = []
    for seed in SEEDS:
        path = FINETUNE_DIR / f"finetune_seed_{seed}_summary.json"
        if not path.is_file():
            print(f"missing {path}, not all seeds are done yet", file=sys.stderr)
            sys.exit(1)
        summaries.append(json.loads(path.read_text()))

    counts = manifest_counts()
    budget = summaries[0]["parameter_budget"]
    grad_audits_pass = all(s["gradient_audit"]["passed"] for s in summaries)
    warm_starts = [s["head_warm_started_from"] for s in summaries]

    vba = [s["metrics"]["volume_level"]["balanced_accuracy"] for s in summaries]
    vauc = [s["metrics"]["volume_level"]["roc_auc_ovr_macro"] for s in summaries
           if s["metrics"]["volume_level"]["roc_auc_ovr_macro"] is not None]
    sba = [s["metrics"]["subject_level"]["balanced_accuracy"] for s in summaries]
    sauc = [s["metrics"]["subject_level"]["roc_auc_ovr_macro"] for s in summaries
           if s["metrics"]["subject_level"]["roc_auc_ovr_macro"] is not None]

    per_class_lines = []
    for cls in CLASSES:
        recalls = [s["metrics"]["volume_level"]["per_class"][cls]["recall"] for s in summaries]
        f1s = [s["metrics"]["volume_level"]["per_class"][cls]["f1"] for s in summaries]
        per_class_lines.append(f"| {cls} | {fmt(recalls)} | {fmt(f1s)} |")
    per_class_table = "\n".join(per_class_lines)

    per_seed_rows = "\n".join(
        f"| {s['seed']} | {s['selection']['epoch']} | "
        f"{s['metrics']['volume_level']['balanced_accuracy']:.4f} | "
        f"{s['metrics']['volume_level']['roc_auc_ovr_macro']:.4f} | "
        f"{s['metrics']['subject_level']['balanced_accuracy']:.4f} | "
        f"{s['metrics']['subject_level']['roc_auc_ovr_macro']:.4f} |"
        for s in summaries
    )

    preprocess_note = ""
    if PREPROCESS_AUDIT.is_file():
        audit = json.loads(PREPROCESS_AUDIT.read_text())
        preprocess_note = (
            f"{audit['succeeded']}/{audit['total_rows']} volumes preprocessed successfully "
            f"({len(audit['failed'])} failed) on the last preprocessing hop; already-preprocessed "
            f"screening volumes from the Group 1 run were reused unchanged (idempotent output paths)."
        )

    dedup_note = ""
    if DEDUP_LOG.is_file():
        with DEDUP_LOG.open(newline="", encoding="utf-8") as handle:
            dedup_rows = list(csv.DictReader(handle))
        multi = sum(1 for row in dedup_rows if row["dropped_image_uids"])
        dedup_note = (f"{len(dedup_rows)} subject+visit entries had a usable CN/MCI/AD image; "
                      f"{multi} had more than one candidate image *for that same visit* (repeat "
                      f"scan / reprocessed variant) and were deduped to one. Different visits of "
                      f"the same subject were never deduped against each other. Full log: "
                      f"`data/manifests/adni_full_mass3_raw.dedup_log.csv`.")

    section = f"""

---

## Group 2 -- 3-class (CN/MCI/AD), every visit, from-scratch head (append-only, Group 1 above unchanged)

Same encoder/tap/native-tokens/unbounded-budget/from-scratch-head protocol as
Group 1, but two changes to the data: (1) MCI is included as a third class
instead of being dropped, and (2) every visit (screening, m6, m12, ...) is
used as its own training example instead of collapsing each subject to one
baseline volume -- subject-level train/val/test splitting still applies, so a
given subject's visits never cross splits. This is a genuinely different task
from Group 1's AD-vs-CN binary classification (3-way balanced accuracy and
macro one-vs-rest AUC are not numerically comparable to Group 1's binary BA/AUC).

### Dataset

{dedup_note}

- Manifest: `data/manifests/adni_full_mass3.csv`
- Total volumes: {counts['volumes']}
- Subjects per split: {counts['subjects_by_split']}
- Volumes per split/group: {counts['by_split_group']}

### Preprocessing

Identical MASS-native recipe as Group 1 (`scripts/preprocess_mass_native.py`,
unchanged) applied to the expanded manifest. {preprocess_note}

### Protocol

Gradient audit passed for all seeds: {grad_audits_pass}. Trainable encoder
parameters: {budget['trainable_encoder_parameters']:,} (same 4 upstream MASS
stages as Group 1). Head warm-started from (should be `null` for all seeds):
{warm_starts}. 3-class attention head (`AttentionPoolHead(..., num_classes=3)`),
trained from scratch.

### Results (3 seeds, mean ± sample SD)

| metric | value |
|---|---:|
| volume balanced accuracy (macro recall) | {fmt(vba)} |
| volume AUC (macro one-vs-rest) | {fmt(vauc) if vauc else 'n/a'} |
| subject balanced accuracy | {fmt(sba)} |
| subject AUC (macro one-vs-rest) | {fmt(sauc) if sauc else 'n/a'} |

#### Per-class recall / F1 (volume level, mean ± sample SD)

| class | recall | F1 |
|---|---:|---:|
{per_class_table}

#### Per-seed detail

| seed | selected epoch | volume BA | volume AUC (macro OVR) | subject BA | subject AUC (macro OVR) |
|---:|---:|---:|---:|---:|---:|
{per_seed_rows}

Artifacts: `artifacts/finetune/ad3/mass_full_scaling_multiclass/finetune_seed_*_summary.json`,
`artifacts/cache/ad3/mass_full_native.npz`, `artifacts/audits/ad/adni_full_mass3_preprocess_audit.json`.
"""
    with OUT.open("a") as handle:
        handle.write(section)
    print(f"appended Group 2 section to {OUT}")


if __name__ == "__main__":
    main()
