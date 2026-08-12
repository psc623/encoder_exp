#!/usr/bin/env python
"""Assemble artifacts/reports/ADNI_MASS_scaling_report.md from this run's
finetune summary JSONs, the manifest/dedup log, and the preprocessing audit.
Run automatically as the last step of sbatch_finetune_adni_full_mass_scaling.sh
once all 3 seeds have a summary JSON, so the report reflects exactly what ran
-- not hand-typed numbers.
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path("/net/projects2/litian-lab/scpan/encoders")
FINETUNE_DIR = ROOT / "artifacts/finetune/ad/mass_full_scaling"
MANIFEST = ROOT / "data/manifests/adni_full_mass.csv"
DEDUP_LOG = ROOT / "data/manifests/adni_full_mass_raw.dedup_log.csv"
PREPROCESS_AUDIT = ROOT / "artifacts/audits/ad/adni_full_mass_preprocess_audit.json"
OUT = ROOT / "artifacts/reports/ADNI_MASS_scaling_report.md"
SEEDS = (0, 1, 2)


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
    vauc = [s["metrics"]["volume_level"]["roc_auc"] for s in summaries]
    sba = [s["metrics"]["subject_level"]["balanced_accuracy"] for s in summaries]
    sauc = [s["metrics"]["subject_level"]["roc_auc"] for s in summaries]

    per_seed_rows = "\n".join(
        f"| {s['seed']} | {s['selection']['epoch']} | "
        f"{s['metrics']['volume_level']['balanced_accuracy']:.4f} | "
        f"{s['metrics']['volume_level']['roc_auc']:.4f} | "
        f"{s['metrics']['subject_level']['balanced_accuracy']:.4f} | "
        f"{s['metrics']['subject_level']['roc_auc']:.4f} |"
        for s in summaries
    )

    preprocess_note = ""
    if PREPROCESS_AUDIT.is_file():
        audit = json.loads(PREPROCESS_AUDIT.read_text())
        preprocess_note = (
            f"{audit['succeeded']}/{audit['total_rows']} raw volumes preprocessed "
            f"successfully ({len(audit['failed'])} failed) in {audit['elapsed_seconds']:.0f}s "
            f"on the last preprocessing hop. Recipe: {json.dumps(audit['recipe'])}."
        )

    dedup_note = ""
    if DEDUP_LOG.is_file():
        with DEDUP_LOG.open(newline="", encoding="utf-8") as handle:
            dedup_rows = list(csv.DictReader(handle))
        multi = sum(1 for row in dedup_rows if row["dropped_image_uids"])
        dedup_note = (f"{len(dedup_rows)} subjects had a usable screening-visit CN/AD image; "
                      f"{multi} had more than one candidate image and were deduped to one "
                      f"(earliest acquisition date, non-repeat processing variant preferred). "
                      f"Full per-subject decision log: `data/manifests/adni_full_mass_raw.dedup_log.csv`.")

    report = f"""# ADNI MASS scaling report

Full-parameter MASS finetune on the complete ADNI_full AD/CN baseline-screening
cohort, with native (unpooled) tokens instead of the shared 4x4x4 grid and an
attention-pooling head trained from scratch. This is a separate track from
`RESULTS.md`'s Group 2-4 (which used the smaller, already skull-stripped
`ADNI_processed_clean` cohort, the shared 4x4x4 pooled grid, and an 8M
parameter-budgeted finetune); nothing under `RESULTS.md` or the Group 2-4
artifacts was modified by this run.

## Dataset

Source: `/net/projects2/litian-lab/scpan/dataset/ADNI_full/ADNI1/ADNI` (raw
NIfTI) cross-referenced against `ADNI_full/ADNI1_Metadata/ADNI/*.xml`
(diagnosis + visit metadata). Baseline/screening visit only, AD vs CN only
(MCI/other groups dropped), one image per subject. {dedup_note}

- Manifest: `data/manifests/adni_full_mass.csv`
- Total volumes: {counts['volumes']}
- Subjects per split: {counts['subjects_by_split']}
- Volumes per split/group: {counts['by_split_group']}

## Preprocessing

MASS's own native preprocessing recipe, replicated by importing
`MASS/inference.py`'s actual functions (not reimplemented): reorient to RAS,
resample to 1.5mm isotropic spacing (linear interpolation), body/foreground
crop (threshold method, margin [16,32,32] zyx), **no skull-stripping** (MASS's
own recipe doesn't skull-strip either). This closes the gap disclosed in
`RESULTS.md` Group 2/3 (previously no un-cropped original was available, so
volumes were resized from an inconsistent bounding box instead of a uniform
physical spacing). Intensity normalization (percentile clip + z-score) is
unchanged -- still applied on-the-fly at cache/finetune time, exactly as for
every other encoder/disease run. {preprocess_note}

- Output directory: `/net/projects2/litian-lab/scpan/dataset/ADNI_full/ADNI_full_preprocessed`
- Preprocessing script: `scripts/preprocess_mass_native.py`

## Architecture / protocol deltas from Group 3/4's budgeted MASS finetune

| | Group 3/4 (`RESULTS.md`) | This run |
|---|---|---|
| Pooling | 4x4x4 adaptive-average pooled grid (64 tokens) | Native tokens (no pooling) -- `--native-tokens` |
| Trainable encoder params | 8M budget (MASS: subset of upstream stages) | Unbounded budget -- every upstream stage (`inc`,`down1`,`down2`,`down3`) trainable, {budget['trainable_encoder_parameters']:,} params total |
| Attention head init | Warm-started from the probe checkpoint | Random init, trained from scratch (`--no-warm-start`) |
| Tap layer | 3 (down3 output) | 3 (unchanged) |
| Dataset | `ADNI_processed_clean` (548 volumes / 416 subjects, pre-skull-stripped) | `ADNI_full` (this run's {counts['volumes']} volumes, raw + MASS-native preprocessed) |

Gradient audit passed for all seeds: {grad_audits_pass}. Head warm-started
from (should be `null`/None for all seeds, confirming training from scratch):
{warm_starts}.

## Results (3 seeds, mean ± sample SD)

| metric | value |
|---|---:|
| volume balanced accuracy | {fmt(vba)} |
| volume AUC | {fmt(vauc)} |
| subject balanced accuracy | {fmt(sba)} |
| subject AUC | {fmt(sauc)} |

### Per-seed detail

| seed | selected epoch | volume BA | volume AUC | subject BA | subject AUC |
|---:|---:|---:|---:|---:|---:|
{per_seed_rows}

### Comparison to Group 3/4's budgeted MASS finetune

| run | volume BA | volume AUC |
|---|---:|---:|
| Group 3 (8M budget, 4x4x4 pooled, warm-started head, `ADNI_processed_clean`) | 0.774 ± 0.019 | 0.852 ± 0.011 |
| Group 4 (same, re-swept layer + overfitting-aware selection) | 0.757 ± 0.027 | 0.840 ± 0.021 |
| This run (unbounded budget, native tokens, from-scratch head, `ADNI_full`) | {fmt(vba)} | {fmt(vauc)} |

Artifacts: `artifacts/finetune/ad/mass_full_scaling/finetune_seed_*_summary.json`
(per-seed metrics, subject-clustered bootstrap CIs, `parameter_budget` and
`gradient_audit` proof that every upstream stage received gradients),
`artifacts/cache/ad/mass_full_native.npz` (frozen cache used only for train
normalization statistics), `artifacts/audits/ad/adni_full_mass_preprocess_audit.json`.
"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(report)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
