"""Aggregate the frozen-encoder + attention-pooling comparison into one table.

Every row is the same experiment: the encoder stays frozen and is read with its
own native preprocessing at its validation-selected depth, and the only thing
trained is a fresh attention-pooling classifier head (3 seeds). The two SynthSeg
rows are baselines, not encoders -- one feeds its 33-class posteriors through the
identical head, the other is classical regional volumetry with no learned
representation at all.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path("/net/projects2/litian-lab/scpan/encoders/artifacts")
ENCODERS = ["medsiglip", "mass", "brainiac", "anatcl"]
SEEDS = (0, 1, 2)


def cache_layer(encoder: str) -> str:
    """The depth the cache was actually built at, recorded in its own metadata."""
    import numpy as np

    path = ROOT / "cache" / "ad" / f"{encoder}.npz"
    if not path.is_file():
        return "?"
    with np.load(path, allow_pickle=False) as data:
        return str(json.loads(str(data["metadata"].item())).get("layer", "?"))


def read_seeds(encoder: str) -> list[dict]:
    runs = []
    for seed in SEEDS:
        path = ROOT / "attention" / "ad" / encoder / f"probe_seed_{seed}_summary.json"
        if path.is_file():
            runs.append(json.loads(path.read_text()))
    return runs


def spread(values: list[float]) -> str:
    if not values:
        return "n/a"
    if len(values) == 1:
        return f"{values[0]:.3f}"
    return f"{statistics.mean(values):.3f} ± {statistics.stdev(values):.3f}"


def main() -> None:
    config = json.loads((ROOT.parent / "config" / "default.yaml").read_text()) \
        if False else None  # layers are reported from each run's own cache metadata

    lines: list[str] = []
    lines.append("# Frozen encoder + attention-pooling: AD vs CN (ADNI1 Screening 1.5T)\n")
    lines.append("Each encoder is frozen and read with its **own native preprocessing** at a "
                 "depth chosen on the validation split; the only trained component is a fresh "
                 "attention-pooling head (3 seeds). Test split is 271 volumes / 120 AD, scored once.\n")

    lines.append("\n## Test-set results\n")
    lines.append("| model | layer | volume-level BA | volume-level AUC | subject-level BA | subject-level AUC |")
    lines.append("|---|---:|---:|---:|---:|---:|")

    rows = []
    for encoder in ENCODERS:
        runs = read_seeds(encoder)
        if not runs:
            lines.append(f"| {encoder} | — | *(missing)* | | | |")
            continue
        layer = cache_layer(encoder)
        volume_ba = [r["metrics"]["volume_level"]["balanced_accuracy"] for r in runs]
        volume_auc = [r["metrics"]["volume_level"]["roc_auc"] for r in runs]
        subject_ba = [r["metrics"]["subject_level"]["balanced_accuracy"] for r in runs]
        subject_auc = [r["metrics"]["subject_level"]["roc_auc"] for r in runs]
        rows.append((encoder, statistics.mean(volume_auc)))
        lines.append(f"| **{encoder}** | {layer} | {spread(volume_ba)} | {spread(volume_auc)} | "
                     f"{spread(subject_ba)} | {spread(subject_auc)} |")

    synthseg = read_seeds("synthseg")
    if synthseg:
        volume_ba = [r["metrics"]["volume_level"]["balanced_accuracy"] for r in synthseg]
        volume_auc = [r["metrics"]["volume_level"]["roc_auc"] for r in synthseg]
        subject_ba = [r["metrics"]["subject_level"]["balanced_accuracy"] for r in synthseg]
        subject_auc = [r["metrics"]["subject_level"]["roc_auc"] for r in synthseg]
        lines.append(f"| *SynthSeg posteriors (baseline)* | n/a | {spread(volume_ba)} | "
                     f"{spread(volume_auc)} | {spread(subject_ba)} | {spread(subject_auc)} |")

    volumetry_path = ROOT / "reports" / "volumetry_baseline_ad.json"
    if volumetry_path.is_file():
        v = json.loads(volumetry_path.read_text())
        lines.append(f"| *SynthSeg volumetry (baseline)* | n/a | "
                     f"{v['volume_level']['balanced_accuracy']:.3f} | "
                     f"{v['volume_level']['roc_auc']:.3f} | "
                     f"{v['subject_level']['balanced_accuracy']:.3f} | "
                     f"{v['subject_level']['roc_auc']:.3f} |")

    lines.append("\nBaselines are deterministic (a single logistic fit), so they carry no seed spread.\n")

    # What the protocol fixes changed, using the archived pre-fix runs.
    old_root = ROOT / "attention" / "ad_oldprotocol"
    if old_root.is_dir():
        lines.append("\n## Effect of the protocol fixes\n")
        lines.append("Earlier runs read every encoder at its deepest layer, fed all 3D encoders one "
                     "generic [0,1] rescaling regardless of what they were trained on, and computed "
                     "the head's normalisation from token-means (which divided AnatCL's dead channels "
                     "by ~0).\n")
        lines.append("| encoder | before (BA) | after (BA) |")
        lines.append("|---|---:|---:|")
        for encoder in ENCODERS:
            before = []
            for seed in SEEDS:
                path = old_root / encoder / f"probe_seed_{seed}_summary.json"
                if path.is_file():
                    before.append(json.loads(path.read_text())["metrics"]["volume_level"]["balanced_accuracy"])
            after = [r["metrics"]["volume_level"]["balanced_accuracy"] for r in read_seeds(encoder)]
            if before and after:
                lines.append(f"| {encoder} | {spread(before)} | {spread(after)} |")

    ablation = ROOT / "attention" / "ad_medsiglip_deepest"
    if ablation.is_dir():
        lines.append("\n## Caveat: the fixes did not help MedSigLIP\n")
        lines.append("MedSigLIP's preprocessing never changed (it uses its own SiglipImageProcessor), "
                     "so re-running the probe on the archived deepest-layer cache isolates the two "
                     "protocol changes:\n")
        lines.append("| variant | volume-level BA | volume-level AUC |")
        lines.append("|---|---:|---:|")
        for tag, folder in [("layer -1, pre-fix head", ROOT / "attention" / "ad_oldprotocol" / "medsiglip"),
                            ("layer -1, fixed head", ablation),
                            ("layer 9 (selected), fixed head", ROOT / "attention" / "ad" / "medsiglip")]:
            ba, auc = [], []
            for seed in SEEDS:
                path = folder / f"probe_seed_{seed}_summary.json"
                if path.is_file():
                    metrics = json.loads(path.read_text())["metrics"]["volume_level"]
                    ba.append(metrics["balanced_accuracy"])
                    auc.append(metrics["roc_auc"])
            if ba:
                lines.append(f"| {tag} | {spread(ba)} | {spread(auc)} |")
        lines.append("\nThe seed spread (±0.02–0.06) is as large as the gaps, so these three variants "
                     "are not separable with 3 seeds. The layer was still chosen on validation and is "
                     "reported as chosen: re-selecting it now that the test numbers are visible would "
                     "leak the test split. The honest reading is that validation-based layer selection "
                     "is unreliable at n=42 validation volumes -- it helped BrainIAC, AnatCL and MASS, "
                     "and did not help MedSigLIP.\n")

    report = "\n".join(lines) + "\n"
    output = ROOT / "reports" / "frozen_encoder_comparison.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report)
    print(report)
    print(f"saved {output}")


if __name__ == "__main__":
    main()
