"""Add a plain (unbalanced) accuracy column to the four main comparison tables.

Balanced accuracy already reported everywhere is (sensitivity+specificity)/2,
which is insensitive to class-count skew by construction. Plain accuracy
((TP+TN)/N) is not, so the gap between the two columns is a direct readout of
how much each run's number benefits from -- or is dragged down by -- an
unbalanced test split. Every probe/finetune summary JSON already stores
tp/tn/fp/fn at both volume_level and subject_level (encoderbench.metrics.
binary_metrics), so this recomputes accuracy from those confusion counts
without re-running anything.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path("/net/projects2/litian-lab/scpan/encoders/artifacts")
SEEDS = (0, 1, 2)


def _accuracy(level: dict) -> float:
    return (level["tp"] + level["tn"]) / level["n"]


def _read(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.is_file() else None


def _spread(values: list[float]) -> str:
    if not values:
        return "n/a"
    if len(values) == 1:
        return f"{values[0]:.3f}"
    return f"{statistics.mean(values):.3f} ± {statistics.stdev(values):.3f}"


def _median(values: list[float]) -> str:
    return f"{statistics.median(values):.3f}" if values else "n/a"


def ad_rows(kind: str, encoders: list[str]) -> list[tuple]:
    subdir = "attention" if kind == "probe" else "finetune"
    prefix = "probe" if kind == "probe" else "finetune"
    rows = []
    for encoder in encoders:
        volume_acc, subject_acc = [], []
        for seed in SEEDS:
            data = _read(ROOT / subdir / "ad" / encoder / f"{prefix}_seed_{seed}_summary.json")
            if data is None:
                continue
            volume_acc.append(_accuracy(data["metrics"]["volume_level"]))
            subject_acc.append(_accuracy(data["metrics"]["subject_level"]))
        rows.append((encoder, _spread(volume_acc), _spread(subject_acc)))
    return rows


def bsnip2_rows(kind: str, encoders: list[str]) -> list[tuple]:
    subdir = "attention" if kind == "probe" else "finetune"
    prefix = "probe" if kind == "probe" else "finetune"
    rows = []
    for encoder in encoders:
        volume_acc = []
        for seed in SEEDS:
            data = _read(ROOT / subdir / "bsnip2" / encoder / f"{prefix}_seed_{seed}_summary.json")
            if data is None:
                continue
            volume_acc.append(_accuracy(data["metrics"]["volume_level"]))
        rows.append((encoder, _spread(volume_acc), _median(volume_acc)))
    return rows


def main() -> None:
    print("## ADNI freeze -- volume accuracy | subject accuracy\n")
    for encoder, vol, subj in ad_rows("probe", ["mass", "medsiglip", "anatcl", "brainiac", "synthseg"]):
        print(f"| {encoder} | {vol} | {subj} |")

    print("\n## ADNI finetune -- volume accuracy | subject accuracy\n")
    for encoder, vol, subj in ad_rows("finetune", ["mass", "medsiglip", "brainiac", "anatcl", "synthseg"]):
        print(f"| {encoder} | {vol} | {subj} |")

    print("\n## bsnip2 freeze -- accuracy mean ± SD | median\n")
    for encoder, spread, median in bsnip2_rows("probe", ["brainiac", "mass", "medsiglip", "synthseg"]):
        print(f"| bsnip2:{encoder} | {spread} | {median} |")

    print("\n## bsnip2 finetune -- accuracy mean ± SD | median\n")
    for encoder, spread, median in bsnip2_rows("finetune", ["brainiac", "mass", "medsiglip", "synthseg"]):
        print(f"| bsnip2:{encoder} | {spread} | {median} |")


if __name__ == "__main__":
    main()
