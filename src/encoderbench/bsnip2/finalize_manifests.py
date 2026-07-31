"""Turn the three per-encoder BSNIP2 preprocessing manifests into
encoderbench-ready manifests: relabel HC->CN (encoderbench's `_labels()`
hard-codes the negative class as the literal string "CN"; SZ is left as-is
and cli.py/finetune.py's `_positive()` now maps disease="bsnip2" -> "SZ"),
add is_repeat=0 (one scan per subject, already deduplicated), and compute
the same stratified train/validation/test split ADNI and UCLA CNP use
(encoderbench.manifest.stratified_subject_split + carve_validation, same
test_fraction=0.5/validation_fraction=0.15/seed=0 as config/default.yaml) so
the BSNIP2 workflow reuses the identical split machinery instead of a new
one. The three encoders share the same 509 (subject_id, group) pairs, and
the split functions only depend on that set (not on row order or path), so
this produces bit-identical train/val/test subject assignments across all
three -- the same test subjects for every encoder, as the shared-manifest
ADNI/SCZ workflows also guarantee.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from encoderbench.manifest import (  # noqa: E402
    FIELDS, carve_validation, stratified_subject_split, validate_manifest, write_manifest,
)

ENCODERS_SOURCES = {
    "brainiac": "data/manifests/bsnip2_brainiac_final.csv",
    "mass": "data/manifests/bsnip2_mass_final.csv",
    "medsiglip": "data/manifests/bsnip2_medsiglip_final.csv",
}


def load_source(path: str) -> list[dict[str, str]]:
    import csv

    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def finalize(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    relabeled = []
    for row in rows:
        group = "CN" if row["group"] == "HC" else row["group"]
        relabeled.append({
            "path": row["path"], "file_id": row["file_id"], "subject_id": row["subject_id"],
            "group": group, "is_repeat": 0, "split": "",
        })
    split = carve_validation(stratified_subject_split(relabeled, 0.5, 0), 0.15, 0)
    return split


def main() -> None:
    for encoder, source in ENCODERS_SOURCES.items():
        rows = load_source(source)
        final_rows = finalize(rows)
        for row in final_rows:
            if not row["path"] or row["group"] not in ("CN", "SZ"):
                raise ValueError(f"Bad row for {encoder}: {row}")
        audit = validate_manifest(final_rows, "bsnip2")
        output = f"data/manifests/bsnip2_{encoder}.csv"
        write_manifest(final_rows, output)
        print(f"{encoder}: wrote {output}")
        print(f"  {audit}")

    # Confirm all three encoders agree on the split assignment per subject,
    # since a fair comparison needs the same held-out test subjects for
    # every encoder even though they were preprocessed into different files.
    from collections import defaultdict

    split_by_encoder: dict[str, dict[str, str]] = {}
    for encoder in ENCODERS_SOURCES:
        rows = load_source(f"data/manifests/bsnip2_{encoder}.csv")
        split_by_encoder[encoder] = {row["subject_id"]: row["split"] for row in rows}
    reference = split_by_encoder["mass"]
    for encoder, mapping in split_by_encoder.items():
        if mapping != reference:
            mismatches = {s: (reference.get(s), mapping.get(s)) for s in set(reference) | set(mapping)
                         if reference.get(s) != mapping.get(s)}
            raise ValueError(f"{encoder} split disagrees with mass for subjects: {mismatches}")
    print("OK: all three encoders share identical per-subject train/validation/test splits")


if __name__ == "__main__":
    main()
