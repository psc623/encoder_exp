"""Build data/manifests/bsnip2_synthseg.csv from the raw BSNIP2 manifest.

Companion to bsnip2_finalize_manifests.py, which does the same relabel
(HC->CN) + stratified-split (encoderbench.manifest.stratified_subject_split +
carve_validation, test_fraction=0.5/validation_fraction=0.15/seed=0) for the
three encoders that needed their own preprocessed copy of each scan. SynthSeg
needs no such copy -- mri_synthseg was run directly on the raw whole-head T1
files in data/manifests/bsnip2_raw.csv (see
freesurfer_install/submit_synthseg_bsnip2.sh) -- so this manifest's `path`
column points straight at those same raw files, matching how
SynthSegExtractor._posterior_path derives each posterior's filename from the
input path's stem.

The split only depends on the (subject_id, group) set, not on row order or
path, so this produces the same per-subject train/validation/test assignment
as the other three bsnip2_<encoder>.csv manifests -- the same held-out test
subjects for every encoder.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from encoderbench.manifest import (  # noqa: E402
    carve_validation, stratified_subject_split, validate_manifest, write_manifest,
)

SOURCE = "data/manifests/bsnip2_raw.csv"
OUTPUT = "data/manifests/bsnip2_synthseg.csv"


def load_source(path: str) -> list[dict[str, str]]:
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
    return carve_validation(stratified_subject_split(relabeled, 0.5, 0), 0.15, 0)


def main() -> None:
    rows = load_source(SOURCE)
    final_rows = finalize(rows)
    for row in final_rows:
        if not row["path"] or row["group"] not in ("CN", "SZ"):
            raise ValueError(f"Bad row: {row}")
    audit = validate_manifest(final_rows, "bsnip2")
    write_manifest(final_rows, OUTPUT)
    print(f"wrote {OUTPUT}")
    print(f"  {audit}")

    # Confirm the split agrees with the other three encoders' manifests
    # subject-for-subject, since a fair comparison needs the same held-out
    # test subjects for every encoder.
    reference_path = Path("data/manifests/bsnip2_mass.csv")
    if reference_path.is_file():
        reference = {row["subject_id"]: row["split"] for row in load_source(str(reference_path))}
        mapping = {row["subject_id"]: row["split"] for row in final_rows}
        mismatches = {s: (reference.get(s), mapping.get(s)) for s in set(reference) | set(mapping)
                     if reference.get(s) != mapping.get(s)}
        if mismatches:
            raise ValueError(f"synthseg split disagrees with mass for subjects: {mismatches}")
        print("OK: synthseg split matches bsnip2_mass.csv subject-for-subject")


if __name__ == "__main__":
    main()
