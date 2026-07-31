"""Build data/manifests/bsnip2_anatcl.csv, reusing BrainIAC's bsnp2 preprocessing.

AnatCL has no preprocessing script of its own on BSNIP2 (never run before).
It doesn't need one: AnatCL's ADNI extractor already treats its input as a
"resize-only adapter" over any skull-stripped whole-brain T1 (see
extractors.py's AnatCLExtractor docstring/load_audit note -- AnatCL's true
native input is a CAT12/VBM grey-matter density map, which nothing in this
repo produces, so ADNI already approximates it with skull-stripped T1). BSNIP2
already has exactly that available for free: bsnip2_brainiac_final.csv points
at images that are N4-corrected, rigidly registered to a standard template,
AND skull-stripped (bsnip2_register_brainiac.py + bsnip2_hdbet_batch.py) --
per this report's own earlier note, that registration-to-standard-space is
incidentally closer to what AnatCL was actually trained on than ADNI's
non-registered skull-stripped input, so this is a strict improvement over the
ADNI setup, not a new domain-shift risk.

Same relabel (HC->CN) + stratified split (test_fraction=0.5/validation_
fraction=0.15/seed=0) as the other three bsnip2 manifests, so the split is
subject-for-subject identical -- verified against bsnip2_mass.csv below.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from encoderbench.manifest import (  # noqa: E402
    carve_validation, stratified_subject_split, validate_manifest, write_manifest,
)

SOURCE = "data/manifests/bsnip2_brainiac_final.csv"
OUTPUT = "data/manifests/bsnip2_anatcl.csv"


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

    reference_path = Path("data/manifests/bsnip2_mass.csv")
    if reference_path.is_file():
        reference = {row["subject_id"]: row["split"] for row in load_source(str(reference_path))}
        mapping = {row["subject_id"]: row["split"] for row in final_rows}
        mismatches = {s: (reference.get(s), mapping.get(s)) for s in set(reference) | set(mapping)
                     if reference.get(s) != mapping.get(s)}
        if mismatches:
            raise ValueError(f"anatcl split disagrees with mass for subjects: {mismatches}")
        print("OK: anatcl split matches bsnip2_mass.csv subject-for-subject")


if __name__ == "__main__":
    main()
