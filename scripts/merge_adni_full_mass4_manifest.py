#!/usr/bin/env python
"""Concatenate the new ADNI_add_full portion (preprocessed, split already
assigned by build_manifest_adni_full_mass.py at --test-fraction 0.15
--validation-fraction 0.1765) with dataset2's reshuffled manifest (same
fractions, recomputed by resplitting adni_full_mass_ss.csv from scratch --
see report/scaling_exp/scaling_exp_report.md's A4/B4 note for why: dataset2's
old 431 subjects are all 1.5T/ADNI1 and the new subjects are all 3T/ADNI2-4,
so splitting each field-strength pool separately at identical fractions and
concatenating is equivalent to a joint (field_strength x diagnosis) stratified
split of the combined 819-subject pool, without needing to extend
manifest.stratified_subject_split with a second stratification key.

Zero subject_id overlap between the two inputs was verified directly before
this experiment (dataset1's 416 AD/CN subjects are a strict subset of
dataset2's 431; dataset2 and ADNI_add_full share none), so validate_manifest's
leakage check here is a safety net, not an expected failure mode.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from encoderbench.manifest import validate_manifest, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-portion", required=True,
                        help="Preprocessed+split manifest for the new ADNI_add_full subjects")
    parser.add_argument("--old-portion", required=True,
                        help="Reshuffled dataset2 manifest (adni_full_mass_d2_reshuffled_ss.csv)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    for path in (args.old_portion, args.new_portion):
        with open(path) as f:
            rows.extend(csv.DictReader(f))

    summary = validate_manifest(rows, "ad")
    write_manifest(rows, args.out)
    print(f"wrote {args.out}")
    print("summary:", summary)


if __name__ == "__main__":
    main()
