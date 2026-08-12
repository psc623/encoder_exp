#!/usr/bin/env python
"""Build a manifest pointing MASS's bsnip2 subjects at their SPM12-registered
(MNI, deformable-warped) 'wmr' whole-head volumes instead of the raw,
unregistered files bsnip2_mass.csv points at -- to test whether the lack of
inter-subject anatomical registration is really what's capping MASS's bsnip2
numbers (see improved_report.md's analysis section).

wmr files: /net/projects2/litian-lab/scpan/dataset/bsnp2_final/summary/segmented/<site>/wmrS<ID>_....nii
  -- SPM12 "Warped bias corrected image" (descrip field verified), 121x145x121,
     1.5mm isotropic (verified against MASS's own target spacing -- no
     resampling needed), whole-head (skull included, matching MASS's own
     un-registered pipeline's lack of skull-stripping).

Not every subject_id in bsnip2_mass.csv has a matching wmr file (469/509 do,
92%); rows without one are dropped, keeping every other subject's existing
group/split assignment from bsnip2_mass.csv unchanged.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

SEGMENTED_ROOT = Path("/net/projects2/litian-lab/scpan/dataset/bsnp2_final/summary/segmented")
SOURCE_MANIFEST = Path("/net/projects2/litian-lab/scpan/encoders/data/manifests/bsnip2_mass.csv")
OUT_MANIFEST = Path("/net/projects2/litian-lab/scpan/encoders/data/manifests/bsnip2_mass_wmr.csv")


def index_wmr_files() -> dict[str, Path]:
    by_subject: dict[str, list[Path]] = defaultdict(list)
    for path in SEGMENTED_ROOT.rglob("wmrS*.nii"):
        stem = path.name[len("wmr"):]
        subject_id = stem.split("_")[0]
        by_subject[subject_id].append(path)
    # A handful of subjects have >1 wmr file (repeat scan/session); take the
    # lexicographically first, deterministic and good enough for this test.
    return {subject_id: sorted(paths)[0] for subject_id, paths in by_subject.items()}


def main() -> None:
    wmr_by_subject = index_wmr_files()
    with SOURCE_MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    kept, dropped = [], []
    for row in rows:
        wmr_path = wmr_by_subject.get(row["subject_id"])
        if wmr_path is None:
            dropped.append(row["subject_id"])
            continue
        kept.append({**row, "path": str(wmr_path)})

    fieldnames = list(rows[0].keys())
    with OUT_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in kept:
            writer.writerow({key: row[key] for key in fieldnames})

    print(f"kept {len(kept)}/{len(rows)} subjects, dropped {len(dropped)} with no wmr match")
    print(f"wrote {OUT_MANIFEST}")
    split_counts = defaultdict(int)
    for row in kept:
        split_counts[(row["split"], row["group"])] += 1
    print("split/group counts:", dict(split_counts))


if __name__ == "__main__":
    main()
