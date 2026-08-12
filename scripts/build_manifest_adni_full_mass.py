#!/usr/bin/env python
"""Build an ADNI_full manifest from the raw XML metadata + NIfTI collection.

Source: /net/projects2/litian-lab/scpan/dataset/ADNI_full/ADNI1_Metadata/ADNI/*.xml
(one XML per acquired image, IDA schema) cross-referenced against the raw
NIfTI tree at .../ADNI_full/ADNI1/ADNI/<subject>/<protocol>/<date>/I<imageUID>/*.nii.

Two modes, controlled by --groups and --all-visits:

- Default (--groups CN,AD, screening only): baseline-visit AD/CN, one row per
  subject. This was the original binary AD-vs-CN protocol.
- Expanded (--groups CN,MCI,AD --all-visits): every visit (screening, m6,
  m12, ...) becomes its own row, MCI included. researchGroup was verified
  empirically to be a static per-subject field (0/843 subjects have more than
  one distinct value across their images), so every row for a given subject
  gets that subject's one fixed diagnosis regardless of which visit it's
  from -- there's no per-visit progression label in this metadata.

Either way, deduping happens within the grouping key (subject alone, or
subject+visit under --all-visits): a repeat scan / reprocessed "_2" variant of
the *same* visit is deduped to exactly one row (earliest dateAcquired first,
then the non-repeat processedDataLabel as a tiebreak) since it's the same
underlying acquisition, not new information. Different visits of the same
subject are never deduped against each other -- they're genuinely different
scan sessions. The full dedup decision is logged to <out>.dedup_log.csv.

This intentionally does not go through manifest.build_adni_from_source: that
function hard-asserts the frozen 548-volume/416-subject AD/CN cohort from
ADNI_processed_clean and would reject this larger, independent source. It
does reuse manifest.stratified_subject_split / carve_validation /
write_manifest unchanged (subject-level splitting already handles multiple
rows per subject correctly -- every row for a subject is assigned that
subject's split), so the split rule itself is identical to every other
manifest in this repo. validate_manifest is only reused for the 2-class CN/AD
case (it hard-checks against manifest.LABELS, which has no 3-class entry); the
3-class case uses an equivalent, unrestricted local check instead.
"""
from __future__ import annotations

import argparse
import csv
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from encoderbench.manifest import (carve_validation, stratified_subject_split,
                                    validate_manifest, write_manifest)

# Defaults are the original ADNI1 source; --metadata-root/--image-root below
# let this same script build a manifest from a different IDA export (e.g.
# ADNI_add_full's ADNI2-4 batch) without touching the ADNI1 callers.
METADATA_ROOT = Path("/net/projects2/litian-lab/scpan/dataset/ADNI_full/ADNI1_Metadata/ADNI")
IMAGE_ROOT = Path("/net/projects2/litian-lab/scpan/dataset/ADNI_full/ADNI1/ADNI")


def parse_xml(path: Path) -> dict | None:
    # <idaxs> declares xmlns="http://ida.loni.usc.edu", but its child <project>
    # re-declares xmlns="" (verified against a sample file), which cancels the
    # default namespace for <project> and everything under it (<subject>,
    # <visit>, <series>, <derivedProduct>, ...). Those elements are therefore
    # unnamespaced and must be looked up with plain tag names, not "i:tag".
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return None
    subject = root.find(".//subject")
    if subject is None:
        return None
    subject_id = subject.findtext("subjectIdentifier")
    group = subject.findtext("researchGroup")
    visit = subject.findtext(".//visit/visitIdentifier")
    date = subject.findtext(".//series/dateAcquired")
    image_uid = subject.findtext(".//derivedProduct/imageUID")
    label = subject.findtext(".//derivedProduct/processedDataLabel") or ""
    if not (subject_id and group and visit and image_uid):
        return None
    return {"subject_id": subject_id, "group": group, "visit": visit,
            "date": date or "9999-99-99", "image_uid": image_uid, "label": label,
            "source_xml": path.name}


def locate_image(subject_id: str, image_uid: str, image_root: Path) -> Path | None:
    matches = sorted((image_root / subject_id).glob(f"*/*/I{image_uid}/*.nii"))
    return matches[0] if matches else None


def is_repeat_variant(label: str) -> bool:
    first_token = label.split(";")[0].strip()
    return label.strip().endswith("_2") or first_token.endswith("-R")


def _validate_generic(rows: list[dict], groups: tuple[str, ...]) -> dict:
    """Same checks as manifest.validate_manifest, minus the disease/LABELS
    restriction, for group sets manifest.LABELS has no entry for (3-class)."""
    allowed = set(groups)
    seen: dict[str, str] = {}
    split_subjects: dict[str, set[str]] = {s: set() for s in ("train", "validation", "test")}
    for row in rows:
        group, split, subject = str(row["group"]), str(row["split"]), str(row["subject_id"])
        if group not in allowed:
            raise ValueError(f"Unexpected label {group!r}; expected one of {sorted(allowed)}")
        if split not in split_subjects:
            raise ValueError(f"Unexpected split {split!r}")
        if subject in seen and seen[subject] != split:
            raise ValueError(f"Subject leakage: {subject} appears in {seen[subject]} and {split}")
        seen[subject] = split
        split_subjects[split].add(subject)
    counts = Counter((str(row["split"]), str(row["group"])) for row in rows)
    return {"volumes": len(rows), "subjects": len(seen),
           "split_label_counts": {f"{s}:{g}": counts[(s, g)] for s in split_subjects for g in groups}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/manifests/adni_full_mass_raw.csv")
    parser.add_argument("--metadata-root", type=Path, default=METADATA_ROOT,
                        help="Directory of per-image IDA XML files (default: ADNI1 source)")
    parser.add_argument("--image-root", type=Path, default=IMAGE_ROOT,
                        help="Directory of <subject>/<protocol>/<date>/I<uid>/*.nii (default: ADNI1 source)")
    parser.add_argument("--groups", default="CN,AD",
                        help="Comma-separated researchGroup values to keep, e.g. CN,MCI,AD")
    parser.add_argument("--all-visits", action="store_true",
                        help="Use every visit per subject (one row per subject+visit) instead "
                             "of collapsing to a single baseline/screening row per subject")
    parser.add_argument("--test-fraction", type=float, default=0.5)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process the first N metadata files (debugging)")
    args = parser.parse_args()
    groups = tuple(group.strip() for group in args.groups.split(","))

    xml_files = sorted(args.metadata_root.glob("*.xml"))
    if args.limit:
        xml_files = xml_files[: args.limit]
    print(f"scanning {len(xml_files)} metadata files under {args.metadata_root}", flush=True)

    by_key: dict[tuple, list[dict]] = defaultdict(list)
    skipped_unparsed = 0
    for xml_path in xml_files:
        record = parse_xml(xml_path)
        if record is None:
            skipped_unparsed += 1
            continue
        if not args.all_visits and "screen" not in record["visit"].lower():
            continue
        if record["group"] not in groups:
            continue
        key = (record["subject_id"], record["visit"]) if args.all_visits else record["subject_id"]
        by_key[key].append(record)

    scope = "all-visit" if args.all_visits else "screening-visit"
    print(f"{len(by_key)} {'subject+visit' if args.all_visits else 'subject'} entries have a "
         f"{scope} {'/'.join(groups)} image "
         f"({skipped_unparsed} XML files failed to parse or lacked required fields)", flush=True)

    rows, dedup_log, missing_files = [], [], []
    for key in sorted(by_key):
        records = sorted(by_key[key], key=lambda r: (r["date"], is_repeat_variant(r["label"])))
        chosen, dropped = records[0], records[1:]
        subject_id = chosen["subject_id"]
        image_path = locate_image(subject_id, chosen["image_uid"], args.image_root)
        if image_path is None:
            missing_files.append((subject_id, chosen["image_uid"]))
            continue
        rows.append({"path": str(image_path), "file_id": f"I{chosen['image_uid']}",
                     "subject_id": subject_id, "group": chosen["group"],
                     "is_repeat": 0, "split": ""})
        dedup_log.append({"subject_id": subject_id, "visit": chosen["visit"],
                          "chosen_image_uid": chosen["image_uid"],
                          "chosen_label": chosen["label"], "chosen_date": chosen["date"],
                          "dropped_image_uids": ";".join(r["image_uid"] for r in dropped)})

    if missing_files:
        print(f"WARNING: {len(missing_files)} entries had metadata but no matching .nii "
             f"file on disk (first 10): {missing_files[:10]}", file=sys.stderr)

    split = carve_validation(
        stratified_subject_split(rows, args.test_fraction, args.split_seed),
        args.validation_fraction, args.split_seed,
    )
    if set(groups) == {"CN", "AD"}:
        summary = validate_manifest(split, "ad")
    else:
        summary = _validate_generic(split, groups)
    write_manifest(split, args.out)
    print(f"wrote {args.out}")
    print("summary:", summary)

    dedup_log_path = Path(args.out).with_suffix("").with_suffix(".dedup_log.csv")
    with dedup_log_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["subject_id", "visit", "chosen_image_uid", "chosen_label", "chosen_date",
                     "dropped_image_uids"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dedup_log)
    print(f"wrote {dedup_log_path}")


if __name__ == "__main__":
    main()
