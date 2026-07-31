"""Build a raw-volume manifest for BSNIP2 SZ/HC subjects.

Joins the diagnosis-group column (`dxgroup`) in the site QC tracker
(spreadsheets/080619_BSNIP2_Clean_Imaging_Status_MS.xlsx, sheet
Clean_Imaging_Status) against the actual raw whole-head T1 NIfTI file each
subject's SPM/VBM preprocessing was run from.

BSNIP2 subject folders hold two layers with very different meaning:
  <site>/subjects/**/<ID>/<series#>/*.nii    -- raw series copy
  <site>/subjects/**/<ID>/PREPROCESSING/*    -- SPM12 output *and* a copy of
                                                 the exact raw file SPM ran on

The raw file living in PREPROCESSING/ is the one that matters here: it is
byte-identical to the series-folder copy but its name proves which series SPM
actually used when a subject has more than one repeat scan. Every SPM-derived
product in that folder adds a lowercase prefix (m0wrp1, m0wrp2, wmr, sm0wrp)
or a `_seg8.mat`/`_seg8.txt` suffix; the untouched raw file is the only entry
that still starts with the uppercase subject ID and ends in plain `.nii`.
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

BSNIP2_ROOT = Path("/net/projects2/litian-lab/scpan/dataset/bsnp2_final")
QC_XLSX = BSNIP2_ROOT / "spreadsheets" / "080619_BSNIP2_Clean_Imaging_Status_MS.xlsx"
SHEET = "Clean_Imaging_Status"
SITES = ["Boston", "Chicago", "Dallas", "Georgia", "Hartford"]
KEEP_GROUPS = {"HC", "SZ"}

SUBJECT_DIR_RE = re.compile(r"^S[0-9A-Z]+$")
RAW_FILE_RE = re.compile(r"^S[0-9A-Za-z_]+\.nii$")


def read_dx_groups() -> list[dict]:
    import openpyxl

    wb = openpyxl.load_workbook(QC_XLSX, read_only=True, data_only=True)
    ws = wb[SHEET]
    rows = list(ws.iter_rows(values_only=True))
    header = rows[0]
    idx = {name: i for i, name in enumerate(header) if name is not None}
    # openpyxl read_only mode trims trailing None cells per row, so a row's
    # tuple length reflects that row's own last populated column, not the
    # sheet's widest row. Guard only against the columns actually read here
    # (max index 11, dxgroup) -- checking against the full header's max index
    # (16, a rarely-filled trailing QC-notes column) silently dropped every
    # row whose optional trailing fields happened to be blank.
    needed = max(idx["nidb_id"], idx["site"], idx["included"], idx["dxgroup"])
    out = []
    for row in rows[1:]:
        if row is None or len(row) <= needed:
            continue
        if row[idx["included"]] != 1:
            continue
        group = row[idx["dxgroup"]]
        if group not in KEEP_GROUPS:
            continue
        nidb_id = row[idx["nidb_id"]]
        site = row[idx["site"]]
        if not nidb_id or not site:
            continue
        out.append({"nidb_id": str(nidb_id).strip(), "site": str(site).strip(), "group": group})
    return out


PREPROC_DIR_NAMES = {"preprocessing", "preprocessed"}


def preproc_subdirs(subject_dir: Path) -> list[Path]:
    """grabFiles.sh's own comment lists 4 spellings sites actually used:
    Preprocessed / PREPROCESSING / Preprocessing / PREPROCESSED. Match
    case-insensitively; some Chicago subjects have *both* an empty
    'PREPROCESSING' and a populated 'Preprocessed', so return every match
    (sorted so the more common all-caps spelling is tried first) and let the
    caller fall through to the next one if a candidate has no raw file."""
    return sorted(
        (child for child in subject_dir.iterdir()
         if child.is_dir() and child.name.lower() in PREPROC_DIR_NAMES),
        key=lambda p: p.name,
    )


def find_subject_dirs(site_dir: Path) -> dict[str, list[Path]]:
    """Map subject-folder basename -> path, searched recursively under subjects/."""
    result: dict[str, list[Path]] = {}
    subjects_root = site_dir / "subjects"
    if not subjects_root.is_dir():
        return result
    for path in subjects_root.rglob("*"):
        if path.is_dir() and SUBJECT_DIR_RE.match(path.name):
            if preproc_subdirs(path):
                result.setdefault(path.name, []).append(path)
    return result


def raw_file_in(preproc_dir: Path) -> Path | None:
    candidates = sorted(
        p for p in preproc_dir.glob("S*.nii")
        if RAW_FILE_RE.match(p.name) and not p.name.endswith(("_seg8.mat", "_seg8.txt"))
    )
    return candidates[0] if candidates else None


def build_manifest(limit: int | None = None) -> list[dict]:
    dx_rows = read_dx_groups()
    site_lookup = {site.lower(): site for site in SITES}

    dirs_by_site: dict[str, dict[str, list[Path]]] = {}
    for site in SITES:
        dirs_by_site[site] = find_subject_dirs(BSNIP2_ROOT / site)

    manifest = []
    unmatched = []
    multi = []
    for i, row in enumerate(dx_rows):
        if limit is not None and i >= limit:
            break
        site = site_lookup.get(row["site"].lower())
        if site is None:
            unmatched.append((row["nidb_id"], row["site"], "unknown site"))
            continue
        nidb_id = row["nidb_id"]
        matches = sorted(name for name in dirs_by_site[site] if name.startswith(nidb_id))
        if not matches:
            unmatched.append((nidb_id, site, "no subject folder with PREPROCESSING"))
            continue
        if len(matches) > 1:
            multi.append((nidb_id, site, matches))
        chosen_name = None
        raw_path = None
        for candidate_name in matches:
            for chosen_dir in sorted(dirs_by_site[site][candidate_name]):
                for candidate in preproc_subdirs(chosen_dir):
                    raw_path = raw_file_in(candidate)
                    if raw_path is not None:
                        break
                if raw_path is not None:
                    break
            if raw_path is not None:
                chosen_name = candidate_name
                break
        if raw_path is None:
            unmatched.append((nidb_id, site, f"no raw .nii in any preprocessing dir under any of {matches}"))
            continue
        manifest.append({
            "file_id": chosen_name,
            "subject_id": nidb_id,
            "site": site,
            "group": row["group"],
            "path": str(raw_path),
        })

    print(f"Matched: {len(manifest)} / {len(dx_rows) if limit is None else min(limit, len(dx_rows))}")
    print(f"Unmatched: {len(unmatched)}")
    for item in unmatched[:30]:
        print("  UNMATCHED", item)
    if len(unmatched) > 30:
        print(f"  ... and {len(unmatched) - 30} more")
    print(f"Subjects with >1 folder match (used the first): {len(multi)}")
    for item in multi[:15]:
        print("  MULTI", item)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/manifests/bsnip2_raw.csv")
    parser.add_argument("--limit", type=int, default=None, help="Debug: only process first N dx rows")
    args = parser.parse_args()

    manifest = build_manifest(limit=args.limit)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file_id", "subject_id", "site", "group", "path"])
        writer.writeheader()
        for row in manifest:
            writer.writerow(row)
    print(f"Wrote {len(manifest)} rows -> {out_path}")

    from collections import Counter
    print("By site x group:", Counter((r["site"], r["group"]) for r in manifest))


if __name__ == "__main__":
    main()
