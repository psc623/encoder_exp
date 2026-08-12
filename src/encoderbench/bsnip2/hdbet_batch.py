"""Batch HD-BET skull-stripping, one process for the whole manifest.

Used two ways:
  - BrainIAC: after N4 + rigid registration to temp_head.nii.gz
    (bsnip2_register_brainiac.py), closing the exact gap ADNI's already
    skull-stripped input left open (RESULTS.md: "None of this runs in our
    pipeline -- we feed ADNI's already skull-stripped ... box-cropped files").
  - medsiglip: directly on the raw manifest. medsiglip has no native 3D
    registration recipe of its own (it is a generic 2D vision-language
    model), but its shared normalize_volume computes intensity percentiles
    over the whole array, so a non-skull-stripped whole head would pull the
    [1,99] window over scalp/skull/eyes -- something ADNI's already
    skull-stripped input never exposed it to either. Skull-stripping alone,
    no atlas registration, keeps it as close as possible to how it saw ADNI.

A single process handles the whole list (rather than one process per volume
via xargs, as the other two BSNIP2 preprocessing scripts do) because
run_hd_bet loads its network once and then iterates -- restarting the
interpreter per file would reload the model every time for no benefit, and
GPU work does not parallelize productively across processes on one device
anyway.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

BRAINIAC_PREPROCESSING = "/net/projects2/litian-lab/scpan/github_repo/BrainIAC/src/preprocessing"


def read_rows(manifest: str) -> list[dict]:
    with open(manifest, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True,
                       help="CSV with at least file_id,path columns (path is the input .nii/.nii.gz)")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--out-manifest", default=None)
    parser.add_argument("--device", default="0", help="GPU id, or 'cpu'")
    parser.add_argument("--mode", default="fast", choices=["fast", "accurate"])
    args = parser.parse_args()

    sys.path.insert(0, BRAINIAC_PREPROCESSING)
    from HD_BET.run import run_hd_bet

    rows = read_rows(args.manifest)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pending_inputs, pending_outputs, pending_rows = [], [], []
    done_rows = []
    for row in rows:
        out_path = out_dir / f"{row['file_id']}_ss.nii.gz"
        if out_path.is_file() and out_path.stat().st_size > 0:
            done_rows.append(row | {"path": str(out_path)})
            continue
        pending_inputs.append(row["path"])
        pending_outputs.append(str(out_path))
        pending_rows.append(row)

    print(f"{len(done_rows)} already done, {len(pending_inputs)} to process", flush=True)
    if pending_inputs:
        device = args.device if args.device == "cpu" else int(args.device)
        run_hd_bet(pending_inputs, pending_outputs, mode=args.mode, device=device,
                   postprocess=True, do_tta=False, keep_mask=False, overwrite=False)

    if args.out_manifest:
        final = done_rows[:]
        for row, out_path in zip(pending_rows, pending_outputs):
            if Path(out_path).is_file() and Path(out_path).stat().st_size > 0:
                final.append(row | {"path": out_path})
        fieldnames = list(rows[0].keys())
        with open(args.out_manifest, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in final:
                writer.writerow({key: row[key] for key in fieldnames})
        print(f"DONE: {len(final)}/{len(rows)} skull-stripped -> {args.out_manifest}")


if __name__ == "__main__":
    main()
