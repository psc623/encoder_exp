#!/usr/bin/env python
"""Materialize MASS's own native preprocessing recipe over raw ADNI_full volumes.

Replicates MASS/inference.py's preprocess_target_image geometry steps verbatim
by importing its actual functions (reorient_image, resample_to_spacing,
maybe_crop_body) rather than reimplementing them -- using MASS/inference.py's
own documented CLI defaults: orientation RAS, target spacing 1.5mm isotropic,
linear interpolation, modality "auto", body crop method "threshold", body
margin [16,32,32] zyx. No skull-stripping: MASS's own recipe doesn't do one
either (confirmed by reading maybe_crop_body -- it's a foreground/body crop,
not a brain extraction).

Intensity normalization (percentile clip + z-score) is deliberately NOT baked
in here -- it already happens on-the-fly in encoderbench.preprocessing at
cache/finetune time (MASSExtractor.intensity = "clip_zscore", which already
reproduces MASS/inference.py's normalize_image exactly). This script only
fixes the previously-disclosed geometry gap: every subject now has uniform
1.5mm spacing before extractors.py's final resize-to-[128,128,128], instead of
resizing an arbitrary bounding box straight to a fixed grid.

Idempotent: skips any output file that already exists, so a killed/resubmitted
SLURM job just continues where it left off.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from encoderbench.manifest import FIELDS, read_manifest, write_manifest

MASS_ROOT = Path("/net/projects2/litian-lab/scpan/github_repo/MASS")
ORIENTATION = "RAS"
TARGET_SPACING_XYZ = (1.5, 1.5, 1.5)
BODY_METHOD = "threshold"
# Calibrated for MASS's own whole-head intensity-threshold crop, where the
# mask is already loose/generous -- proportionally small margin there.
BODY_MARGIN_ZYX = (16, 32, 32)
# A tight, already-precise skull-stripped mask needs a much smaller margin;
# the body margin above (24/48/48mm at 1.5mm spacing) would nearly double a
# skull-stripped brain's extent instead of tightening it (measured: it left
# nonzero_frac ~0.17, i.e. 83% zero-padding). A few voxels is a normal safety
# buffer for a segmentation-quality mask.
SKULLSTRIP_MARGIN_ZYX = (4, 4, 4)
CT_BODY_THRESHOLD = -500.0
MODALITY = "auto"


def _load_mass_inference():
    # Same pattern as encoderbench.extractors._load_module: register in
    # sys.modules before exec so inference.py's own top-level `import models`/
    # `from utils.registry import get_model` (relative to the MASS repo root)
    # resolve correctly.
    if str(MASS_ROOT) not in sys.path:
        sys.path.insert(0, str(MASS_ROOT))
    spec = importlib.util.spec_from_file_location("encoderbench_mass_inference_raw",
                                                   MASS_ROOT / "inference.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["encoderbench_mass_inference_raw"] = module
    spec.loader.exec_module(module)
    return module


def _nonzero_crop(mass, image, margin_zyx):
    """Crop to the bounding box of nonzero voxels, for already skull-stripped
    input. Reuses MASS's own bbox_from_mask/crop_sitk_zyx (the same functions
    maybe_crop_body uses internally) but with mask=(array != 0) instead of
    make_body_mask's intensity-threshold heuristic -- once the input is truly
    skull-stripped, nonzero already *is* the brain, so the threshold heuristic
    (designed for whole-head input with no better mask available) is not
    needed and would only add imprecision.
    """
    array = mass.image_to_array(image, dtype=np.float32)
    mask = array != 0
    bbox = mass.bbox_from_mask(mask, margin_zyx)
    if bbox is None:
        return image, None
    return mass.crop_sitk_zyx(image, bbox), bbox


def preprocess_one(mass, source_path: Path, dest_path: Path, skull_stripped: bool) -> None:
    import SimpleITK as sitk

    image = mass.read_nifti(str(source_path))
    oriented = mass.reorient_image(image, ORIENTATION)
    resampled = mass.resample_to_spacing(
        oriented, TARGET_SPACING_XYZ, sitk.sitkLinear,
        default_value=0.0, pixel_id=sitk.sitkFloat32,
    )
    if skull_stripped:
        cropped, _bbox = _nonzero_crop(mass, resampled, SKULLSTRIP_MARGIN_ZYX)
    else:
        cropped, _bbox = mass.maybe_crop_body(
            resampled, modality=MODALITY, method=BODY_METHOD,
            margin_zyx=BODY_MARGIN_ZYX, ct_body_threshold=CT_BODY_THRESHOLD,
        )
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    # SimpleITK infers the writer from the file extension, so the temp name
    # must still end in .nii.gz (a ".tmp" suffix on the end makes it
    # unrecognized -- prefix the temp marker instead).
    tmp_path = dest_path.with_name(f".tmp_{dest_path.name}")
    sitk.WriteImage(cropped, str(tmp_path))
    tmp_path.rename(dest_path)


def dest_for(row: dict, out_dir: Path, skull_stripped: bool) -> Path:
    suffix = "ss_1p5mm_crop" if skull_stripped else "mass_1p5mm_bodycrop"
    return out_dir / f"{row['subject_id']}_{row['file_id']}_{suffix}.nii.gz"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="Raw-path manifest (build_manifest_adni_full_mass.py output)")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--out-manifest", required=True)
    parser.add_argument("--audit", default=None, help="Optional JSON audit output path")
    parser.add_argument("--skull-stripped", action="store_true",
                        help="Input paths are already skull-stripped (e.g. HD-BET output): crop to "
                             "the nonzero bounding box instead of MASS's own intensity-threshold "
                             "body crop, which is unnecessary once nonzero already means brain")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    rows = read_manifest(args.manifest)
    mass = _load_mass_inference()

    t0 = time.time()
    done, failed = 0, []
    for index, row in enumerate(rows):
        dest = dest_for(row, out_dir, args.skull_stripped)
        if dest.is_file() and dest.stat().st_size > 0:
            done += 1
        else:
            try:
                preprocess_one(mass, Path(row["path"]), dest, args.skull_stripped)
                done += 1
            except Exception as exc:  # noqa: BLE001 -- keep going, log and report at the end
                failed.append({"subject_id": row["subject_id"], "file_id": row["file_id"],
                               "path": row["path"], "error": str(exc)})
        if (index + 1) % 25 == 0 or index + 1 == len(rows):
            elapsed = time.time() - t0
            print(f"[{elapsed:7.1f}s] {index + 1}/{len(rows)} processed "
                 f"({len(failed)} failed so far)", flush=True)

    out_rows = []
    for row in rows:
        dest = dest_for(row, out_dir, args.skull_stripped)
        if not (dest.is_file() and dest.stat().st_size > 0):
            continue
        out_rows.append({**{key: row[key] for key in FIELDS}, "path": str(dest)})
    write_manifest(out_rows, args.out_manifest)
    print(f"wrote {args.out_manifest}: {len(out_rows)}/{len(rows)} rows have a preprocessed volume")

    if failed:
        print(f"WARNING: {len(failed)} volumes failed preprocessing:", file=sys.stderr)
        for item in failed[:10]:
            print(f"  {item['subject_id']} {item['file_id']}: {item['error']}", file=sys.stderr)

    if args.audit:
        import json

        audit = {"manifest": args.manifest, "out_dir": str(out_dir),
                 "total_rows": len(rows), "succeeded": len(out_rows),
                 "failed": failed, "elapsed_seconds": time.time() - t0,
                 "recipe": {"source": "MASS/inference.py (imported directly)",
                           "orientation": ORIENTATION,
                           "target_spacing_xyz": TARGET_SPACING_XYZ,
                           "body_method": "nonzero_crop" if args.skull_stripped else BODY_METHOD,
                           "body_margin_zyx": SKULLSTRIP_MARGIN_ZYX if args.skull_stripped else BODY_MARGIN_ZYX,
                           "modality": MODALITY, "skull_stripped": args.skull_stripped}}
        Path(args.audit).parent.mkdir(parents=True, exist_ok=True)
        Path(args.audit).write_text(json.dumps(audit, indent=2))
        print(f"wrote {args.audit}")

    if len(out_rows) < len(rows):
        sys.exit(2)  # signal "not fully done yet" to the sbatch chain wrapper


if __name__ == "__main__":
    main()
