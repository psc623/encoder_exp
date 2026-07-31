"""Run MASS's own native geometry pipeline on raw whole-head BSNIP2 T1 volumes.

ADNI could not exercise this because its files arrived already skull-stripped
and bounding-box cropped, with no un-cropped original to resample from
(RESULTS.md: "MASS resamples to 1.5mm isotropic before its own body crop; we
have no un-cropped original to resample from"). BSNIP2's raw series files are
genuinely raw, so this calls MASS/inference.py's real
`preprocess_target_image` (reorient to RAS -> resample to MASS's native
1.5mm isotropic spacing -> threshold body crop) directly, using MASS's own
CLI defaults. Only the geometry step runs here; MASS's own
`normalize_image` (2-98th percentile clip + whole-array z-score) is applied
later at cache time by encoderbench's `clip_zscore` intensity mode, matching
how the ADNI pipeline already separates geometry from intensity.

Same --single-index / --merge split as bsnip2_register_brainiac.py, driven by
shell-level xargs parallelism (pure CPU/SimpleITK work, no GPU needed).
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

MASS_ROOT = Path("/net/projects2/litian-lab/scpan/MASS")
MASS_INFERENCE_PATH = MASS_ROOT / "inference.py"

# MASS/inference.py's own argparse defaults (see --target-spacing, --orientation,
# --modality, --body-method, --body-margin in its main()).
ORIENTATION = "RAS"
TARGET_SPACING_XYZ = (1.5, 1.5, 1.5)
MODALITY = "auto"
BODY_METHOD = "threshold"
BODY_MARGIN_ZYX = (16, 32, 32)
CT_BODY_THRESHOLD = -500.0
CT_CLIP = (-991.0, 500.0)


def _load_mass_inference():
    if str(MASS_ROOT) not in sys.path:
        sys.path.insert(0, str(MASS_ROOT))
    spec = importlib.util.spec_from_file_location("mass_inference", MASS_INFERENCE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_rows(manifest: str) -> list[dict]:
    with open(manifest, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def output_path_for(row: dict, out_dir: str) -> Path:
    return Path(out_dir) / f"{row['file_id']}_mass_prep.nii.gz"


def process_one(row: dict, out_dir: str, mass) -> None:
    import SimpleITK as sitk

    output_path = output_path_for(row, out_dir)
    if output_path.is_file() and output_path.stat().st_size > 0:
        print(f"SKIP (exists) {row['file_id']}", flush=True)
        return

    _original, cropped, _tensor = mass.preprocess_target_image(
        image_path=row["path"],
        orientation=ORIENTATION,
        target_spacing_xyz=TARGET_SPACING_XYZ,
        image_interpolator=sitk.sitkLinear,
        modality=MODALITY,
        body_method=BODY_METHOD,
        body_margin_zyx=BODY_MARGIN_ZYX,
        ct_body_threshold=CT_BODY_THRESHOLD,
        ct_clip=CT_CLIP,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f"tmp_{output_path.name}")
    sitk.WriteImage(cropped, str(tmp_path))
    tmp_path.rename(output_path)
    print(f"OK {row['file_id']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--out-manifest", default=None)
    parser.add_argument("--single-index", type=int, default=None)
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()

    rows = read_rows(args.manifest)

    if args.single_index is not None:
        mass = _load_mass_inference()
        process_one(rows[args.single_index], args.out_dir, mass)
        return

    if args.merge:
        fieldnames = list(rows[0].keys())
        done, missing = [], []
        for row in rows:
            output_path = output_path_for(row, args.out_dir)
            if output_path.is_file() and output_path.stat().st_size > 0:
                done.append(row | {"path": str(output_path)})
            else:
                missing.append(row["file_id"])
        with open(args.out_manifest, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in done:
                writer.writerow({key: row[key] for key in fieldnames})
        print(f"DONE: {len(done)}/{len(rows)} preprocessed, {len(missing)} missing")
        if missing:
            print("Missing IDs:", missing)
        print(f"Wrote -> {args.out_manifest}")
        return

    parser.error("Pass either --single-index N or --merge")


if __name__ == "__main__":
    main()
