"""Rigid-register already skull-stripped ADNI volumes to BrainIAC's own
registration template (temp_head.nii.gz), skipping HD-BET since the input
is already brain-extracted (by AFNI @SSwarper upstream).

Mirrors the registration() step in
BrainIAC/src/preprocessing/mri_preprocess_3d_simple.py exactly (same N4
bias correction, same Euler3D + Mattes Mutual Information rigid
registration, same multi-resolution schedule) minus the HD-BET call.

Two modes, driven by shell-level parallelism (xargs) instead of Python
multiprocessing, since this cluster's venvs symlink to the node's floating
/usr/bin/python3 and spawned worker processes can land on a different
interpreter than the parent on some nodes:

  --single-index N   process exactly one manifest row, then exit
  --merge            scan --out-dir for finished outputs and write the
                      final manifest (only rows whose output file exists)
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_rows(manifest: str) -> list[dict]:
    with open(manifest, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def output_path_for(row: dict, out_dir: str) -> Path:
    return Path(out_dir) / f"{row['file_id']}_ss_box_reg.nii.gz"


def register_one(row: dict, template_path: str, out_dir: str) -> None:
    import SimpleITK as sitk

    output_path = output_path_for(row, out_dir)
    if output_path.is_file() and output_path.stat().st_size > 0:
        print(f"SKIP (exists) {row['file_id']}", flush=True)
        return

    fixed_img = sitk.ReadImage(template_path, sitk.sitkFloat32)
    moving_img = sitk.ReadImage(row["path"], sitk.sitkFloat32)
    moving_img = sitk.N4BiasFieldCorrection(moving_img)

    old_size = fixed_img.GetSize()
    old_spacing = fixed_img.GetSpacing()
    new_spacing = (1, 1, 1)
    new_size = [
        int(round((old_size[0] * old_spacing[0]) / float(new_spacing[0]))),
        int(round((old_size[1] * old_spacing[1]) / float(new_spacing[1]))),
        int(round((old_size[2] * old_spacing[2]) / float(new_spacing[2]))),
    ]
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(new_size)
    resample.SetOutputOrigin(fixed_img.GetOrigin())
    resample.SetOutputDirection(fixed_img.GetDirection())
    resample.SetInterpolator(sitk.sitkLinear)
    resample.SetDefaultPixelValue(fixed_img.GetPixelIDValue())
    resample.SetOutputPixelType(sitk.sitkFloat32)
    fixed_img = resample.Execute(fixed_img)

    transform = sitk.CenteredTransformInitializer(
        fixed_img, moving_img, sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY)

    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=1.0, numberOfIterations=100,
        convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    registration_method.SetOptimizerScalesFromPhysicalShift()
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration_method.SetInitialTransform(transform)

    final_transform = registration_method.Execute(fixed_img, moving_img)
    moving_resampled = sitk.Resample(
        moving_img, fixed_img, final_transform, sitk.sitkLinear, 0.0,
        moving_img.GetPixelID())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f"tmp_{output_path.name}")
    sitk.WriteImage(moving_resampled, str(tmp_path))
    tmp_path.rename(output_path)
    print(f"OK {row['file_id']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--template", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--out-manifest", required=True)
    parser.add_argument("--single-index", type=int, default=None,
                        help="Process exactly this 0-based manifest row and exit")
    parser.add_argument("--merge", action="store_true",
                        help="Assemble the final manifest from finished outputs")
    args = parser.parse_args()

    rows = read_rows(args.manifest)

    if args.single_index is not None:
        register_one(rows[args.single_index], args.template, args.out_dir)
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
        print(f"DONE: {len(done)}/{len(rows)} registered, {len(missing)} missing")
        if missing:
            print("Missing IDs:", missing)
        print(f"Wrote -> {args.out_manifest}")
        return

    parser.error("Pass either --single-index N or --merge")


if __name__ == "__main__":
    main()
