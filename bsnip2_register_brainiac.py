"""Rigid-register raw whole-head BSNIP2 T1 volumes to BrainIAC's own
registration template (temp_head.nii.gz).

Unlike register_brainiac_template.py (ADNI), BSNIP2's raw manifest rows are
genuinely un-skull-stripped whole-head scans, so this mirrors the full
registration() step in BrainIAC/src/preprocessing/mri_preprocess_3d_simple.py
(same N4 bias correction, same Euler3D + Mattes Mutual Information rigid
registration, same multi-resolution schedule). HD-BET skull-stripping is a
separate later step (bsnip2_hdbet_batch.py) run once in a batched GPU pass
rather than per-file here, since spinning up a fresh process per volume would
reload the network every time.

Same --single-index / --merge split as register_brainiac_template.py, for the
same reason: this cluster's venvs symlink to the node's floating
/usr/bin/python3, so shell-level xargs parallelism is used instead of Python
multiprocessing.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_rows(manifest: str) -> list[dict]:
    with open(manifest, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def output_path_for(row: dict, out_dir: str) -> Path:
    return Path(out_dir) / f"{row['file_id']}_reg.nii.gz"


def n4_correct_fast(image: "sitk.Image", shrink_factor: int = 4) -> "sitk.Image":
    """N4 on a shrunk copy, then apply the (smooth, low-frequency) bias field
    back at full resolution -- the standard SimpleITK speedup recipe.

    BSNIP2's raw whole-head volumes are 256x256x170 (~11M voxels) vs. ADNI's
    already skull-stripped/box-cropped 122x127x120 (~1.9M voxels) that
    register_brainiac_template.py's plain `sitk.N4BiasFieldCorrection(image)`
    call was tuned against. N4's cost grows much faster than linearly with
    voxel count, and running it unshrunk here measured well under 1 volume
    per 30 minutes even after fixing thread oversubscription -- 509 volumes
    would not finish in a reasonable window. The bias field itself is
    low-frequency by construction, so estimating it on a shrunk image and
    resampling the field (not the corrected image) back up is standard
    practice, not an accuracy shortcut.
    """
    import SimpleITK as sitk

    mask_image = sitk.OtsuThreshold(image, 0, 1, 200)
    image_shrunk = sitk.Shrink(image, [shrink_factor] * image.GetDimension())
    mask_shrunk = sitk.Shrink(mask_image, [shrink_factor] * image.GetDimension())
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.Execute(image_shrunk, mask_shrunk)
    log_bias_field = corrector.GetLogBiasFieldAsImage(image)
    # sitk's arithmetic operators promote to float64, which downstream
    # CenteredTransformInitializer rejects as a type mismatch against the
    # sitkFloat32 fixed/moving images it expects.
    return sitk.Cast(image / sitk.Exp(log_bias_field), sitk.sitkFloat32)


def register_one(row: dict, template_path: str, out_dir: str) -> None:
    import SimpleITK as sitk

    # Parallelism already comes from the caller's `xargs -P 32` (one process per
    # volume). SimpleITK/ITK filters multithread internally by default, so
    # without this every one of those 32 processes would also fan out across
    # all available cores -- 32x oversubscription on top of whatever else is
    # already running on this (shared, non-exclusive) node.
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)

    output_path = output_path_for(row, out_dir)
    if output_path.is_file() and output_path.stat().st_size > 0:
        print(f"SKIP (exists) {row['file_id']}", flush=True)
        return

    fixed_img = sitk.ReadImage(template_path, sitk.sitkFloat32)
    moving_img = sitk.ReadImage(row["path"], sitk.sitkFloat32)
    moving_img = n4_correct_fast(moving_img)

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
    parser.add_argument("--out-manifest", default=None)
    parser.add_argument("--single-index", type=int, default=None)
    parser.add_argument("--merge", action="store_true")
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
