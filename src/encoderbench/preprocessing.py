"""Shared NIfTI preprocessing and encoder-native adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


def robust_bounds(volume: np.ndarray, low: float = 1.0, high: float = 99.0) -> tuple[float, float]:
    if volume.ndim != 3 or volume.size == 0:
        raise ValueError(f"Expected a non-empty 3D volume, got shape {volume.shape}")
    if not np.isfinite(volume).all():
        raise ValueError("Input volume contains NaN or Inf")
    foreground = volume[volume > 0]
    values = foreground if foreground.size else volume.reshape(-1)
    lo, hi = float(np.percentile(values, low)), float(np.percentile(values, high))
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def normalize_volume(volume: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    lo, hi = robust_bounds(volume, low, high)
    result = np.clip(volume, lo, hi).astype(np.float32)
    result = (result - lo) / (hi - lo)
    return result


def zscore_nonzero_volume(volume: np.ndarray) -> np.ndarray:
    """Standardize over non-zero (brain) voxels, leaving background pinned at 0.

    Matches MONAI's `NormalizeIntensityd(nonzero=True)` exactly (see
    `monai.transforms.NormalizeIntensity._normalize`): statistics are computed
    over non-zero voxels only, and *only those voxels are rewritten* -- the
    rest of the array is passed through unchanged. On skull-stripped input
    the background is already exactly 0, so it stays exactly 0. An earlier
    version of this function applied `(volume - mean) / std` to the whole
    array, which pushed background to about -3.05 SD (a large uniform block
    BrainIAC never saw during training) instead of leaving it at 0.
    """
    if volume.ndim != 3 or volume.size == 0:
        raise ValueError(f"Expected a non-empty 3D volume, got shape {volume.shape}")
    if not np.isfinite(volume).all():
        raise ValueError("Input volume contains NaN or Inf")
    result = volume.astype(np.float32)
    foreground = result != 0
    if not foreground.any():
        return result
    values = result[foreground]
    mean, std = float(values.mean()), float(values.std())
    div = std if std > 0 else 1.0
    result = result.copy()
    result[foreground] = (values - mean) / div
    return result


def zscore_volume(volume: np.ndarray, clip: Sequence[float] | None = None) -> np.ndarray:
    """Whole-array percentile clip + z-score, matching MASS/inference.py's `normalize_image`.

    MASS computes both the percentile bounds and the mean/std over every voxel
    it receives -- it does not mask out background -- because its own pipeline
    crops tightly to the body first, leaving only a small margin of background.
    This function reproduces that "no masking" behavior faithfully. It is not
    interchangeable with `normalize_volume`: on these ADNI volumes the [0,1]
    rescaling yields brain-voxel mean 0.62 / std 0.26, while the encoder
    expects the input distribution `normalize_image` actually produces.

    Caveat this does *not* fix: MASS's own pipeline also resamples to 1.5mm
    isotropic spacing before that body crop, so voxels have a fixed physical
    size. This benchmark's ADNI files are already skull-stripped and
    bounding-box cropped (`*_ss_box.nii.gz`) with no un-cropped original
    available, so `volume_to_tensor` resizes each subject's box straight to a
    fixed voxel grid instead -- physical scale (mm/voxel) varies subject to
    subject. That gap is architectural (no raw volume to re-resample from),
    not a normalization bug, and is recorded in RESULTS.md rather than silently
    left unmeasured.
    """
    if volume.ndim != 3 or volume.size == 0:
        raise ValueError(f"Expected a non-empty 3D volume, got shape {volume.shape}")
    if not np.isfinite(volume).all():
        raise ValueError("Input volume contains NaN or Inf")
    result = volume.astype(np.float32)
    if clip is not None:
        lo, hi = (float(v) for v in np.percentile(result, list(clip)))
        if hi <= lo:
            hi = lo + 1.0
        result = np.clip(result, lo, hi)
    mean, std = float(result.mean()), float(result.std())
    div = std if std > 1e-6 else 1.0
    return (result - mean) / div


# Each encoder's own repository specifies how its inputs were normalized, and a
# frozen encoder cannot adapt to a different one. Applying each encoder's native
# recipe is what makes the comparison fair: it lets every model see the
# distribution it was trained on rather than one generic pipeline that happens
# to match only some of them.
INTENSITY_MODES = {
    # BrainIAC/src/dataset.py: NormalizeIntensityd(nonzero=True) -- and its
    # ScaleIntensityd(0,1) line is deliberately commented out.
    "zscore_nonzero": lambda volume: zscore_nonzero_volume(volume),
    # MASS/inference.py: "modality-specific clipping followed by per-volume z-score".
    "clip_zscore": lambda volume: zscore_volume(volume, (2.0, 98.0)),
    # AnatCL/README.md asks for Normalize(mean=0, std=1) -- an identity -- over
    # CAT12/VBM grey-matter density maps, which are already bounded in [0,1].
    # We cannot produce CAT12 maps (SPM12/MATLAB), so [0,1] skull-stripped T1
    # stays the closest available numeric proxy for that native range.
    "minmax_percentile": lambda volume: normalize_volume(volume),
}


def load_volume_ras(path: str | Path) -> tuple[np.ndarray, tuple[float, float, float], str]:
    import nibabel as nib

    source = Path(path).expanduser().resolve()
    if not source.is_file() or source.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty NIfTI: {source}")
    try:
        image = nib.as_closest_canonical(nib.load(str(source)))
        volume = np.asarray(image.dataobj, dtype=np.float32)
    except Exception as exc:
        raise ValueError(f"Unable to read NIfTI {source}: {exc}") from exc
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D NIfTI at {source}, got {volume.shape}")
    spacing = tuple(float(v) for v in image.header.get_zooms()[:3])
    orientation = "".join(nib.aff2axcodes(image.affine))
    return volume, spacing, orientation


def slice_indices(length: int, count: int, bounds: Sequence[float] = (0.15, 0.85)) -> list[int]:
    if length < 1 or count < 1:
        raise ValueError("length and count must be positive")
    if len(bounds) != 2 or not 0 <= bounds[0] <= bounds[1] <= 1:
        raise ValueError("slice bounds must satisfy 0 <= low <= high <= 1")
    start = max(0, round(float(bounds[0]) * (length - 1)))
    end = min(length - 1, round(float(bounds[1]) * (length - 1)))
    available = end - start + 1
    if count >= available:
        return list(range(start, end + 1))
    if count == 1:
        return [round((start + end) / 2)]
    return [start + round(i / (count - 1) * (available - 1)) for i in range(count)]


def volume_to_slices(
    path: str | Path,
    axis: int = 2,
    count: int = 24,
    bounds: Sequence[float] = (0.15, 0.85),
    roi_fraction: float = 1.0,
) -> list["Image.Image"]:
    from PIL import Image

    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2")
    if not 0 < roi_fraction <= 1:
        raise ValueError("roi_fraction must be in (0, 1]")
    volume, _, _ = load_volume_ras(path)
    volume = normalize_volume(volume)
    images = []
    for index in slice_indices(volume.shape[axis], count, bounds):
        plane = np.rot90(np.take(volume, index, axis=axis))
        if roi_fraction < 1:
            height, width = plane.shape
            crop_h, crop_w = max(1, round(height * roi_fraction)), max(1, round(width * roi_fraction))
            row, col = (height - crop_h) // 2, (width - crop_w) // 2
            plane = plane[row:row + crop_h, col:col + crop_w]
        pixels = np.rint(plane * 255).astype(np.uint8)
        images.append(Image.fromarray(pixels).convert("RGB"))
    return images


def volume_to_tensor(path: str | Path, shape: Sequence[int],
                     intensity: str = "minmax_percentile") -> "torch.Tensor":
    """Return `[1,D,H,W]` with canonical axial depth, normalized the encoder's own way.

    Resize happens *before* normalization, matching both source pipelines: BrainIAC's
    `Resized` precedes `NormalizeIntensityd` in its MONAI Compose, and MASS's
    `preprocess_target_image` resamples/crops before calling `normalize_image`. Doing
    it the other way round changes which voxels trilinear interpolation blends into the
    zero background, which in turn changes the foreground mask `nonzero`-style stats are
    computed over -- on one measured volume this moved 39% of BrainIAC's foreground
    voxels by more than 0.05 in z-score units (max 2.67).
    """
    import torch
    import torch.nn.functional as functional

    if len(shape) != 3 or any(int(size) <= 0 for size in shape):
        raise ValueError(f"Invalid target shape: {shape}")
    if intensity not in INTENSITY_MODES:
        raise ValueError(f"Unknown intensity mode {intensity!r}; "
                         f"choose from {sorted(INTENSITY_MODES)}")
    volume, _, _ = load_volume_ras(path)
    # Canonical nibabel arrays are (R/L, A/P, I/S). The 3D models use
    # (depth, height, width), so axial I/S is first, matching the 2D slice order.
    volume = np.transpose(volume, (2, 1, 0)).copy()
    tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0)
    resized = functional.interpolate(tensor, size=tuple(map(int, shape)), mode="trilinear",
                                     align_corners=False)
    normalized = INTENSITY_MODES[intensity](resized.squeeze(0).squeeze(0).numpy())
    return torch.from_numpy(normalized).unsqueeze(0).contiguous()


def load_posterior_tensor(path: str | Path, shape: Sequence[int]) -> "torch.Tensor":
    """Return `[C,D,H,W]` resampled class-posterior probabilities; values are not renormalized."""
    import nibabel as nib
    import torch
    import torch.nn.functional as functional

    if len(shape) != 3 or any(int(size) <= 0 for size in shape):
        raise ValueError(f"Invalid target shape: {shape}")
    source = Path(path).expanduser().resolve()
    if not source.is_file() or source.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty NIfTI: {source}")
    try:
        image = nib.as_closest_canonical(nib.load(str(source)))
        volume = np.asarray(image.dataobj, dtype=np.float32)
    except Exception as exc:
        raise ValueError(f"Unable to read NIfTI {source}: {exc}") from exc
    if volume.ndim != 4:
        raise ValueError(f"Expected a 4D posterior NIfTI at {source}, got {volume.shape}")
    if not np.isfinite(volume).all():
        raise ValueError("Posterior volume contains NaN or Inf")
    # Canonical nibabel arrays are (R/L, A/P, I/S, C); move class to the front and
    # put I/S depth first, matching volume_to_tensor's (depth, height, width) order.
    volume = np.transpose(volume, (3, 2, 1, 0)).copy()
    tensor = torch.from_numpy(volume).unsqueeze(0)
    resized = functional.interpolate(tensor, size=tuple(map(int, shape)), mode="trilinear",
                                     align_corners=False)
    return resized.squeeze(0).contiguous()


def save_montage(images: Sequence["Image.Image"], path: str | Path, columns: int = 6) -> Path:
    from PIL import Image, ImageDraw

    if not images:
        raise ValueError("Cannot create a montage without images")
    thumbnails = [image.copy() for image in images]
    for image in thumbnails:
        image.thumbnail((192, 192))
    width = max(image.width for image in thumbnails)
    height = max(image.height for image in thumbnails)
    rows = (len(thumbnails) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * width, rows * (height + 18)), "black")
    draw = ImageDraw.Draw(canvas)
    for index, image in enumerate(thumbnails):
        x, y = (index % columns) * width, (index // columns) * (height + 18)
        canvas.paste(image, (x, y))
        draw.text((x + 3, y + height + 2), str(index + 1), fill="white")
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return output
