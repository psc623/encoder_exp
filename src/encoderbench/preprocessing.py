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


def volume_to_tensor(path: str | Path, shape: Sequence[int]) -> "torch.Tensor":
    """Return `[1,D,H,W]` with canonical axial depth; resize is the only adapter."""
    import torch
    import torch.nn.functional as functional

    if len(shape) != 3 or any(int(size) <= 0 for size in shape):
        raise ValueError(f"Invalid target shape: {shape}")
    volume, _, _ = load_volume_ras(path)
    # Canonical nibabel arrays are (R/L, A/P, I/S). The 3D models use
    # (depth, height, width), so axial I/S is first, matching the 2D slice order.
    volume = np.transpose(volume, (2, 1, 0)).copy()
    tensor = torch.from_numpy(normalize_volume(volume)).unsqueeze(0).unsqueeze(0)
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
