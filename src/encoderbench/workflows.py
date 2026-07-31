"""Feature extraction and Phase 0/data audit workflows."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from encoderbench.cache import FeatureCache, save_cache
from encoderbench.manifest import read_manifest, validate_manifest
from encoderbench.preprocessing import load_volume_ras, normalize_volume, save_montage, volume_to_slices
from encoderbench.utils import write_json


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def audit_dataset(manifest_path: str | Path, disease: str, output_dir: str | Path,
                  data_settings: dict[str, Any]) -> dict[str, Any]:
    rows = read_manifest(manifest_path)
    summary = validate_manifest(rows, disease)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    records, failures = [], []
    representatives: dict[str, dict[str, str]] = {}
    for row in rows:
        try:
            volume, spacing, orientation = load_volume_ras(row["path"])
            normalized = normalize_volume(volume, data_settings["low_percentile"],
                                          data_settings["high_percentile"])
            records.append({"file_id": row["file_id"], "subject_id": row["subject_id"],
                            "group": row["group"], "split": row["split"],
                            "shape": list(volume.shape), "spacing": list(spacing),
                            "orientation": orientation, "normalized_min": float(normalized.min()),
                            "normalized_max": float(normalized.max()),
                            "normalized_mean": float(normalized.mean()),
                            "normalized_std": float(normalized.std())})
            representatives.setdefault(row["group"], row)
        except Exception as exc:  # retain a complete corruption audit
            failures.append({"file_id": row["file_id"], "path": row["path"],
                             "error": f"{type(exc).__name__}: {exc}"})
    montages = {}
    for group, row in representatives.items():
        images = volume_to_slices(row["path"], axis=data_settings["axis"],
                                  count=data_settings["num_slices"],
                                  bounds=data_settings["slice_range"],
                                  roi_fraction=data_settings["roi_fraction"])
        montage = save_montage(images, output / f"representative_{group}.png")
        montages[group] = str(montage)
    summary.update({"disease": disease, "label_counts": dict(Counter(row["group"] for row in rows)),
                    "split_counts": {f"{split}:{label}": count for (split, label), count in
                                     Counter((row["split"], row["group"]) for row in rows).items()},
                    "orientation_counts": dict(Counter(record["orientation"] for record in records)),
                    "shape_counts": dict(Counter("x".join(map(str, record["shape"]))
                                                 for record in records)),
                    "spacing_counts": dict(Counter("x".join(f"{v:.4g}" for v in record["spacing"])
                                                   for record in records)),
                    "failures": failures, "montages": montages, "records": records})
    write_json(output / "dataset_audit.json", summary)
    return summary


def smoke_extractor(manifest_path: str | Path, disease: str, encoder: str,
                    config: dict[str, Any], output_path: str | Path,
                    device: str = "auto") -> dict[str, Any]:
    from encoderbench.extractors import build_extractor

    rows = read_manifest(manifest_path)
    positive = "AD" if disease == "ad" else "SCZ"
    selected = []
    for label in (positive, "CN"):
        matches = [row for row in rows if row["group"] == label]
        if not matches:
            raise ValueError(f"No {label} sample exists for the two-case smoke test")
        selected.append(sorted(matches, key=lambda row: row["file_id"])[0])
    extractor = build_extractor(encoder, config, resolve_device(device))
    cases = []
    expected_shape = None
    for row in selected:
        tokens = extractor.extract(row["path"])
        audit = extractor.audit(tokens)
        if expected_shape is None:
            expected_shape = tuple(tokens.shape)
        elif tuple(tokens.shape) != expected_shape:
            raise ValueError(f"Unstable token shape: expected {expected_shape}, got {tuple(tokens.shape)}")
        cases.append({"group": row["group"], "file_id": row["file_id"], "audit": audit})
    result = {"phase": 0, "disease": disease, "encoder": encoder, "cases": cases,
              "shape_stable": True, "excluded_tokens": ["CLS", "global_pool", "padding",
                                                            "projector_output"]}
    write_json(output_path, result)
    return result


def cache_features(manifest_path: str | Path, disease: str, encoder: str,
                   config: dict[str, Any], output_path: str | Path,
                   device: str = "auto", layer: int | None = None) -> Path:
    from encoderbench.extractors import build_extractor

    rows = read_manifest(manifest_path)
    extractor = build_extractor(encoder, config, resolve_device(device), layer)
    features, audits = [], []
    native_grids: set[tuple[int, int, int]] = set()
    for position, row in enumerate(rows):
        tokens = extractor.extract(row["path"])
        features.append(tokens.numpy())
        if extractor.last_native_grid is not None:
            native_grids.add(extractor.last_native_grid)
        if position < 2:
            audits.append(extractor.audit(tokens))
    array = np.stack(features)
    dtype = np.float16 if config["features"]["cache_dtype"] == "float16" else np.float32
    metadata = {"disease": disease, "encoder": encoder, "sample_count": len(rows),
                "layer": extractor.layer,
                "token_shape": list(array.shape[1:]), "native_grids": [list(g) for g in sorted(native_grids)],
                "position_encoding": "fixed_3d_sinusoidal", "sample_audits": audits}
    cache = FeatureCache(features=array.astype(dtype),
                         file_ids=np.asarray([row["file_id"] for row in rows]),
                         subject_ids=np.asarray([row["subject_id"] for row in rows]),
                         labels=np.asarray([row["group"] for row in rows]),
                         splits=np.asarray([row["split"] for row in rows]), metadata=metadata)
    return save_cache(output_path, cache)
