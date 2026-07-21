"""Portable feature-cache format and integrity checks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class FeatureCache:
    features: np.ndarray
    file_ids: np.ndarray
    subject_ids: np.ndarray
    labels: np.ndarray
    splits: np.ndarray
    metadata: dict[str, Any]

    def validate(self) -> None:
        if self.features.ndim != 3 or self.features.shape[1] != 64:
            raise ValueError(f"Expected cache shape [samples,64,dim], got {self.features.shape}")
        count = self.features.shape[0]
        for name, values in (("file_ids", self.file_ids), ("subject_ids", self.subject_ids),
                             ("labels", self.labels), ("splits", self.splits)):
            if len(values) != count:
                raise ValueError(f"{name} has {len(values)} rows but features has {count}")
        if not np.isfinite(self.features).all():
            raise ValueError("Feature cache contains NaN or Inf")
        if set(self.splits.tolist()) != {"train", "validation", "test"}:
            raise ValueError("Feature cache must contain train, validation, and test")


def save_cache(path: str | Path, cache: FeatureCache) -> Path:
    cache.validate()
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, features=cache.features, file_ids=cache.file_ids,
                        subject_ids=cache.subject_ids, labels=cache.labels, splits=cache.splits,
                        metadata=np.asarray(json.dumps(cache.metadata, sort_keys=True)))
    return output


def load_cache(path: str | Path) -> FeatureCache:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Feature cache not found: {source}")
    with np.load(source, allow_pickle=False) as data:
        required = {"features", "file_ids", "subject_ids", "labels", "splits", "metadata"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"Feature cache is missing arrays: {sorted(missing)}")
        cache = FeatureCache(
            features=data["features"].astype(np.float32), file_ids=data["file_ids"],
            subject_ids=data["subject_ids"], labels=data["labels"], splits=data["splits"],
            metadata=json.loads(str(data["metadata"].item())),
        )
    cache.validate()
    return cache
