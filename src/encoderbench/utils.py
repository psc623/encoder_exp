"""Shared deterministic, serialization, and checkpoint helpers."""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch (when installed)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def ensure_parent(path: str | Path) -> Path:
    result = Path(path).expanduser().resolve()
    result.parent.mkdir(parents=True, exist_ok=True)
    return result


def write_json(path: str | Path, value: Any) -> Path:
    output = ensure_parent(path)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    return output


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_identifier(path: str | Path) -> dict[str, Any]:
    """Return a deterministic identifier without hashing every multi-GB shard."""
    target = Path(path).expanduser().resolve()
    if not target.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {target}")
    if target.is_file():
        stat = target.stat()
        return {
            "path": str(target),
            "kind": "file",
            "size_bytes": stat.st_size,
            "sha256": sha256_file(target),
        }
    preferred = [
        "config.json",
        "model_config.json",
        "model.safetensors.index.json",
        "projector_vis_scale.pt",
    ]
    entries: list[dict[str, Any]] = []
    for name in preferred:
        candidate = target / name
        if candidate.is_file():
            entries.append(
                {"relative_path": name, "size_bytes": candidate.stat().st_size,
                 "sha256": sha256_file(candidate)}
            )
    if not entries:
        entries = [
            {"relative_path": str(p.relative_to(target)), "size_bytes": p.stat().st_size}
            for p in sorted(target.rglob("*")) if p.is_file()
        ]
    return {"path": str(target), "kind": "directory", "identifiers": entries}


def resolve_env(value: Any) -> Any:
    """Expand environment variables and user paths recursively in config values."""
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, list):
        return [resolve_env(item) for item in value]
    if isinstance(value, dict):
        return {key: resolve_env(item) for key, item in value.items()}
    return value

