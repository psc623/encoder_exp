"""AD-to-SCZ phase gate and immutable configuration lock."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from encoderbench.config import ENCODERS
from encoderbench.utils import sha256_file, write_json


LOCK_NAME = "ad_configuration_lock.json"


def create_ad_lock(output_root: str | Path, config_path: str | Path) -> Path:
    root = Path(output_root).resolve()
    summaries = []
    for path in root.rglob("*_summary.json"):
        try:
            with path.open(encoding="utf-8") as handle:
                value = json.load(handle)
            if value.get("disease") == "ad":
                summaries.append(value)
        except (OSError, json.JSONDecodeError):
            continue
    probe = {(item.get("encoder"), item.get("seed"), bool(item.get("shuffled_labels")))
             for item in summaries if item.get("kind") == "attention_probe"}
    bridge = {(item.get("encoder"), item.get("seed"), item.get("kind"))
              for item in summaries if item.get("kind") in ("linear_bridge", "resampler_bridge")}
    zero = {item.get("model") for item in summaries if item.get("kind") == "zero_shot"}
    missing = []
    for encoder in ENCODERS:
        for seed in (0, 1, 2):
            for shuffled in (False, True):
                if (encoder, seed, shuffled) not in probe:
                    missing.append(f"probe:{encoder}:seed{seed}:shuffled={shuffled}")
            for kind in ("linear_bridge", "resampler_bridge"):
                if (encoder, seed, kind) not in bridge:
                    missing.append(f"{kind}:{encoder}:seed{seed}")
    for model in ("medgemma", "braingemma3d"):
        if model not in zero:
            missing.append(f"zero_shot:{model}")
    if missing:
        preview = ", ".join(missing[:8])
        raise RuntimeError(f"AD phase is incomplete ({len(missing)} missing runs): {preview}")
    lock = {"phase": "AD complete; configuration locked for SCZ transfer",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config_path": str(Path(config_path).resolve()),
            "config_sha256": sha256_file(config_path), "verified_run_count": len(summaries)}
    return write_json(root / LOCK_NAME, lock)


def require_ad_lock(output_root: str | Path, config_path: str | Path) -> None:
    lock_path = Path(output_root).resolve() / LOCK_NAME
    if not lock_path.is_file():
        raise RuntimeError(
            "SCZ execution is gated until AD is complete. Run `encoderbench lock-ad` after all AD runs."
        )
    with lock_path.open(encoding="utf-8") as handle:
        lock = json.load(handle)
    current = sha256_file(config_path)
    if lock.get("config_sha256") != current:
        raise RuntimeError("Configuration changed after the AD lock; SCZ transfer must use the locked settings")

