"""Configuration loading with cross-experiment fairness validation."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from encoderbench.utils import resolve_env


DEFAULT_CONFIG = Path("/net/projects2/litian-lab/scpan/encoders/config/default.yaml")
ENCODERS = ("medsiglip", "braingemma3d", "mass", "brainiac", "anatcl", "synthseg")
DISEASES = ("ad", "scz", "bsnip2")


@dataclass(frozen=True)
class ExperimentConfig:
    raw: dict[str, Any]
    source: Path

    def section(self, name: str) -> dict[str, Any]:
        value = self.raw.get(name)
        if not isinstance(value, dict):
            raise ValueError(f"Configuration section {name!r} is missing or not a mapping")
        return value

    def manifest(self, disease: str) -> Path:
        if disease not in DISEASES:
            raise ValueError(f"Unknown disease {disease!r}; choose from {DISEASES}")
        return Path(self.section("manifests")[disease]).resolve()

    @property
    def output_root(self) -> Path:
        return Path(self.raw["output_root"]).resolve()


def load_config(path: str | Path | None = None) -> ExperimentConfig:
    selected = Path(path or os.environ.get("ENCODERBENCH_CONFIG", DEFAULT_CONFIG)).resolve()
    if not selected.is_file():
        raise FileNotFoundError(f"Configuration file not found: {selected}")
    with selected.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Configuration root must be a mapping")
    raw = resolve_env(raw)
    _validate(raw)
    return ExperimentConfig(raw=raw, source=selected)


def _validate_seeds(name: str, seeds: Any) -> None:
    """Seeds must *extend* the canonical [0, 1, 2], never replace or reorder it.

    The original guard pinned every formal experiment to exactly [0, 1, 2] so
    numbers stayed comparable across experiments. That intent is preserved here
    while allowing more seeds, because three of them cannot resolve the ~0.02
    differences this series is trying to measure: on a 65-subject validation
    split the standard error of AUC is about 0.034, wider than most of the
    A-vs-B and dataset2-vs-dataset4 gaps being compared. Requiring the list to
    be exactly range(n) keeps seeds 0-2 present and first, so any comparison
    against an older three-seed run still reads the same three runs.
    """
    values = tuple(seeds)
    if len(values) < 3 or values != tuple(range(len(values))):
        raise ValueError(f"{name} seeds must be [0, 1, 2] or an extension of it "
                         f"(for example [0, 1, 2, 3, 4]); got {list(values)}")


def _validate(raw: dict[str, Any]) -> None:
    for section in ("manifests", "checkpoints", "source_repositories", "data", "features",
                    "probe", "finetune", "bridge", "evaluation"):
        if not isinstance(raw.get(section), dict):
            raise ValueError(f"Missing configuration section: {section}")
    data = raw["data"]
    if data.get("split_seed") != 0:
        raise ValueError("The data split seed is frozen at 0")
    if data.get("axis") != 2 or data.get("num_slices") != 24:
        raise ValueError("The shared slice protocol is frozen at axis=2 and num_slices=24")
    if tuple(data.get("slice_range", ())) != (0.15, 0.85):
        raise ValueError("The shared slice range is frozen at [0.15, 0.85]")
    if tuple(raw["features"].get("pooled_grid", ())) != (4, 4, 4):
        raise ValueError("The common token grid must be [4, 4, 4]")
    _validate_seeds("Probe", raw["probe"].get("seeds", ()))
    finetune = raw["finetune"]
    _validate_seeds("Finetune", finetune.get("seeds", ()))
    if int(finetune.get("parameter_budget", 0)) <= 0:
        raise ValueError("Finetune parameter_budget must be positive")
    if int(finetune.get("gradient_accumulation", 0)) <= 0:
        raise ValueError("Finetune gradient_accumulation must be positive")
    if finetune.get("precision") != "bf16":
        raise ValueError("Finetune precision must be bf16")
    bridge = raw["bridge"]
    _validate_seeds("Bridge", bridge.get("seeds", ()))
    effective_batch = int(bridge["micro_batch_size"]) * int(bridge["gradient_accumulation"])
    if effective_batch != 16:
        raise ValueError(f"Bridge effective batch size must equal 16, got {effective_batch}")
    max_new_tokens = int(raw["evaluation"].get("zero_shot_max_new_tokens", 0))
    if not 1 <= max_new_tokens <= 256:
        raise ValueError("zero_shot_max_new_tokens must be between 1 and 256")
