"""EEG_AD preprocessing and leakage-safe window datasets for BIOT."""

from __future__ import annotations

import bisect
import fcntl
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from scipy.signal import resample_poly
from torch.utils.data import Dataset


BIOT_CHANNELS = (
    "FP1-F7",
    "F7-T7",
    "T7-P7",
    "P7-O1",
    "FP2-F8",
    "F8-T8",
    "T8-P8",
    "P8-O2",
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
)

CACHE_FORMAT_VERSION = 1

# EEG_AD uses the legacy temporal labels T3/T4/T5/T6.  They correspond to
# modern T7/T8/P7/P8 and yield the exact BIOT bipolar ordering above.
EEG_AD_BIPOLAR_PAIRS = (
    ("FP1", "F7"),
    ("F7", "T3"),
    ("T3", "T5"),
    ("T5", "O1"),
    ("FP2", "F8"),
    ("F8", "T4"),
    ("T4", "T6"),
    ("T6", "O2"),
    ("FP1", "F3"),
    ("F3", "C3"),
    ("C3", "P3"),
    ("P3", "O1"),
    ("FP2", "F4"),
    ("F4", "C4"),
    ("C4", "P4"),
    ("P4", "O2"),
)


@dataclass(frozen=True)
class SubjectRecord:
    subject_id: str
    label: int
    group: str
    set_path: Path


@dataclass(frozen=True)
class CachedSubject:
    subject_id: str
    label: int
    group: str
    cache_path: Path
    n_windows: int


@dataclass(frozen=True)
class PreprocessingConfig:
    target_rate: int = 200
    window_seconds: float = 10.0
    stride_seconds: float = 10.0
    quantile: float = 0.95

    def __post_init__(self) -> None:
        if self.target_rate <= 0:
            raise ValueError("target_rate must be positive")
        if self.window_seconds <= 0 or self.stride_seconds <= 0:
            raise ValueError("window_seconds and stride_seconds must be positive")
        if not 0.0 < self.quantile <= 1.0:
            raise ValueError("quantile must be in (0, 1]")


def _canonical_channel(name: str) -> str:
    return "".join(character for character in str(name).upper() if character.isalnum())


def load_ac_subjects(dataset_root: str | Path, use_derivatives: bool = True) -> list[SubjectRecord]:
    """Load only Alzheimer (A=1) and control (C=0); F is excluded entirely."""

    dataset_root = Path(dataset_root)
    participants_path = dataset_root / "participants.tsv"
    if not participants_path.is_file():
        raise FileNotFoundError(f"participants.tsv not found: {participants_path}")
    participants = pd.read_csv(participants_path, sep="\t", dtype=str)
    required_columns = {"participant_id", "Group"}
    missing_columns = required_columns - set(participants.columns)
    if missing_columns:
        raise ValueError(f"participants.tsv lacks columns: {sorted(missing_columns)}")

    participants["Group"] = participants["Group"].str.strip().str.upper()
    participants = participants[participants["Group"].isin(["A", "C"])].copy()
    if participants.empty:
        raise ValueError("No A/C participants found after excluding Group F")

    base = dataset_root / "derivatives" if use_derivatives else dataset_root
    label_map = {"C": 0, "A": 1}
    records: list[SubjectRecord] = []
    for row in participants.itertuples(index=False):
        subject_id = str(row.participant_id).strip()
        group = str(row.Group).strip().upper()
        set_path = base / subject_id / "eeg" / f"{subject_id}_task-eyesclosed_eeg.set"
        if not set_path.is_file():
            raise FileNotFoundError(f"Missing EEG file for {subject_id}: {set_path}")
        records.append(SubjectRecord(subject_id, label_map[group], group, set_path))

    subject_ids = [record.subject_id for record in records]
    if len(subject_ids) != len(set(subject_ids)):
        raise ValueError("participants.tsv contains duplicate A/C participant IDs")
    return records


def read_eeglab_set(path: str | Path) -> tuple[np.ndarray, float, list[str]]:
    """Read the embedded MATLAB-v5 EEGLAB files in EEG_AD without MNE."""

    path = Path(path)
    mat = loadmat(
        path,
        simplify_cells=True,
        variable_names=["data", "srate", "nbchan", "chanlocs"],
    )
    required = {"data", "srate", "nbchan", "chanlocs"}
    missing = required - set(mat)
    if missing:
        raise ValueError(f"{path} lacks EEGLAB variables: {sorted(missing)}")

    data = np.asarray(mat["data"], dtype=np.float32)
    n_channels = int(mat["nbchan"])
    if data.ndim != 2:
        raise ValueError(f"Expected continuous 2D EEG in {path}, received {data.shape}")
    if data.shape[0] != n_channels and data.shape[1] == n_channels:
        data = data.T
    if data.shape[0] != n_channels:
        raise ValueError(
            f"EEGLAB nbchan={n_channels}, but data shape is {data.shape} in {path}"
        )

    channel_locations = mat["chanlocs"]
    if isinstance(channel_locations, dict):
        channel_locations = [channel_locations]
    channel_names = [str(location["labels"]).strip() for location in channel_locations]
    if len(channel_names) != n_channels:
        raise ValueError(
            f"Found {len(channel_names)} channel labels for {n_channels} channels in {path}"
        )
    if not np.isfinite(data).all():
        raise ValueError(f"Non-finite samples found in {path}")
    return data, float(mat["srate"]), channel_names


def build_biot_bipolar_montage(data: np.ndarray, channel_names: list[str]) -> np.ndarray:
    """Convert EEG_AD's 19 referential electrodes to BIOT's ordered 16 bipolar channels."""

    index: dict[str, int] = {}
    for position, name in enumerate(channel_names):
        canonical = _canonical_channel(name)
        if canonical in index:
            raise ValueError(f"Duplicate channel after canonicalization: {name}")
        index[canonical] = position

    required = {_canonical_channel(name) for pair in EEG_AD_BIPOLAR_PAIRS for name in pair}
    missing = sorted(required - set(index))
    if missing:
        raise ValueError(f"Cannot construct BIOT montage; missing EEG_AD electrodes: {missing}")

    montage = np.stack(
        [
            data[index[_canonical_channel(left)]] - data[index[_canonical_channel(right)]]
            for left, right in EEG_AD_BIPOLAR_PAIRS
        ],
        axis=0,
    )
    return np.asarray(montage, dtype=np.float32)


def preprocess_recording(
    set_path: str | Path,
    config: PreprocessingConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    data, source_rate, channel_names = read_eeglab_set(set_path)
    montage = build_biot_bipolar_montage(data, channel_names)

    rounded_rate = int(round(source_rate))
    if not math.isclose(source_rate, rounded_rate, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"Non-integral source sampling rate {source_rate} in {set_path}")
    if rounded_rate != config.target_rate:
        divisor = math.gcd(rounded_rate, config.target_rate)
        montage = resample_poly(
            montage,
            up=config.target_rate // divisor,
            down=rounded_rate // divisor,
            axis=-1,
        ).astype(np.float32, copy=False)

    window_points = int(round(config.window_seconds * config.target_rate))
    stride_points = int(round(config.stride_seconds * config.target_rate))
    if window_points < 200:
        raise ValueError("BIOT windows must contain at least the 200-point FFT window")
    if stride_points <= 0:
        raise ValueError("stride_seconds must be positive")
    starts = list(range(0, montage.shape[1] - window_points + 1, stride_points))
    if not starts:
        raise ValueError(f"Recording is shorter than one window: {set_path}")
    windows = np.stack(
        [montage[:, start : start + window_points] for start in starts],
        axis=0,
    ).astype(np.float32, copy=False)

    scales = np.quantile(np.abs(windows), config.quantile, axis=-1, keepdims=True)
    valid = np.isfinite(windows).all(axis=(1, 2)) & (scales[..., 0] > 1e-8).all(axis=1)
    rejected_windows = int((~valid).sum())
    windows = windows[valid]
    scales = scales[valid]
    if windows.shape[0] == 0:
        raise ValueError(f"All windows were rejected as invalid: {set_path}")
    windows = (windows / (scales + 1e-8)).astype(np.float32, copy=False)

    metadata = {
        "source_path": str(Path(set_path).resolve()),
        "source_rate": source_rate,
        "source_channels": channel_names,
        "biot_channels": list(BIOT_CHANNELS),
        "target_rate": config.target_rate,
        "window_seconds": config.window_seconds,
        "stride_seconds": config.stride_seconds,
        "quantile": config.quantile,
        "shape": list(windows.shape),
        "rejected_windows": rejected_windows,
    }
    return windows, metadata


def _cache_is_current(
    record: SubjectRecord,
    cache_path: Path,
    metadata_path: Path,
    config: PreprocessingConfig,
) -> tuple[bool, int]:
    if not cache_path.is_file() or not metadata_path.is_file():
        return False, 0
    try:
        metadata = json.loads(metadata_path.read_text())
        source_stat = record.set_path.stat()
        expected = {
            "format_version": CACHE_FORMAT_VERSION,
            "source_path": str(record.set_path.resolve()),
            "source_size": source_stat.st_size,
            "source_mtime_ns": source_stat.st_mtime_ns,
            "preprocessing": asdict(config),
        }
        if metadata.get("cache_key") != expected:
            return False, 0
        array = np.load(cache_path, mmap_mode="r")
        if array.ndim != 3 or array.shape[1:] != (
            len(BIOT_CHANNELS),
            int(round(config.target_rate * config.window_seconds)),
        ):
            return False, 0
        return True, int(array.shape[0])
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False, 0


def prepare_subject_cache(
    record: SubjectRecord,
    cache_dir: str | Path,
    config: PreprocessingConfig,
) -> CachedSubject:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{record.subject_id}.npy"
    metadata_path = cache_dir / f"{record.subject_id}.json"
    lock_path = cache_dir / f"{record.subject_id}.lock"

    with lock_path.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        current, n_windows = _cache_is_current(
            record, cache_path, metadata_path, config
        )
        if not current:
            windows, metadata = preprocess_recording(record.set_path, config)
            source_stat = record.set_path.stat()
            metadata["cache_key"] = {
                "format_version": CACHE_FORMAT_VERSION,
                "source_path": str(record.set_path.resolve()),
                "source_size": source_stat.st_size,
                "source_mtime_ns": source_stat.st_mtime_ns,
                "preprocessing": asdict(config),
            }
            temporary_array = cache_dir / f".{record.subject_id}-{os.getpid()}.tmp.npy"
            temporary_metadata = cache_dir / f".{record.subject_id}-{os.getpid()}.tmp.json"
            np.save(temporary_array, windows, allow_pickle=False)
            temporary_metadata.write_text(json.dumps(metadata, indent=2, sort_keys=True))
            os.replace(temporary_array, cache_path)
            os.replace(temporary_metadata, metadata_path)
            n_windows = int(windows.shape[0])
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    return CachedSubject(
        record.subject_id,
        record.label,
        record.group,
        cache_path,
        n_windows,
    )


def prepare_all_caches(
    records: list[SubjectRecord],
    cache_dir: str | Path,
    config: PreprocessingConfig,
    *,
    max_workers: int = 1,
) -> list[CachedSubject]:
    if max_workers <= 0:
        raise ValueError("max_workers must be positive")

    cached: list[CachedSubject | None] = [None] * len(records)
    if max_workers == 1:
        completed = ((position, prepare_subject_cache(record, cache_dir, config))
                     for position, record in enumerate(records))
        for done, (position, item) in enumerate(completed, start=1):
            cached[position] = item
            print(
                f"[{done:02d}/{len(records):02d}] {item.subject_id}: "
                f"group={item.group}, windows={item.n_windows}",
                flush=True,
            )
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(prepare_subject_cache, record, cache_dir, config): position
                for position, record in enumerate(records)
            }
            for done, future in enumerate(as_completed(futures), start=1):
                position = futures[future]
                item = future.result()
                cached[position] = item
                print(
                    f"[{done:02d}/{len(records):02d}] {item.subject_id}: "
                    f"group={item.group}, windows={item.n_windows}",
                    flush=True,
                )

    if any(item is None for item in cached):
        raise AssertionError("Cache preparation did not return every subject")
    return [item for item in cached if item is not None]


class SubjectWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor, str]]):
    """Window dataset with equal per-subject sampling during training."""

    def __init__(
        self,
        subjects: list[CachedSubject],
        *,
        windows_per_subject: int | None,
        seed: int,
    ) -> None:
        if not subjects:
            raise ValueError("SubjectWindowDataset requires at least one subject")
        self.subjects = subjects
        self.windows_per_subject = windows_per_subject
        self.seed = seed
        self.epoch = 0
        self._arrays: dict[str, np.ndarray] = {}
        self._selections: list[np.ndarray] = []
        self._cumulative: list[int] = []
        if windows_per_subject is None:
            total = 0
            for subject in subjects:
                total += subject.n_windows
                self._cumulative.append(total)
        elif windows_per_subject <= 0:
            raise ValueError("windows_per_subject must be positive or None")
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        if self.windows_per_subject is None:
            return
        rng = np.random.default_rng(self.seed + 1_000_003 * epoch)
        self._selections = []
        for subject in self.subjects:
            replace = subject.n_windows < self.windows_per_subject
            selected = rng.choice(
                subject.n_windows,
                size=self.windows_per_subject,
                replace=replace,
            )
            self._selections.append(np.asarray(selected, dtype=np.int64))

    def __len__(self) -> int:
        if self.windows_per_subject is not None:
            return len(self.subjects) * self.windows_per_subject
        return self._cumulative[-1]

    def _resolve_index(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        if self.windows_per_subject is not None:
            subject_index, slot = divmod(index, self.windows_per_subject)
            return subject_index, int(self._selections[subject_index][slot])
        subject_index = bisect.bisect_right(self._cumulative, index)
        previous = 0 if subject_index == 0 else self._cumulative[subject_index - 1]
        return subject_index, index - previous

    def _array(self, subject: CachedSubject) -> np.ndarray:
        if subject.subject_id not in self._arrays:
            self._arrays[subject.subject_id] = np.load(subject.cache_path, mmap_mode="r")
        return self._arrays[subject.subject_id]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        subject_index, window_index = self._resolve_index(index)
        subject = self.subjects[subject_index]
        # Copy prevents PyTorch warnings and keeps a worker from mutating the mmap.
        window = np.array(self._array(subject)[window_index], dtype=np.float32, copy=True)
        return (
            torch.from_numpy(window),
            torch.tensor(subject.label, dtype=torch.float32),
            subject.subject_id,
        )
