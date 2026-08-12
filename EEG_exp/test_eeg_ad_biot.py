"""Fast unit and integration checks for the EEG_AD BIOT experiment."""

from pathlib import Path

import numpy as np
import torch

from biot_model import BIOTEncoder, configure_encoder_trainability, load_pretrained_encoder
from eeg_ad_data import (
    BIOT_CHANNELS,
    PreprocessingConfig,
    build_biot_bipolar_montage,
    load_ac_subjects,
    preprocess_recording,
)
from train_biot_ac import build_splits, select_balanced_accuracy_threshold


ROOT = Path("/net/projects2/litian-lab/scpan")
DATASET = ROOT / "dataset/EEG_AD"
CHECKPOINT = ROOT / "github_repo/BIOT/pretrained-models/EEG-six-datasets-18-channels.ckpt"


def test_bipolar_montage_order() -> None:
    names = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz"]
    data = np.arange(len(names), dtype=np.float32)[:, None]
    montage = build_biot_bipolar_montage(data, names)
    lookup = {name.upper(): index for index, name in enumerate(names)}
    expected_first = lookup["FP1"] - lookup["F7"]
    expected_last = lookup["P4"] - lookup["O2"]
    assert montage.shape == (len(BIOT_CHANNELS), 1)
    assert montage[0, 0] == expected_first
    assert montage[-1, 0] == expected_last


def test_f_group_is_excluded_and_splits_do_not_leak() -> None:
    records = load_ac_subjects(DATASET)
    assert len(records) == 65
    assert sum(record.group == "A" for record in records) == 36
    assert sum(record.group == "C" for record in records) == 29
    assert all(record.group != "F" for record in records)

    # build_splits only needs CachedSubject-like objects with subject_id/label.
    splits = build_splits(records, n_splits=5, validation_fraction=0.2, seed=2026)
    test_indices: list[int] = []
    for train, validation, test in splits:
        assert not (set(train) & set(validation))
        assert not (set(train) & set(test))
        assert not (set(validation) & set(test))
        test_indices.extend(test)
    assert sorted(test_indices) == list(range(65))


def test_threshold_selection() -> None:
    labels = np.asarray([0, 0, 1, 1])
    probabilities = np.asarray([0.1, 0.4, 0.6, 0.9])
    threshold = select_balanced_accuracy_threshold(labels, probabilities)
    assert 0.4 < threshold <= 0.6


def test_invalid_preprocessing_configuration_is_rejected() -> None:
    try:
        PreprocessingConfig(target_rate=0)
    except ValueError as error:
        assert "target_rate" in str(error)
    else:
        raise AssertionError("Invalid target rate was accepted")


def test_encoder_training_regimes_select_expected_parameters() -> None:
    encoder = BIOTEncoder(n_channels=18)
    configure_encoder_trainability(encoder, "pretrained_frozen")
    assert not any(parameter.requires_grad for parameter in encoder.parameters())

    configure_encoder_trainability(encoder, "pretrained_last2")
    trainable = {
        name for name, parameter in encoder.named_parameters() if parameter.requires_grad
    }
    assert trainable
    assert all(name.startswith("transformer.layers.layers.2") or
               name.startswith("transformer.layers.layers.3") for name in trainable)

    configure_encoder_trainability(encoder, "pretrained_full")
    assert not encoder.index.requires_grad
    assert all(
        parameter.requires_grad
        for name, parameter in encoder.named_parameters()
        if name != "index"
    )


def test_real_subject_preprocessing() -> None:
    path = DATASET / "derivatives/sub-001/eeg/sub-001_task-eyesclosed_eeg.set"
    windows, metadata = preprocess_recording(path, PreprocessingConfig())
    assert windows.ndim == 3
    assert windows.shape[1:] == (16, 2000)
    assert windows.dtype == np.float32
    assert np.isfinite(windows).all()
    assert metadata["source_rate"] == 500.0


def test_released_checkpoint_strict_load_and_forward() -> None:
    encoder = load_pretrained_encoder(CHECKPOINT)
    assert encoder.channel_tokens.num_embeddings == 18
    encoder.eval()
    sample = torch.randn(1, 16, 2000)
    with torch.no_grad():
        embedding = encoder(sample)
    assert embedding.shape == (1, 256)
    assert torch.isfinite(embedding).all()
