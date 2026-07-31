import numpy as np
import pytest

from encoderbench.preprocessing import normalize_volume, robust_bounds, slice_indices


def test_normalization_uses_positive_voxels_and_clips():
    volume = np.zeros((3, 3, 3), dtype=np.float32)
    volume.ravel()[1:] = np.arange(1, 27)
    normalized = normalize_volume(volume)
    assert normalized.dtype == np.float32
    assert normalized.min() == 0.0
    assert normalized.max() == 1.0
    assert np.isfinite(normalized).all()


def test_constant_volume_is_safe():
    volume = np.full((2, 2, 2), 7.0, dtype=np.float32)
    lo, hi = robust_bounds(volume)
    assert (lo, hi) == (7.0, 8.0)
    assert np.all(normalize_volume(volume) == 0)


def test_slice_sampling_has_endpoints_and_24_unique_indices():
    indices = slice_indices(100, 24, (0.15, 0.85))
    assert indices[0] == 15
    assert indices[-1] == 84
    assert len(indices) == len(set(indices)) == 24


def test_each_encoder_gets_its_own_intensity_recipe():
    """A frozen encoder cannot adapt to a distribution it was not trained on, so
    BrainIAC/MASS must receive z-scored input rather than the [0,1] rescaling."""
    from encoderbench.extractors import (AnatCLExtractor, BrainIACExtractor, MASSExtractor,
                                         MedSigLIPExtractor)
    from encoderbench.preprocessing import INTENSITY_MODES

    assert BrainIACExtractor.intensity == "zscore_nonzero"
    assert MASSExtractor.intensity == "clip_zscore"
    assert AnatCLExtractor.intensity == "minmax_percentile"
    assert MedSigLIPExtractor.intensity == "minmax_percentile"  # unused; it has its own processor

    rng = np.random.default_rng(0)
    volume = np.abs(rng.normal(size=(12, 12, 12))).astype(np.float32) * 300.0
    volume[:2] = 0.0  # background
    for mode in ("zscore_nonzero", "clip_zscore"):
        out = INTENSITY_MODES[mode](volume)
        foreground = out[volume > 0]
        assert abs(float(foreground.mean())) < 0.05
        assert abs(float(foreground.std()) - 1.0) < 0.15


def test_load_posterior_tensor_resamples_without_intensity_normalization(tmp_path):
    nib = pytest.importorskip("nibabel")
    from encoderbench.preprocessing import load_posterior_tensor

    data = np.random.default_rng(0).random((10, 11, 9, 5)).astype(np.float32)
    source = tmp_path / "posterior.nii.gz"
    nib.save(nib.Nifti1Image(data, affine=np.eye(4)), source)
    tensor = load_posterior_tensor(source, (8, 8, 8))
    assert tensor.shape == (5, 8, 8, 8)
    assert float(tensor.min()) >= 0.0
    assert float(tensor.max()) <= 1.0

