import numpy as np

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

