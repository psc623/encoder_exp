import numpy as np
import pytest

torch = pytest.importorskip("torch")
nib = pytest.importorskip("nibabel")

from encoderbench.extractors import SynthSegExtractor, build_extractor


def _write_posterior(root, stem, shape=(10, 11, 9, 5)):
    post_dir = root / "post"
    post_dir.mkdir(parents=True, exist_ok=True)
    data = np.random.default_rng(0).random(shape).astype(np.float32)
    nib.save(nib.Nifti1Image(data, affine=np.eye(4)), post_dir / f"{stem}_synthseg_post.nii.gz")
    return data.shape[-1]


def test_synthseg_extractor_pools_posterior_to_grid(tmp_path):
    width = _write_posterior(tmp_path, "0002_ss_box")
    data = {"volume_shapes": {"synthseg": [8, 8, 8]}}
    extractor = SynthSegExtractor(tmp_path, "cpu", data, (4, 4, 4))
    tokens = extractor.extract(tmp_path / "0002_ss_box.nii.gz")
    assert tokens.shape == (64, width)
    assert torch.isfinite(tokens).all()
    assert extractor.last_native_grid == (8, 8, 8)


def test_synthseg_extractor_rejects_missing_posterior(tmp_path):
    # Width inference needs at least one real posterior on disk; this one is a
    # different stem from the one requested below, which must still 404.
    _write_posterior(tmp_path, "0002_ss_box")
    data = {"volume_shapes": {"synthseg": [8, 8, 8]}}
    extractor = SynthSegExtractor(tmp_path, "cpu", data, (4, 4, 4))
    with pytest.raises(FileNotFoundError):
        extractor.extract(tmp_path / "missing_ss_box.nii.gz")


def test_synthseg_extractor_adapter_is_identity_before_training(tmp_path):
    """The post-pooling adapter's second layer is zero-initialized, so an
    untrained SynthSegExtractor (as used by `cache`/`probe`) must produce
    exactly the same tokens as plain pooling -- frozen SynthSeg numbers
    already reported must not shift just because finetune became possible."""
    width = _write_posterior(tmp_path, "0002_ss_box")
    data = {"volume_shapes": {"synthseg": [8, 8, 8]}}
    extractor = SynthSegExtractor(tmp_path, "cpu", data, (4, 4, 4), adapter_hidden_size=16)
    tokens = extractor.extract(tmp_path / "0002_ss_box.nii.gz")
    posterior_path = extractor._posterior_path(tmp_path / "0002_ss_box.nii.gz")
    from encoderbench.preprocessing import load_posterior_tensor

    posterior = load_posterior_tensor(posterior_path, (8, 8, 8))
    plain_pooled = extractor._pool(posterior.unsqueeze(0)).squeeze(0)
    assert torch.allclose(tokens, plain_pooled, atol=1e-6)
    assert tokens.shape == (64, width)


def test_synthseg_extractor_finetune_groups_are_trainable(tmp_path):
    from encoderbench.finetune import configure_parameter_budget

    _write_posterior(tmp_path, "0002_ss_box")
    data = {"volume_shapes": {"synthseg": [8, 8, 8]}}
    extractor = SynthSegExtractor(tmp_path, "cpu", data, (4, 4, 4), adapter_hidden_size=16)
    total_params = sum(p.numel() for p in extractor.model.parameters())
    audit = configure_parameter_budget(extractor, budget=total_params)
    assert audit["trainable_encoder_parameters"] == total_params
    assert all(p.requires_grad for p in extractor.model.parameters())


def test_dead_channels_do_not_explode_normalization():
    """AnatCL's ResNet18 layer4 is post-ReLU and has channels that are identical
    in every volume; the old token-mean statistics gave them std exactly 0."""
    from encoderbench.training import token_normalization

    rng = np.random.default_rng(0)
    features = rng.normal(size=(20, 64, 8)).astype(np.float32)
    features[:, :, 3] = np.linspace(-1, 1, 64)[None, :]  # same in every volume
    mean, std = token_normalization(features)
    assert std.min() > 0.0
    normalized = (torch.from_numpy(features) - mean) / std
    assert normalized.abs().max() < 100.0


def test_build_extractor_dispatches_synthseg(tmp_path):
    _write_posterior(tmp_path, "0002_ss_box")
    config = {
        "checkpoints": {"synthseg": str(tmp_path)},
        "source_repositories": {},
        "data": {"volume_shapes": {"synthseg": [8, 8, 8]}},
        "features": {"pooled_grid": [4, 4, 4]},
    }
    extractor = build_extractor("synthseg", config, "cpu")
    assert isinstance(extractor, SynthSegExtractor)
