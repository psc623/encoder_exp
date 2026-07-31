"""Frozen spatial-token extractors for the four registered encoders."""

from __future__ import annotations

import importlib.util
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as functional

from encoderbench.models import fixed_3d_position_encoding
from encoderbench.preprocessing import load_posterior_tensor, volume_to_slices, volume_to_tensor
from encoderbench.utils import checkpoint_identifier


class FrozenExtractor(ABC):
    name: str

    #: Intensity recipe from this encoder's own repository (see INTENSITY_MODES).
    intensity: str = "minmax_percentile"

    def __init__(self, device: str, pooled_grid: Sequence[int] = (4, 4, 4),
                 layer: int = -1):
        self.device = torch.device(device)
        self.pooled_grid = tuple(int(value) for value in pooled_grid)
        # Depth to read features from, indexing an encoder-specific list of taps
        # ordered shallow to deep. -1 keeps the original protocol (the deepest
        # output, after any final norm). Self-supervised backbones often
        # specialise their last blocks to the pretext task, so the most linearly
        # probe-able representation is not always the final one.
        self.layer = int(layer)
        self.model: torch.nn.Module
        self.last_native_grid: tuple[int, int, int] | None = None
        self.load_audit: dict[str, Any] = {}

    def _stage_groups(self, named_stages: Sequence[tuple[str, Any]],
                      deepest_extra: Sequence[tuple[str, Any]] = ()):
        """Build fine-tune groups that stop at the configured tap.

        `named_stages` is ordered shallow to deep and aligned with this encoder's
        taps. Anything past the tap does not contribute to the loss, so offering
        it for unfreezing would both waste the parameter budget and leave those
        parameters without gradients, which the fine-tune gradient audit rejects.
        """
        stages = list(named_stages)
        if self.layer == -1:
            return [(name, module) for name, module in reversed(stages)], list(deepest_extra)
        if not 0 <= self.layer < len(stages):
            raise ValueError(f"{self.name}: layer {self.layer} outside 0..{len(stages) - 1}")
        used = stages[: self.layer + 1]
        return [(name, module) for name, module in reversed(used)], []

    def _select(self, taps: Sequence[Any], deepest: Any) -> Any:
        """Return the requested tap; -1 selects the protocol's original output."""
        if self.layer == -1:
            return deepest
        if not 0 <= self.layer < len(taps):
            raise ValueError(f"{self.name}: layer {self.layer} outside 0..{len(taps) - 1} "
                             f"(or -1 for the deepest output)")
        return taps[self.layer]

    def freeze(self) -> None:
        self.model.eval()
        self.model.requires_grad_(False)
        trainable = sum(parameter.numel() for parameter in self.model.parameters()
                        if parameter.requires_grad)
        if trainable != 0:
            raise RuntimeError(f"{self.name} has {trainable} trainable encoder parameters")

    @abstractmethod
    def forward_tokens(self, path: str | Path) -> torch.Tensor:
        """Return a differentiable device tensor shaped `[1,64,D]`."""

    @abstractmethod
    def finetune_groups(self) -> tuple[list[tuple[str, torch.nn.Module]],
                                       list[tuple[str, torch.nn.Module]]]:
        """Return output-to-input primary groups and always-train output modules."""

    @torch.inference_mode()
    def extract(self, path: str | Path) -> torch.Tensor:
        """Return one frozen CPU tensor shaped `[64,D]`."""
        return self.forward_tokens(path).squeeze(0).cpu()

    @torch.inference_mode()
    def extract_at(self, path: str | Path, layer: int) -> torch.Tensor:
        """Return `[64,D]` read at `layer`, leaving the configured depth intact."""
        original = self.layer
        self.layer = int(layer)
        try:
            return self.forward_tokens(path).squeeze(0).cpu()
        finally:
            self.layer = original

    def _pool(self, feature_map: torch.Tensor) -> torch.Tensor:
        if feature_map.ndim != 5:
            raise ValueError(f"Expected [B,C,D,H,W], got {tuple(feature_map.shape)}")
        self.last_native_grid = tuple(int(value) for value in feature_map.shape[2:])
        native_count = feature_map.shape[2] * feature_map.shape[3] * feature_map.shape[4]
        flattened = feature_map.flatten(2).transpose(1, 2)
        if flattened.shape[1] != native_count:
            raise RuntimeError("Native feature grid did not flatten losslessly")
        round_trip = flattened.transpose(1, 2).reshape_as(feature_map)
        if not torch.equal(feature_map, round_trip):
            raise RuntimeError("Feature flatten/reshape changed token order")
        pooled = functional.adaptive_avg_pool3d(feature_map.float(), self.pooled_grid)
        tokens = pooled.flatten(2).transpose(1, 2)
        expected = self.pooled_grid[0] * self.pooled_grid[1] * self.pooled_grid[2]
        if tokens.shape[1] != expected:
            raise RuntimeError(f"Expected {expected} pooled tokens, got {tokens.shape[1]}")
        positions = fixed_3d_position_encoding(self.pooled_grid, tokens.shape[-1]).to(tokens.device)
        tokens = tokens + positions.unsqueeze(0)
        if not torch.isfinite(tokens).all():
            raise FloatingPointError(f"{self.name} produced NaN or Inf")
        return tokens

    def audit(self, tokens: torch.Tensor) -> dict[str, Any]:
        return {
            "encoder": self.name,
            "native_grid": self.last_native_grid,
            "native_token_count": None if self.last_native_grid is None else
                int(self.last_native_grid[0] * self.last_native_grid[1] * self.last_native_grid[2]),
            "pooled_grid": self.pooled_grid,
            "layer": self.layer,
            "pooled_shape": list(tokens.shape),
            "dtype": str(tokens.dtype),
            "finite": bool(torch.isfinite(tokens).all()),
            "eval_mode": not self.model.training,
            "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            "load": self.load_audit,
        }


class MedSigLIPExtractor(FrozenExtractor):
    name = "medsiglip"

    def __init__(self, checkpoint: str | Path, device: str, data: dict[str, Any],
                 pooled_grid: Sequence[int], layer: int = -1):
        super().__init__(device, pooled_grid, layer)
        from transformers import AutoImageProcessor, AutoModel

        self.checkpoint = Path(checkpoint).resolve()
        self.processor = AutoImageProcessor.from_pretrained(self.checkpoint, local_files_only=True)
        self.model = AutoModel.from_pretrained(self.checkpoint, local_files_only=True).to(self.device)
        self.data = data
        vision_config = self.model.config.vision_config
        self.patch_grid = int(vision_config.image_size) // int(vision_config.patch_size)
        self.load_audit = {"checkpoint": checkpoint_identifier(self.checkpoint),
                           "missing_keys": [], "unexpected_keys": []}
        self.freeze()

    def forward_tokens(self, path: str | Path) -> torch.Tensor:
        images = volume_to_slices(path, axis=self.data["axis"], count=self.data["num_slices"],
                                  bounds=self.data["slice_range"],
                                  roi_fraction=self.data["roi_fraction"])
        if len(images) != self.data["num_slices"]:
            raise ValueError(f"MedSigLIP requires exactly {self.data['num_slices']} slices")
        inputs = self.processor(images=images, return_tensors="pt")
        pixels = inputs["pixel_values"].to(self.device)
        vision = self.model.vision_model
        if self.layer == -1:
            tokens = vision(pixel_values=pixels, return_dict=True).last_hidden_state
        else:
            # The deepest tap is post_layernorm'd by the module's own forward;
            # intermediate taps are the raw block outputs.
            embedded = vision.embeddings(pixels, interpolate_pos_encoding=False)
            tokens = self._select(
                _siglip_layer_outputs(vision.encoder, embedded, stop_after=self.layer), None)
        expected = self.patch_grid * self.patch_grid
        if tokens.shape[1] != expected:
            raise ValueError(f"MedSigLIP expected {expected} patch tokens without CLS, got {tokens.shape[1]}")
        feature_map = tokens.reshape(len(images), self.patch_grid, self.patch_grid, -1)
        feature_map = feature_map.permute(3, 0, 1, 2).unsqueeze(0)
        return self._pool(feature_map)

    def finetune_groups(self):
        vision = self.model.vision_model
        stages = [(f"vision.encoder.layers.{index}", layer)
                  for index, layer in enumerate(vision.encoder.layers)]
        return self._stage_groups(stages, [("vision.post_layernorm", vision.post_layernorm)])


def _siglip_layer_outputs(encoder: torch.nn.Module, embedded: torch.Tensor,
                          stop_after: int | None = None) -> list[torch.Tensor]:
    """Run a `SiglipEncoder` layer by layer and keep every intermediate output.

    `SiglipEncoder.forward` in transformers 4.57 absorbs `output_hidden_states`
    into `**kwargs` and returns only `last_hidden_state`, so asking it for
    hidden states silently yields None. Walking `encoder.layers` directly is the
    supported way to tap intermediate depths.

    `stop_after` ends the walk once the requested depth is reached. That matters
    for fine-tuning: layers past the tap contribute nothing to the loss, so
    running them would waste compute and leave them without gradients.
    """
    hidden = embedded
    outputs = []
    for index, layer in enumerate(encoder.layers):
        hidden = layer(hidden, None)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        outputs.append(hidden)
        if stop_after is not None and index >= stop_after:
            break
    return outputs


def _load_module(name: str, source: Path, extra_path: Path | None = None):
    if extra_path is not None and str(extra_path) not in sys.path:
        sys.path.insert(0, str(extra_path))
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class BrainGemma3DExtractor(FrozenExtractor):
    name = "braingemma3d"

    def __init__(self, checkpoint: str | Path, repository: str | Path, device: str,
                 data: dict[str, Any], pooled_grid: Sequence[int], layer: int = -1):
        super().__init__(device, pooled_grid, layer)
        root = Path(checkpoint).resolve()
        architecture = _load_module("encoderbench_braingemma_architecture",
                                    Path(repository).resolve() / "braingemma3d_architecture.py")
        self.model = architecture.MedSigLIP3D(str(root / "vision_model"), depth=2).to(self.device)
        self.shape = tuple(data["volume_shapes"]["braingemma3d"])
        self.load_audit = {"checkpoint": checkpoint_identifier(root / "vision_model"),
                           "inflation_depth": 2, "missing_keys": [], "unexpected_keys": []}
        self.freeze()

    def forward_tokens(self, path: str | Path) -> torch.Tensor:
        volume = volume_to_tensor(path, self.shape, self.intensity).unsqueeze(0).to(self.device)
        dtype = next(self.model.parameters()).dtype
        vision = self.model.vision_model
        patch_module = vision.patch_embedding_3d
        patch_grid = patch_module(volume.to(dtype=dtype)).shape[2:]
        if self.layer == -1:
            tokens = self.model.encode_image(volume.to(dtype=dtype))
        else:
            # Mirror SiglipVisionTransformer3D.forward, but keep every layer.
            embedded = patch_module(volume.to(dtype=dtype))
            _, _, depth, height, width = embedded.shape
            embedded = embedded.flatten(2).transpose(1, 2)
            positions = vision.get_position_embedding_3d(
                depth, height * width, height, width
            ).to(embedded.device, dtype=embedded.dtype)
            tokens = self._select(
                _siglip_layer_outputs(vision.encoder, embedded + positions, stop_after=self.layer),
                None)
        if tokens.shape[1] != int(torch.tensor(patch_grid).prod().item()):
            raise ValueError("BrainGemma3D patch count does not match its inflated 3D grid")
        feature_map = tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], *patch_grid)
        return self._pool(feature_map)

    def finetune_groups(self):
        vision = self.model.vision_model
        stages = [(f"vision.encoder.layers.{index}", layer)
                  for index, layer in enumerate(vision.encoder.layers)]
        return self._stage_groups(stages, [("vision.post_layernorm", vision.post_layernorm)])


class MASSExtractor(FrozenExtractor):
    name = "mass"
    intensity = "clip_zscore"

    def __init__(self, checkpoint: str | Path, repository: str | Path, device: str,
                 data: dict[str, Any], pooled_grid: Sequence[int], layer: int = -1):
        super().__init__(device, pooled_grid, layer)
        root = Path(repository).resolve()
        module = _load_module("encoderbench_mass_inference", root / "inference.py", root)
        self.model, model_config = module.load_model(Path(checkpoint), self.device, use_ema=True)
        self.shape = tuple(data["volume_shapes"]["mass"])
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if "ema_model" in payload:
            weight_key = "ema_model"
        elif "ema_state_dict" in payload:
            weight_key = "ema_state_dict"
        elif "model" in payload:
            weight_key = "model"
        elif "model_state_dict" in payload:
            weight_key = "model_state_dict"
        else:
            weight_key = "raw_state_dict"
        source = payload if weight_key == "raw_state_dict" else payload[weight_key]
        clean = {key.removeprefix("module."): value for key, value in source.items()}
        missing, unexpected = self.model.load_state_dict(clean, strict=False)
        self.load_audit = {"checkpoint": checkpoint_identifier(checkpoint), "weight_key": weight_key,
                           "model_config": model_config.get("model", {}),
                           "missing_keys": list(missing), "unexpected_keys": list(unexpected)}
        self.freeze()

    def forward_tokens(self, path: str | Path) -> torch.Tensor:
        volume = volume_to_tensor(path, self.shape, self.intensity).unsqueeze(0).to(self.device)
        # The U-Net encoder returns (x5, x4, x3, x2, x1), deep to shallow.
        stages = self.model.encoder(volume)
        deepest = stages[0]
        if deepest.shape[1] != 512:
            raise ValueError(f"MASS deepest encoder width must be 512, got {deepest.shape[1]}")
        return self._pool(self._select(tuple(reversed(stages)), deepest))

    def finetune_groups(self):
        encoder = self.model.encoder
        stages = [(f"encoder.{name}", getattr(encoder, name))
                  for name in ("inc", "down1", "down2", "down3", "down4")]
        return self._stage_groups(stages)


class BrainIACExtractor(FrozenExtractor):
    name = "brainiac"
    intensity = "zscore_nonzero"

    def __init__(self, checkpoint: str | Path, device: str, data: dict[str, Any],
                 pooled_grid: Sequence[int], layer: int = -1):
        super().__init__(device, pooled_grid, layer)
        from monai.networks.nets import ViT

        self.model = ViT(in_channels=1, img_size=(96, 96, 96), patch_size=(16, 16, 16),
                         hidden_size=768, mlp_dim=3072, num_layers=12, num_heads=12,
                         save_attn=True).to(self.device)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        source = payload.get("state_dict", payload)
        weights = {key.removeprefix("backbone."): value for key, value in source.items()
                   if key.startswith("backbone.")}
        if not weights:
            raise ValueError("BrainIAC checkpoint contains no 'backbone.' weights")
        missing, unexpected = self.model.load_state_dict(weights, strict=False)
        meaningful_missing = [key for key in missing if not key.startswith("classification_head")]
        if meaningful_missing or unexpected:
            raise RuntimeError(f"BrainIAC key mismatch: missing={meaningful_missing}, unexpected={unexpected}")
        self.shape = tuple(data["volume_shapes"]["brainiac"])
        self.load_audit = {"checkpoint": checkpoint_identifier(checkpoint),
                           "missing_keys": missing, "unexpected_keys": unexpected}
        self.freeze()

    def forward_tokens(self, path: str | Path) -> torch.Tensor:
        volume = volume_to_tensor(path, self.shape, self.intensity).unsqueeze(0).to(self.device)
        output = self.model(volume)
        # MONAI's ViT returns (final_normed_output, [every block's output]).
        if isinstance(output, tuple):
            tokens = self._select(tuple(output[1]), output[0])
        else:
            tokens = output
        # Released BrainIAC constructs MONAI ViT with classification=False,
        # which emits patches only; tolerate a CLS-bearing compatible release.
        self.load_audit["cls_token_present"] = tokens.shape[1] == 217
        patch_tokens = tokens[:, 1:] if tokens.shape[1] == 217 else tokens
        if patch_tokens.shape[1] != 216:
            raise ValueError(f"BrainIAC expected 216 spatial patch tokens, got {patch_tokens.shape[1]}")
        feature_map = patch_tokens.transpose(1, 2).reshape(tokens.shape[0], -1, 6, 6, 6)
        return self._pool(feature_map)

    def finetune_groups(self):
        stages = [(f"blocks.{index}", block) for index, block in enumerate(self.model.blocks)]
        return self._stage_groups(stages, [("norm", self.model.norm)])


def _register_anatcl_pickle_stub() -> None:
    """AnatCL release checkpoints bundle a sklearn age-debiasing estimator
    (models.estimators.AgeEstimator) alongside the backbone state dict. We
    only read the 'model' key, but torch.load unpickles the whole payload,
    so a minimal importable stub is required for that class to resolve."""
    if "models.estimators" in sys.modules:
        return
    import types

    models_pkg = sys.modules.get("models", types.ModuleType("models"))
    estimators_mod = types.ModuleType("models.estimators")

    class AgeEstimator:
        pass

    estimators_mod.AgeEstimator = AgeEstimator
    models_pkg.estimators = estimators_mod
    sys.modules["models"] = models_pkg
    sys.modules["models.estimators"] = estimators_mod


class AnatCLExtractor(FrozenExtractor):
    name = "anatcl"

    def __init__(self, checkpoint: str | Path, repository: str | Path, device: str,
                 data: dict[str, Any], pooled_grid: Sequence[int], layer: int = -1):
        super().__init__(device, pooled_grid, layer)
        root = Path(repository).resolve()
        module = _load_module("encoderbench_anatcl_resnet3d", root / "anatcl" / "models" / "resnet3d.py")
        self.model = module.SupConResNet(name="resnet18", feat_dim=128, use_head=False).to(self.device)
        _register_anatcl_pickle_stub()
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        source = payload["model"] if "model" in payload else payload
        missing, unexpected = self.model.load_state_dict(source, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"AnatCL key mismatch: missing={missing}, unexpected={unexpected}")
        self.shape = tuple(data["volume_shapes"]["anatcl"])
        self.load_audit = {"checkpoint": checkpoint_identifier(checkpoint),
                           "missing_keys": missing, "unexpected_keys": unexpected,
                           "preprocessing_note": ("AnatCL was trained on CAT12/VBM gray-matter "
                                                  "density maps; this cache uses skull-stripped "
                                                  "T1 volumes resized to the same 121x128x121 "
                                                  "shape (resize-only adapter), which is a domain "
                                                  "shift from AnatCL's native training input.")}
        self.freeze()

    def forward_tokens(self, path: str | Path) -> torch.Tensor:
        volume = volume_to_tensor(path, self.shape, self.intensity).unsqueeze(0).to(self.device)
        dtype = next(self.model.parameters()).dtype
        encoder = self.model.encoder
        x = encoder.conv1(volume.to(dtype=dtype))
        x = encoder.maxpool(encoder.relu(encoder.bn1(x)))
        x1 = encoder.layer1(x)
        x2 = encoder.layer2(x1)
        x3 = encoder.layer3(x2)
        x4 = encoder.layer4(x3)
        if x4.shape[1] != 512:
            raise ValueError(f"AnatCL deepest encoder width must be 512, got {x4.shape[1]}")
        return self._pool(self._select((x1, x2, x3, x4), x4))

    def finetune_groups(self):
        encoder = self.model.encoder
        stages = [(f"encoder.{name}", getattr(encoder, name))
                  for name in ("layer1", "layer2", "layer3", "layer4")]
        return self._stage_groups(stages)


class SynthSegPosteriorAdapter(torch.nn.Module):
    """Optional trainable refinement of SynthSeg's frozen posterior tokens.

    SynthSeg itself is an external, frozen FreeSurfer tool with no differentiable
    parameters, so there is no real "encoder" to unfreeze the way MASS/BrainIAC/
    MedSigLIP/AnatCL do. What *can* be fine-tuned without pretending otherwise is
    a small per-token residual MLP applied after pooling, on top of the 33-class
    posterior probabilities: a learned nonlinear recombination of "how much
    hippocampus/ventricle/etc. is here", not a re-run of segmentation itself.

    The second linear layer is zero-initialized, so before any training this
    adapter is an exact identity function -- `cache`/`probe` (which never call
    `finetune_groups`) therefore see bit-identical tokens to the pre-adapter
    code, and every already-reported frozen SynthSeg number is unaffected.

    Deliberately small (tens of thousands of parameters, not the shared 8M
    finetune budget every other encoder uses): the input is a 33-dimensional
    per-voxel probability simplex, so a wide MLP over it has rank <= 33 no
    matter how large its hidden layer is -- padding it out to millions of
    parameters would not add expressiveness, only overfitting risk on ~235
    ADNI training volumes.
    """

    def __init__(self, width: int, hidden_size: int):
        super().__init__()
        self.norm = torch.nn.LayerNorm(width)
        self.fc1 = torch.nn.Linear(width, hidden_size)
        self.act = torch.nn.GELU()
        self.fc2 = torch.nn.Linear(hidden_size, width)
        torch.nn.init.zeros_(self.fc2.weight)
        torch.nn.init.zeros_(self.fc2.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens + self.fc2(self.act(self.fc1(self.norm(tokens))))


def _infer_posterior_width(posteriors_dir: Path) -> int:
    import nibabel as nib

    candidates = sorted(posteriors_dir.glob("*_synthseg_post.nii.gz"))
    if not candidates:
        raise FileNotFoundError(
            f"No SynthSeg posterior files found under {posteriors_dir} to infer channel width"
        )
    shape = nib.load(str(candidates[0])).shape
    if len(shape) != 4:
        raise ValueError(f"Expected a 4D posterior NIfTI, got shape {shape} from {candidates[0]}")
    return int(shape[-1])


class SynthSegExtractor(FrozenExtractor):
    """Wraps precomputed FreeSurfer `mri_synthseg --post` class-posterior volumes.

    SynthSeg is an external, frozen TensorFlow tool (not a PyTorch module we can
    forward through), so its 33-class anatomical posterior-probability volume is
    treated as the fixed per-voxel feature map: it is pooled with the same
    `_pool()` grid/position-encoding path as every other encoder. It has no
    differentiable segmentation parameters, but `finetune_groups()` exposes a
    small trainable post-pooling adapter (`SynthSegPosteriorAdapter`) so the
    `finetune` command is meaningful for it too -- see that class's docstring.
    """

    name = "synthseg"

    def __init__(self, checkpoint: str | Path, device: str, data: dict[str, Any],
                 pooled_grid: Sequence[int], adapter_hidden_size: int = 4096):
        super().__init__(device, pooled_grid)
        root = Path(checkpoint).resolve()
        self.posteriors_dir = root / "post"
        if not self.posteriors_dir.is_dir():
            raise FileNotFoundError(f"SynthSeg posterior directory not found: {self.posteriors_dir}")
        width = _infer_posterior_width(self.posteriors_dir)
        self.model = SynthSegPosteriorAdapter(width, adapter_hidden_size).to(self.device)
        self.shape = tuple(data["volume_shapes"]["synthseg"])
        self.load_audit = {"tool": "FreeSurfer mri_synthseg --robust --parc --post",
                           "posteriors_dir": str(self.posteriors_dir),
                           "missing_keys": [], "unexpected_keys": []}
        metadata_path = root / "run_metadata.json"
        if metadata_path.is_file():
            self.load_audit["run_metadata"] = checkpoint_identifier(metadata_path)
        self.freeze()

    def _posterior_path(self, path: str | Path) -> Path:
        stem = Path(path).name
        for suffix in (".nii.gz", ".nii"):
            if stem.endswith(suffix):
                stem = stem[:-len(suffix)]
                break
        candidate = self.posteriors_dir / f"{stem}_synthseg_post.nii.gz"
        if not candidate.is_file():
            raise FileNotFoundError(f"Missing SynthSeg posterior for {path}: {candidate}")
        return candidate

    def forward_tokens(self, path: str | Path) -> torch.Tensor:
        posterior = load_posterior_tensor(self._posterior_path(path), self.shape).to(self.device)
        return self.model(self._pool(posterior.unsqueeze(0)))

    def finetune_groups(self):
        return [("adapter", self.model)], []


def build_extractor(name: str, config: dict[str, Any], device: str,
                    layer: int | None = None) -> FrozenExtractor:
    checkpoints, repositories = config["checkpoints"], config["source_repositories"]
    data, grid = config["data"], config["features"]["pooled_grid"]
    if layer is None:
        layer = int(config["features"].get("layers", {}).get(name, -1))
    if name == "medsiglip":
        return MedSigLIPExtractor(checkpoints[name], device, data, grid, layer)
    if name == "braingemma3d":
        return BrainGemma3DExtractor(checkpoints[name], repositories[name], device, data, grid, layer)
    if name == "mass":
        return MASSExtractor(checkpoints[name], repositories[name], device, data, grid, layer)
    if name == "brainiac":
        return BrainIACExtractor(checkpoints[name], device, data, grid, layer)
    if name == "anatcl":
        return AnatCLExtractor(checkpoints[name], repositories[name], device, data, grid, layer)
    if name == "synthseg":
        # SynthSeg posteriors are a single fixed output; there is no depth to pick.
        if layer != -1:
            raise ValueError("synthseg has no intermediate layers; use layer -1")
        adapter_hidden_size = int(config.get("finetune", {}).get("synthseg_adapter_hidden_size", 4096))
        return SynthSegExtractor(checkpoints[name], device, data, grid, adapter_hidden_size)
    raise ValueError(f"Unknown encoder {name!r}")
