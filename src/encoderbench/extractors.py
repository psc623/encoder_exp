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
from encoderbench.preprocessing import volume_to_slices, volume_to_tensor
from encoderbench.utils import checkpoint_identifier


class FrozenExtractor(ABC):
    name: str

    def __init__(self, device: str, pooled_grid: Sequence[int] = (4, 4, 4)):
        self.device = torch.device(device)
        self.pooled_grid = tuple(int(value) for value in pooled_grid)
        self.model: torch.nn.Module
        self.last_native_grid: tuple[int, int, int] | None = None
        self.load_audit: dict[str, Any] = {}

    def freeze(self) -> None:
        self.model.eval()
        self.model.requires_grad_(False)
        trainable = sum(parameter.numel() for parameter in self.model.parameters()
                        if parameter.requires_grad)
        if trainable != 0:
            raise RuntimeError(f"{self.name} has {trainable} trainable encoder parameters")

    @abstractmethod
    def extract(self, path: str | Path) -> torch.Tensor:
        """Return one CPU tensor shaped `[64,D]`."""

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
        return tokens.squeeze(0).cpu()

    def audit(self, tokens: torch.Tensor) -> dict[str, Any]:
        return {
            "encoder": self.name,
            "native_grid": self.last_native_grid,
            "native_token_count": None if self.last_native_grid is None else
                int(self.last_native_grid[0] * self.last_native_grid[1] * self.last_native_grid[2]),
            "pooled_grid": self.pooled_grid,
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
                 pooled_grid: Sequence[int]):
        super().__init__(device, pooled_grid)
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

    @torch.inference_mode()
    def extract(self, path: str | Path) -> torch.Tensor:
        images = volume_to_slices(path, axis=self.data["axis"], count=self.data["num_slices"],
                                  bounds=self.data["slice_range"],
                                  roi_fraction=self.data["roi_fraction"])
        if len(images) != self.data["num_slices"]:
            raise ValueError(f"MedSigLIP requires exactly {self.data['num_slices']} slices")
        inputs = self.processor(images=images, return_tensors="pt")
        pixels = inputs["pixel_values"].to(self.device)
        output = self.model.vision_model(pixel_values=pixels, return_dict=True)
        tokens = output.last_hidden_state
        expected = self.patch_grid * self.patch_grid
        if tokens.shape[1] != expected:
            raise ValueError(f"MedSigLIP expected {expected} patch tokens without CLS, got {tokens.shape[1]}")
        feature_map = tokens.reshape(len(images), self.patch_grid, self.patch_grid, -1)
        feature_map = feature_map.permute(3, 0, 1, 2).unsqueeze(0)
        return self._pool(feature_map)


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
                 data: dict[str, Any], pooled_grid: Sequence[int]):
        super().__init__(device, pooled_grid)
        root = Path(checkpoint).resolve()
        architecture = _load_module("encoderbench_braingemma_architecture",
                                    Path(repository).resolve() / "braingemma3d_architecture.py")
        self.model = architecture.MedSigLIP3D(str(root / "vision_model"), depth=2).to(self.device)
        self.shape = tuple(data["volume_shapes"]["braingemma3d"])
        self.load_audit = {"checkpoint": checkpoint_identifier(root / "vision_model"),
                           "inflation_depth": 2, "missing_keys": [], "unexpected_keys": []}
        self.freeze()

    @torch.inference_mode()
    def extract(self, path: str | Path) -> torch.Tensor:
        volume = volume_to_tensor(path, self.shape).unsqueeze(0).to(self.device)
        dtype = next(self.model.parameters()).dtype
        patch_module = self.model.vision_model.patch_embedding_3d
        patch_grid = patch_module(volume.to(dtype=dtype)).shape[2:]
        tokens = self.model.encode_image(volume.to(dtype=dtype))
        if tokens.shape[1] != int(torch.tensor(patch_grid).prod().item()):
            raise ValueError("BrainGemma3D patch count does not match its inflated 3D grid")
        feature_map = tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], *patch_grid)
        return self._pool(feature_map)


class MASSExtractor(FrozenExtractor):
    name = "mass"

    def __init__(self, checkpoint: str | Path, repository: str | Path, device: str,
                 data: dict[str, Any], pooled_grid: Sequence[int]):
        super().__init__(device, pooled_grid)
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

    @torch.inference_mode()
    def extract(self, path: str | Path) -> torch.Tensor:
        volume = volume_to_tensor(path, self.shape).unsqueeze(0).to(self.device)
        deepest = self.model.encoder(volume)[0]
        if deepest.shape[1] != 512:
            raise ValueError(f"MASS deepest encoder width must be 512, got {deepest.shape[1]}")
        return self._pool(deepest)


class BrainIACExtractor(FrozenExtractor):
    name = "brainiac"

    def __init__(self, checkpoint: str | Path, device: str, data: dict[str, Any],
                 pooled_grid: Sequence[int]):
        super().__init__(device, pooled_grid)
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

    @torch.inference_mode()
    def extract(self, path: str | Path) -> torch.Tensor:
        volume = volume_to_tensor(path, self.shape).unsqueeze(0).to(self.device)
        output = self.model(volume)
        tokens = output[0] if isinstance(output, tuple) else output
        # Released BrainIAC constructs MONAI ViT with classification=False,
        # which emits patches only; tolerate a CLS-bearing compatible release.
        self.load_audit["cls_token_present"] = tokens.shape[1] == 217
        patch_tokens = tokens[:, 1:] if tokens.shape[1] == 217 else tokens
        if patch_tokens.shape[1] != 216:
            raise ValueError(f"BrainIAC expected 216 spatial patch tokens, got {patch_tokens.shape[1]}")
        feature_map = patch_tokens.transpose(1, 2).reshape(tokens.shape[0], 768, 6, 6, 6)
        return self._pool(feature_map)


def build_extractor(name: str, config: dict[str, Any], device: str) -> FrozenExtractor:
    checkpoints, repositories = config["checkpoints"], config["source_repositories"]
    data, grid = config["data"], config["features"]["pooled_grid"]
    if name == "medsiglip":
        return MedSigLIPExtractor(checkpoints[name], device, data, grid)
    if name == "braingemma3d":
        return BrainGemma3DExtractor(checkpoints[name], repositories[name], device, data, grid)
    if name == "mass":
        return MASSExtractor(checkpoints[name], repositories[name], device, data, grid)
    if name == "brainiac":
        return BrainIACExtractor(checkpoints[name], device, data, grid)
    raise ValueError(f"Unknown encoder {name!r}")
