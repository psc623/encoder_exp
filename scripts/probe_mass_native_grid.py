#!/usr/bin/env python
"""One-off: print MASS's native (pre-pooling) spatial grid at each U-Net stage,
to size the no-pooling extraction change."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from encoderbench.config import load_config
from encoderbench.extractors import build_extractor
from encoderbench.manifest import read_manifest

config = load_config()
rows = read_manifest("/net/projects2/litian-lab/scpan/encoders/data/manifests/bsnip2_mass.csv")
extractor = build_extractor("mass", config.raw, "cuda")
import torch
from encoderbench.preprocessing import volume_to_tensor

volume = volume_to_tensor(rows[0]["path"], extractor.shape, extractor.intensity).unsqueeze(0).to(extractor.device)
with torch.inference_mode():
    stages = extractor.model.encoder(volume)
for index, stage in enumerate(reversed(stages)):  # shallow to deep, matches _select's ordering
    print(f"shallow-to-deep index {index}: shape={tuple(stage.shape)}  "
         f"native_tokens={stage.shape[2]*stage.shape[3]*stage.shape[4]}")
