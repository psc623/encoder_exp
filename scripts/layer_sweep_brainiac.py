"""Test whether BrainIAC's AD/CN signal lives in an intermediate ViT block.

The frozen-extractor protocol reads only the final block (MONAI ViT returns
`(final_normed_output, [all 12 block outputs])` and the extractor keeps
element 0). Self-supervised backbones often specialise their last layers to
the pretext task, so this sweeps every block and reports how linearly
separable AD/CN is at each depth, using the same manifest split.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from monai.networks.nets import ViT

from encoderbench.preprocessing import volume_to_tensor

CHECKPOINT = "/net/projects2/litian-lab/scpan/model_weights/brainIAC/BrainIAC.ckpt"
MANIFEST = "/net/projects2/litian-lab/scpan/encoders/data/manifests/adni.csv"
OUT = Path("/net/projects2/litian-lab/scpan/encoders/artifacts/audits/ad/brainiac_layer_sweep.npz")


def build_model(device: torch.device) -> ViT:
    model = ViT(in_channels=1, img_size=(96, 96, 96), patch_size=(16, 16, 16),
                hidden_size=768, mlp_dim=3072, num_layers=12, num_heads=12,
                save_attn=False).to(device)
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    source = payload.get("state_dict", payload)
    weights = {k.removeprefix("backbone."): v for k, v in source.items() if k.startswith("backbone.")}
    missing, unexpected = model.load_state_dict(weights, strict=False)
    meaningful = [k for k in missing if not k.startswith("classification_head")]
    if meaningful or unexpected:
        raise RuntimeError(f"key mismatch: missing={meaningful} unexpected={unexpected}")
    model.eval().requires_grad_(False)
    return model


@torch.inference_mode()
def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)
    model = build_model(device)

    with open(MANIFEST, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    print(f"volumes: {len(rows)}", flush=True)

    per_layer: list[list[np.ndarray]] = [[] for _ in range(13)]  # 12 blocks + final norm
    for index, row in enumerate(rows):
        volume = volume_to_tensor(row["path"], (96, 96, 96)).unsqueeze(0).to(device)
        final, hidden = model(volume)
        # hidden is the list of all block outputs; `final` is hidden[-1] after model.norm
        for layer, tokens in enumerate(list(hidden) + [final]):
            patch = tokens[:, 1:] if tokens.shape[1] == 217 else tokens
            # mean-pool over the 216 spatial patch tokens: enough to test separability
            per_layer[layer].append(patch.float().mean(dim=1).squeeze(0).cpu().numpy())
        if index % 50 == 0:
            print(f"  {index}/{len(rows)}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT,
        features=np.stack([np.stack(layer) for layer in per_layer]),  # [13, N, 768]
        labels=np.asarray([r["group"] for r in rows]),
        splits=np.asarray([r["split"] for r in rows]),
        subject_ids=np.asarray([r["subject_id"] for r in rows]),
        metadata=np.asarray(json.dumps({"layers": "0-11 = ViT blocks, 12 = final model.norm output",
                                        "pooling": "mean over 216 spatial patch tokens"})),
    )
    print(f"saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
