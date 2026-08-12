#!/usr/bin/env python
"""Build the native (unpooled) MASS/bsnip2 feature cache: every one of the
16x16x16=4096 layer-3 spatial locations as its own token (width 256), instead
of the pooled_grid=[4,4,4]=64-token cache the rest of encoderbench uses.
Real GPU forward passes over all 509 volumes -- this is the actual compute,
unlike mode1/mode2/mode3-6 which just read whichever cache they're pointed at.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from encoderbench.cache import FeatureCache, save_cache
from encoderbench.config import load_config
from encoderbench.extractors import build_extractor
from encoderbench.manifest import read_manifest

MANIFEST = "/net/projects2/litian-lab/scpan/encoders/data/manifests/bsnip2_mass.csv"
OUT = "/net/projects2/litian-lab/scpan/encoders/artifacts/cache/bsnip2/mass_native.npz"


def main() -> None:
    config = load_config()
    rows = read_manifest(MANIFEST)
    extractor = build_extractor("mass", config.raw, "cuda", native_tokens=True)

    t0 = time.time()
    features = []
    for index, row in enumerate(rows):
        tokens = extractor.extract(row["path"])  # [n_tokens, width], CPU
        features.append(tokens.numpy().astype(np.float16))
        if (index + 1) % 50 == 0 or index + 1 == len(rows):
            print(f"[{time.time()-t0:7.1f}s] {index+1}/{len(rows)}  shape={tuple(tokens.shape)}", flush=True)

    array = np.stack(features, axis=0)
    print(f"final cache array shape: {array.shape}, dtype {array.dtype}, "
         f"~{array.nbytes/1e9:.2f} GB", flush=True)
    cache = FeatureCache(
        features=array,
        file_ids=np.asarray([row["file_id"] for row in rows]),
        subject_ids=np.asarray([row["subject_id"] for row in rows]),
        labels=np.asarray([row["group"] for row in rows]),
        splits=np.asarray([row["split"] for row in rows]),
        metadata={"disease": "bsnip2", "encoder": "mass", "native_tokens": True,
                 "layer": int(config.raw["features"]["layers"]["mass"]),
                 "native_grid": list(extractor.last_native_grid),
                 "token_shape": list(array.shape[1:])},
    )
    out_path = save_cache(OUT, cache)
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
