#!/usr/bin/env python
"""Build a de-pooled SynthSeg/bsnip2 feature cache.

SynthSeg's native pre-pool grid is 128^3 at 33 channels (2,097,152 tokens) --
flattening that for the exact-linear channel is a ~69M-dim vector per sample,
which is not tractable to fit in memory (the Gram-trick train matrix alone
would be ~60GB+ at float32). So this uses pooled_grid=(32,32,32) instead of
the frozen-config [4,4,4]: 32768 tokens x 33 channels = 1,081,344-dim flatten,
matched in order of magnitude to MASS's native 4096x256=1,048,576-dim flatten
from cache_mass_native_bsnip2.py, rather than [4,4,4]'s 64x33=2,112. Still a
>500x finer grid than the original protocol, at a computationally honest budget.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from encoderbench.cache import FeatureCache, save_cache
from encoderbench.config import load_config
from encoderbench.extractors import build_extractor
from encoderbench.manifest import read_manifest

MANIFEST = "/net/projects2/litian-lab/scpan/encoders/data/manifests/bsnip2_synthseg.csv"
OUT = "/net/projects2/litian-lab/scpan/encoders/artifacts/cache/bsnip2/synthseg_native.npz"
POOLED_GRID = (32, 32, 32)
# default.yaml's checkpoints.synthseg points at ADNI's mri_synthseg output --
# SynthSeg has no single shared checkpoint across datasets (unlike MASS's one
# pretrained weights file), each dataset needs its own segmentation run. The
# BSNIP2-specific config points checkpoints.synthseg at the right posteriors dir.
CONFIG_PATH = "/net/projects2/litian-lab/scpan/encoders/config/bsnip2_synthseg.yaml"


def main() -> None:
    config = load_config(CONFIG_PATH)
    rows = read_manifest(MANIFEST)
    extractor = build_extractor("synthseg", config.raw, "cuda", pooled_grid_override=POOLED_GRID)

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
        metadata={"disease": "bsnip2", "encoder": "synthseg", "native_tokens": False,
                 "pooled_grid_override": list(POOLED_GRID),
                 "native_grid_true": [128, 128, 128],
                 "note": "de-pooled to 32^3 (not fully native 128^3) for tractability; "
                         "see module docstring", "token_shape": list(array.shape[1:])},
    )
    out_path = save_cache(OUT, cache)
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
