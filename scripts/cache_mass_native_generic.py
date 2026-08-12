#!/usr/bin/env python
"""Generic version of cache_mass_native_bsnip2.py: build a native (unpooled,
16x16x16=4096-token) MASS feature cache from any manifest/output pair, so the
same code builds the wmr-registered variants (cropped and uncropped) without
duplicating the extraction loop.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from encoderbench.cache import FeatureCache, save_cache
from encoderbench.config import load_config
from encoderbench.extractors import build_extractor
from encoderbench.manifest import read_manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    config = load_config()
    rows = read_manifest(args.manifest)
    extractor = build_extractor("mass", config.raw, "cuda", native_tokens=True)

    t0 = time.time()
    features = []
    for index, row in enumerate(rows):
        tokens = extractor.extract(row["path"])
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
                 "manifest": args.manifest, "note": args.note,
                 "token_shape": list(array.shape[1:])},
    )
    out_path = save_cache(args.out, cache)
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
