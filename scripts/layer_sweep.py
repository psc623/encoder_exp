"""Choose the depth each frozen encoder is read at, using validation data only.

For every candidate tap the encoder is run over the train and validation
volumes, the 64 pooled tokens are mean-reduced, a logistic probe is fit on
train and scored on validation. The winning layer is the one with the best
validation ROC-AUC. The test split is never read here, so the choice stays
honest; the selected depth then goes into config/default.yaml and the real
attention probes are re-run from scratch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from encoderbench.config import load_config
from encoderbench.extractors import build_extractor
from encoderbench.manifest import read_manifest
from encoderbench.workflows import resolve_device

# Number of taps each encoder exposes, ordered shallow to deep.
LAYER_COUNTS = {"medsiglip": 27, "brainiac": 12, "mass": 5, "anatcl": 4}


def evaluate(train_x: np.ndarray, train_y: np.ndarray,
             val_x: np.ndarray, val_y: np.ndarray) -> tuple[float, float]:
    mean, std = train_x.mean(0), train_x.std(0) + 1e-6
    probe = LogisticRegression(max_iter=4000, C=0.1, class_weight="balanced")
    probe.fit((train_x - mean) / std, train_y)
    probability = probe.predict_proba((val_x - mean) / std)[:, 1]
    return (float(balanced_accuracy_score(val_y, probability >= 0.5)),
            float(roc_auc_score(val_y, probability)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("encoder", choices=sorted(LAYER_COUNTS))
    parser.add_argument("--disease", default="ad")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stride", type=int, default=1,
                        help="Sample every Nth tap; the 27-layer ViTs are swept coarsely "
                             "because each candidate costs a full forward pass")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    config = load_config(None)
    rows = [row for row in read_manifest(config.manifest(args.disease))
            if row["split"] in ("train", "validation")]
    positive = "AD" if args.disease == "ad" else "SCZ"
    labels = np.asarray([1 if row["group"] == positive else 0 for row in rows])
    is_train = np.asarray([row["split"] == "train" for row in rows])
    print(f"{args.encoder}: {is_train.sum()} train + {(~is_train).sum()} validation volumes "
          f"(test split deliberately untouched)", flush=True)

    extractor = build_extractor(args.encoder, config.raw, resolve_device(args.device))
    total = LAYER_COUNTS[args.encoder]
    candidates = sorted({*range(0, total, max(1, args.stride)), total - 1}) + [-1]
    results = []
    for layer in candidates:
        pooled = np.stack([extractor.extract_at(row["path"], layer).float().mean(dim=0).numpy()
                           for row in rows])
        accuracy, auc = evaluate(pooled[is_train], labels[is_train],
                                 pooled[~is_train], labels[~is_train])
        results.append({"layer": layer, "width": int(pooled.shape[-1]),
                        "validation_balanced_accuracy": accuracy, "validation_auc": auc})
        print(f"  layer {layer:>3} (D={pooled.shape[-1]:>4}): "
              f"val BA={accuracy:.3f}  val AUC={auc:.3f}", flush=True)

    best = max(results, key=lambda item: item["validation_auc"])
    deepest = next(item for item in results if item["layer"] == -1)
    print(f"  -> best layer {best['layer']} (val AUC {best['validation_auc']:.3f}) "
          f"vs deepest {deepest['validation_auc']:.3f}", flush=True)

    output = Path(args.out_dir or config.output_root / "audits" / args.disease)
    output.mkdir(parents=True, exist_ok=True)
    (output / f"layer_sweep_{args.encoder}.json").write_text(json.dumps(
        {"encoder": args.encoder, "disease": args.disease,
         "selection_split": "validation", "selection_metric": "roc_auc",
         "train_volumes": int(is_train.sum()), "validation_volumes": int((~is_train).sum()),
         "results": results, "selected_layer": best["layer"],
         "deepest_validation_auc": deepest["validation_auc"]},
        indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
