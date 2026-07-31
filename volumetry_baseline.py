"""Classical regional-volumetry baseline: SynthSeg structure volumes -> logistic regression.

This is the reference line a reviewer asks for -- "how much better than measuring
the hippocampus is your foundation model?" -- so it deliberately uses no learned
representation at all, just the per-structure volumes mri_synthseg already writes
with --vol. Volumes are divided by total intracranial volume to remove head-size
effects, which is the standard correction in the AD volumetry literature.

Selection (the regularisation strength) uses the validation split only; the test
split is scored once at the end with the same subject-level aggregation and
subject-clustered bootstrap the attention probes use.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from encoderbench.config import load_config
from encoderbench.manifest import read_manifest
from encoderbench.metrics import aggregate_subjects, binary_metrics, cluster_bootstrap

VOLUME_DIR = Path("/net/projects2/litian-lab/scpan/dataset/ADNI_processed/synthseg_outputs/vol")
ICV_COLUMN = "total intracranial"


def load_volumes(rows: list[dict]) -> tuple[np.ndarray, list[str], list[int]]:
    """Return the per-structure volume matrix plus the manifest rows it covers."""
    features: list[list[float]] = []
    kept: list[int] = []
    names: list[str] | None = None
    for index, row in enumerate(rows):
        stem = Path(row["path"]).name
        for suffix in (".nii.gz", ".nii"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        path = VOLUME_DIR / f"{stem}_synthseg_vol.csv"
        if not path.is_file() or path.stat().st_size == 0:
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            record = next(iter(csv.DictReader(handle)), None)
        if record is None:
            continue
        # mri_synthseg writes the scan id under an unnamed first column, and
        # "total intracranial" is the ICV used as the head-size denominator, so
        # neither belongs in the structure feature vector.
        columns = [key for key in record if key not in ("", "subject", ICV_COLUMN)]
        values = [float(record[key]) for key in columns] + [float(record[ICV_COLUMN])]
        if names is None:
            names = columns
        elif columns != names:
            continue
        features.append(values)
        kept.append(index)
    if names is None:
        raise SystemExit(f"No SynthSeg volume CSVs found under {VOLUME_DIR}")
    return np.asarray(features, dtype=np.float64), names, kept


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--disease", default="ad")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    config = load_config(None)
    rows = read_manifest(config.manifest(args.disease))
    matrix, names, kept = load_volumes(rows)
    rows = [rows[index] for index in kept]
    positive = "AD" if args.disease == "ad" else "SCZ"
    print(f"{len(rows)} volumes with SynthSeg volumetry, {len(names)} structures", flush=True)

    # Head-size correction: express every structure as a fraction of the
    # intracranial volume, the standard adjustment in the AD volumetry literature.
    volumes, icv = matrix[:, :-1], matrix[:, -1:]
    features = np.log(np.clip(volumes / np.clip(icv, 1e-6, None), 1e-9, None))

    labels = np.asarray([1 if row["group"] == positive else 0 for row in rows])
    splits = np.asarray([row["split"] for row in rows])
    subjects = np.asarray([row["subject_id"] for row in rows])
    truth = np.asarray([row["group"] for row in rows])
    train, validation, test = splits == "train", splits == "validation", splits == "test"
    for name, mask in (("train", train), ("validation", validation), ("test", test)):
        print(f"  {name}: {int(mask.sum())} volumes, {int(labels[mask].sum())} {positive}", flush=True)

    mean, std = features[train].mean(0), features[train].std(0) + 1e-9
    scaled = (features - mean) / std

    best = None
    for strength in (0.001, 0.01, 0.1, 1.0, 10.0):
        model = LogisticRegression(max_iter=5000, C=strength, class_weight="balanced")
        model.fit(scaled[train], labels[train])
        probability = model.predict_proba(scaled[validation])[:, 1]
        prediction = np.where(probability >= 0.5, positive, "CN")
        score = binary_metrics(truth[validation], prediction, probability, positive)
        print(f"  C={strength:<7} validation BA={score['balanced_accuracy']:.3f} "
              f"AUC={score['roc_auc']:.3f}", flush=True)
        if best is None or score["roc_auc"] > best[1]:
            best = (strength, score["roc_auc"])

    strength = best[0]
    model = LogisticRegression(max_iter=5000, C=strength, class_weight="balanced")
    model.fit(scaled[train], labels[train])
    probability = model.predict_proba(scaled[test])[:, 1]
    prediction = np.where(probability >= 0.5, positive, "CN")

    volume_level = binary_metrics(truth[test], prediction, probability, positive)
    subject_truth, subject_pred, subject_prob, subject_ids = aggregate_subjects(
        truth[test], probability, subjects[test], positive)
    evaluation = config.raw["evaluation"]
    result = {
        "kind": "synthseg_volumetry_logistic_regression",
        "disease": args.disease, "selected_C": strength,
        "structures": names, "feature": "log(structure volume / total volume)",
        "volumes_scored": int(test.sum()),
        "volume_level": volume_level,
        "volume_level_subject_cluster_ci": cluster_bootstrap(
            truth[test], prediction, probability, subjects[test], positive,
            int(evaluation["bootstrap_samples"]), int(evaluation["bootstrap_seed"])),
        "subject_level": binary_metrics(subject_truth, subject_pred, subject_prob, positive),
        "top_coefficients": sorted(
            ({"structure": name, "coefficient": float(weight)}
             for name, weight in zip(names, model.coef_[0])),
            key=lambda item: abs(item["coefficient"]), reverse=True)[:12],
    }
    print(f"\nTEST volume-level  BA={volume_level['balanced_accuracy']:.3f} "
          f"AUC={volume_level['roc_auc']:.3f}")
    print(f"TEST subject-level BA={result['subject_level']['balanced_accuracy']:.3f} "
          f"AUC={result['subject_level']['roc_auc']:.3f}")
    print("\nMost informative structures:")
    for item in result["top_coefficients"][:8]:
        print(f"  {item['coefficient']:+.3f}  {item['structure']}")

    output = Path(args.out or config.output_root / "reports" / f"volumetry_baseline_{args.disease}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"\nsaved {output}")


if __name__ == "__main__":
    main()
