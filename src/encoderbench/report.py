"""Aggregate seeds without best-seed selection and keep experiment families separate."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from encoderbench.metrics import paired_bootstrap_difference
from encoderbench.utils import write_json


def _seed_stats(items: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = ("balanced_accuracy", "roc_auc", "sensitivity", "specificity", "f1")
    result: dict[str, Any] = {"seeds": [item["seed"] for item in items], "runs": items}
    for metric in metric_names:
        values = []
        for item in items:
            metrics = item["metrics"].get("volume_level", item["metrics"])
            if metrics.get(metric) is not None:
                values.append(float(metrics[metric]))
        if values:
            result[metric] = {"mean": float(np.mean(values)),
                              "standard_deviation": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                              "median": float(np.median(values)), "values": values}
    return result


def _subject_predictions(path: str | Path, positive: str) -> tuple[dict[str, str], dict[str, str]]:
    grouped_probability: dict[str, list[float]] = defaultdict(list)
    truth: dict[str, str] = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            subject = row["subject_id"]
            truth[subject] = row["true"]
            grouped_probability[subject].append(float(row["positive_probability"]))
    prediction = {subject: positive if np.mean(values) >= 0.5 else "CN"
                  for subject, values in grouped_probability.items()}
    return truth, prediction


def generate_report(input_root: str | Path, output_dir: str | Path,
                    bootstrap_samples: int = 2000, bootstrap_seed: int = 0) -> dict[str, Any]:
    root = Path(input_root).resolve()
    records = []
    for path in root.rglob("*_summary.json"):
        try:
            with path.open(encoding="utf-8") as handle:
                value = json.load(handle)
            if value.get("kind") in ("attention_probe", "zero_shot", "linear_bridge", "resampler_bridge"):
                value["_summary_path"] = str(path)
                records.append(value)
        except (OSError, json.JSONDecodeError):
            continue
    if not records:
        raise ValueError(f"No run summaries found beneath {root}")
    report: dict[str, Any] = {"attention_probe": {}, "zero_shot": {},
                              "linear_bridge": {}, "resampler_bridge": {}}
    for family in ("attention_probe", "linear_bridge", "resampler_bridge"):
        groups: dict[tuple[str, str, bool], list[dict[str, Any]]] = defaultdict(list)
        for item in records:
            if item["kind"] == family:
                groups[(item["disease"], item["encoder"], bool(item.get("shuffled_labels")))].append(item)
        for (disease, encoder, shuffled), items in groups.items():
            key = f"{disease}:{encoder}" + (":shuffled" if shuffled else "")
            report[family][key] = _seed_stats(sorted(items, key=lambda item: item["seed"]))
    for item in records:
        if item["kind"] == "zero_shot":
            report["zero_shot"][f"{item['disease']}:{item['model']}"] = item
    report["paired_balanced_accuracy_differences"] = _paired(records, bootstrap_samples,
                                                               bootstrap_seed)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "final_report.json", report)
    _write_markdown(output / "final_report.md", report)
    return report


def _paired(records: list[dict[str, Any]], samples: int, seed: int) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for family in ("attention_probe", "linear_bridge", "resampler_bridge"):
        for disease in ("ad", "scz"):
            subset = [item for item in records if item["kind"] == family and item["disease"] == disease
                      and not item.get("shuffled_labels")]
            by_encoder: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for item in subset:
                by_encoder[item["encoder"]].append(item)
            for first, second in combinations(sorted(by_encoder), 2):
                seed_values = []
                first_by_seed = {item["seed"]: item for item in by_encoder[first]}
                second_by_seed = {item["seed"]: item for item in by_encoder[second]}
                for run_seed in sorted(set(first_by_seed) & set(second_by_seed)):
                    a, b = first_by_seed[run_seed], second_by_seed[run_seed]
                    positive = "AD" if disease == "ad" else "SCZ"
                    truth_a, prediction_a = _subject_predictions(a["predictions"], positive)
                    truth_b, prediction_b = _subject_predictions(b["predictions"], positive)
                    if truth_a != truth_b:
                        raise ValueError(f"Paired test subjects differ for {first} and {second}")
                    seed_values.append({"seed": run_seed, **paired_bootstrap_difference(
                        truth_a, prediction_a, prediction_b, positive, samples, seed
                    )})
                if seed_values:
                    output[f"{family}:{disease}:{first}_minus_{second}"] = seed_values
    return output


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = ["# Encoder Experiment Report", "",
             "Attention probes, native zero-shot VLMs, linear bridges, and resampler bridges are "
             "reported separately; no cross-family ranking is computed.", ""]
    for family, title in (("attention_probe", "Attention probe"), ("zero_shot", "Native zero-shot"),
                          ("linear_bridge", "Linear bridge"), ("resampler_bridge", "Resampler bridge")):
        lines.extend((f"## {title}", ""))
        entries = report[family]
        if not entries:
            lines.extend(("No completed runs.", ""))
            continue
        if family == "zero_shot":
            lines.extend(("| run | balanced accuracy | sensitivity | specificity | F1 |", "|---|---:|---:|---:|---:|"))
            for key, item in sorted(entries.items()):
                metric = item["metrics"]
                lines.append(f"| {key} | {metric['balanced_accuracy']:.3f} | {metric['sensitivity']:.3f} | "
                             f"{metric['specificity']:.3f} | {metric['f1']:.3f} |")
        else:
            lines.extend(("| run | balanced accuracy mean ± SD | median | ROC-AUC mean |", "|---|---:|---:|---:|"))
            for key, item in sorted(entries.items()):
                balanced = item.get("balanced_accuracy", {})
                auc = item.get("roc_auc", {})
                lines.append(f"| {key} | {balanced.get('mean', float('nan')):.3f} ± "
                             f"{balanced.get('standard_deviation', float('nan')):.3f} | "
                             f"{balanced.get('median', float('nan')):.3f} | "
                             f"{auc.get('mean', float('nan')):.3f} |")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
