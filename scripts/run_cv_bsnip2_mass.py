#!/usr/bin/env python
"""MASS/bsnip2 six modes under the repeated-random-split (smri-style) protocol.

Scope, deliberately narrow: MASS only, on the MASS-preprocessed BSNIP2 volumes
(`data/manifests/bsnip2_mass.csv` + the native unpooled 4096x256 cache
`artifacts/cache/bsnip2/mass_native.npz`). No wmr variants, no SynthSeg.

Protocol change vs. the runs already in improved_report.md: instead of one
frozen train/validation/test split scored with 3 model seeds, this draws 5
independent stratified subject-level splits (test 30%, validation 20% of the
remaining train fold) and reports mean +/- SD over those 5 repeats -- matching
`bsnip2-smri-classification/classification/classify.py`'s `--n-repeats 5
--test-frac 0.30`. The exact-linear head additionally selects C by inner 5-fold
CV inside the training fold (`--inner-folds 5`), smri's own method. See
`encoderbench/cv_protocol.py` for why.

Modes are unchanged from run_improved_bsnip2_mass.py:
  1 frozen encoder + attention head       4 encoder-only finetune + linear head (warm, head frozen)
  2 frozen encoder + exact-linear head    5 joint finetune + attention head (head from zero)
  3 encoder-only finetune + attention     6 joint finetune + linear head (head from zero)
    (warm, head frozen)
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from encoderbench.config import load_config
from encoderbench.cv_protocol import describe, repeated_splits
from encoderbench.manifest import read_manifest
from encoderbench.training import run_probe

ENCODER = "mass"
DISEASE = "bsnip2"
POSITIVE = "SZ"
ROOT = Path("/net/projects2/litian-lab/scpan/encoders")
MANIFEST = ROOT / "data" / "manifests" / "bsnip2_mass.csv"
CACHE = ROOT / "artifacts" / "cache" / DISEASE / "mass_native.npz"
OUT_ROOT = ROOT / "artifacts" / "cv" / DISEASE / ENCODER
REPORT_DIR = ROOT / "artifacts" / "reports"
SECTION_OUT = REPORT_DIR / "improved_report_cv_section.md"
TUNED_PATH = OUT_ROOT / "finetune_settings.json"
N_REPEATS = 5
INNER_FOLDS = 5

MODES = {
    1: dict(kind="probe", head="attention", label="frozen encoder + attention head"),
    2: dict(kind="linear", head="linear", label="frozen encoder + exact-linear head"),
    3: dict(kind="finetune", head="attention", train_head=False, warm=True, budget=8_000_000,
            label="encoder-only finetune + attention head (warm-started, head frozen)"),
    4: dict(kind="finetune", head="linear", train_head=False, warm=True, budget=8_000_000,
            label="encoder-only finetune + linear head (warm-started, head frozen)"),
    5: dict(kind="finetune", head="attention", train_head=True, warm=False, budget=None,
            label="joint finetune + attention head (head from zero, encoder unbudgeted)"),
    6: dict(kind="finetune", head="linear", train_head=True, warm=False, budget=None,
            label="joint finetune + linear head (head from zero, encoder unbudgeted)"),
}


def _rows_and_labels():
    rows = read_manifest(MANIFEST)
    subject_ids = np.asarray([row["subject_id"] for row in rows])
    labels = np.asarray([row["group"] for row in rows])
    return rows, subject_ids, labels


def _splits_for(repeat: int, shape: str = "three_way") -> np.ndarray:
    _, subject_ids, labels = _rows_and_labels()
    for index, splits in repeated_splits(subject_ids, labels, shape=shape, n_repeats=N_REPEATS):
        if index == repeat:
            return splits
    raise ValueError(f"repeat {repeat} outside 0..{N_REPEATS - 1}")


def _mode_dir(mode: int, variant: str = "") -> Path:
    return OUT_ROOT / f"mode{mode}{('_' + variant) if variant else ''}"


def tuned_settings() -> dict:
    """Finetune hyperparameters chosen by the sweep (validation only). Empty
    until `sweep` has run, in which case the config defaults apply."""
    if TUNED_PATH.is_file():
        return json.loads(TUNED_PATH.read_text()).get("chosen", {})
    return {}


def run_mode(mode: int, repeats: list[int], settings_override: dict | None = None,
             variant: str = "", quiet: bool = False) -> list[dict]:
    config = load_config()
    spec = MODES[mode]
    out_dir = _mode_dir(mode, variant)
    results = []
    for repeat in repeats:
        shape = "two_way" if spec["kind"] == "linear" else "three_way"
        splits = _splits_for(repeat, shape="three_way")  # same rows for every mode
        suffix = f"_rep{repeat}"
        if spec["kind"] == "probe":
            result = run_probe(CACHE, out_dir, POSITIVE, seed=0,
                               settings=config.section("probe"),
                               evaluation=config.section("evaluation"),
                               splits_override=splits, tag_suffix=suffix)
        elif spec["kind"] == "linear":
            from encoderbench.linear_head import run_linear_probe
            result = run_linear_probe(CACHE, out_dir, POSITIVE, seed=0,
                                      evaluation=config.section("evaluation"),
                                      splits_override=splits, tag_suffix=suffix,
                                      inner_folds=INNER_FOLDS)
        else:
            from encoderbench.finetune_variants import run_finetune_variant
            override = dict(tuned_settings())
            override.update(settings_override or {})
            # Warm-start reads mode1/mode2's checkpoint for the *same* repeat.
            warm_dir = _mode_dir(1 if spec["head"] == "attention" else 2)
            result = run_finetune_variant(
                MANIFEST, CACHE, DISEASE, ENCODER, config.raw, out_dir, seed=0,
                head_kind=spec["head"], train_head=spec["train_head"],
                warm_start_head=spec["warm"], parameter_budget=spec["budget"],
                device="cuda", restart=True, native_tokens=True,
                splits_override=splits, tag_suffix=suffix,
                settings_override=override or None,
                warm_start_dir=warm_dir, warm_start_tag=f"seed_0{suffix}",
            )
        results.append(result)
        metrics = result["metrics"]["volume_level"]
        if not quiet:
            print(f"[mode{mode}{'/' + variant if variant else ''} repeat={repeat}] "
                  f"BA={metrics['balanced_accuracy']:.4f} AUC={metrics['roc_auc']:.4f} "
                  f"params={result['trainable_parameters']}", flush=True)
    return results


def _collect(mode: int, variant: str = "") -> dict:
    spec = MODES[mode]
    prefix = {"probe": "probe", "linear": "linear"}.get(spec["kind"])
    if prefix is None:
        head = spec["head"]
        prefix = (f"finetune_{head}_{'trainhead' if spec['train_head'] else 'headfrozen'}_"
                  f"{'warm' if spec['warm'] else 'cold'}")
    bas, aucs, params, done = [], [], None, []
    for repeat in range(N_REPEATS):
        path = _mode_dir(mode, variant) / f"{prefix}_seed_0_rep{repeat}_summary.json"
        if path.is_file():
            data = json.loads(path.read_text())
            bas.append(data["metrics"]["volume_level"]["balanced_accuracy"])
            aucs.append(data["metrics"]["volume_level"]["roc_auc"])
            params = data["trainable_parameters"]
            done.append(repeat)
    return {"ba": bas, "auc": aucs, "params": params, "repeats": done}


def _fmt(values: list[float]) -> str:
    if not values:
        return "—"
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{mean:.3f}±{sd:.3f}"


def cmd_report(args) -> None:
    lines = [
        "## MASS/bsnip2 six modes, re-run under smri's repeated-random-split protocol\n",
        "Same MASS-preprocessed BSNIP2 volumes and same native unpooled 4096x256 cache as the "
        "first section of this report; only the *evaluation protocol* and the finetune "
        "hyperparameters changed. Splits now follow "
        "`bsnip2-smri-classification/classification/classify.py`: 5 independent stratified "
        "subject-level draws, test 30% each, reported as mean±SD over the 5 repeats (previously: "
        "one frozen split scored with 3 model seeds). Per repeat the training fold is 284 volumes "
        "and validation 72 (previously 216 / 38). The exact-linear head selects C by inner 5-fold "
        "CV inside the training fold, smri's own method.\n",
    ]
    lines.append("| mode | test BA (mean±SD) | test AUC (mean±SD) | trainable params | repeats |")
    lines.append("|---|---:|---:|---:|---:|")
    for mode in sorted(MODES):
        stats = _collect(mode)
        lines.append(f"| {mode} | {_fmt(stats['ba'])} | {_fmt(stats['auc'])} "
                     f"| {stats['params'] if stats['params'] else '—'} | {stats['repeats']} |")
    lines.append("")
    for mode in sorted(MODES):
        lines.append(f"- **{mode}**: {MODES[mode]['label']}")
    lines.append("")
    if TUNED_PATH.is_file():
        lines.append("### Finetune settings chosen by the sweep\n")
        lines.append("```json")
        lines.append(TUNED_PATH.read_text().rstrip())
        lines.append("```")
    SECTION_OUT.parent.mkdir(parents=True, exist_ok=True)
    SECTION_OUT.write_text("\n".join(lines) + "\n")
    print(f"Wrote {SECTION_OUT}")


def cmd_splits(args) -> None:
    _, subject_ids, labels = _rows_and_labels()
    for repeat, splits in repeated_splits(subject_ids, labels, shape="three_way",
                                          n_repeats=N_REPEATS):
        print(f"repeat {repeat}: {describe(splits, labels)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("--mode", type=int, required=True, choices=sorted(MODES))
    run.add_argument("--repeats", default="all")
    run.add_argument("--variant", default="")
    run.add_argument("--override", default=None, help="JSON dict of finetune setting overrides")
    sub.add_parser("report")
    sub.add_parser("splits")
    args = parser.parse_args()

    if args.cmd == "run":
        repeats = (list(range(N_REPEATS)) if args.repeats == "all"
                   else [int(v) for v in args.repeats.split(",")])
        override = json.loads(args.override) if args.override else None
        run_mode(args.mode, repeats, settings_override=override, variant=args.variant)
    elif args.cmd == "report":
        cmd_report(args)
    elif args.cmd == "splits":
        cmd_splits(args)


if __name__ == "__main__":
    main()
