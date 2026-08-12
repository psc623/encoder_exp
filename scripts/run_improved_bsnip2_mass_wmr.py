#!/usr/bin/env python
"""Registration-hypothesis test: MASS/bsnip2 mode1 (frozen attention) and
mode2 (frozen exact-linear) on SPM12 MNI-registered 'wmr' whole-head volumes,
instead of bsnip2_mass.csv's raw unregistered ones -- see improved_report.md's
analysis section for why this is the natural next experiment. Two variants:

  nocrop: wmr fed straight to MASS's clip_zscore + resize(128^3)
  crop:   wmr first run through MASS's own body-crop geometry step, then same

Writes its own report section (does not touch improved_report.md directly --
that file's canonical MASS section was written by run_improved_bsnip2_mass.py
and shouldn't be clobbered by a different script); merge manually once done.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from encoderbench.config import load_config
from encoderbench.training import run_probe

DISEASE = "bsnip2"
POSITIVE = "SZ"
ROOT = Path("/net/projects2/litian-lab/scpan/encoders")
REPORT_DIR = ROOT / "artifacts" / "reports"
SECTION_OUT = REPORT_DIR / "improved_report_wmr_section.md"
SEEDS = (0, 1, 2)
VARIANTS = ("nocrop", "crop")


def _seeds(value: str) -> list[int]:
    return SEEDS if value == "all" else [int(v) for v in value.split(",")]


def _out_root(variant: str) -> Path:
    return ROOT / "artifacts" / "improved" / DISEASE / f"mass_wmr_{variant}"


def _cache_path(variant: str) -> Path:
    return ROOT / "artifacts" / "cache" / DISEASE / f"mass_wmr_{variant}_native.npz"


def cmd_mode1(args) -> None:
    config = load_config()
    cache = _cache_path(args.variant)
    out_dir = _out_root(args.variant) / "attention"
    settings = config.section("probe")
    for seed in _seeds(args.seeds):
        result = run_probe(cache, out_dir, POSITIVE, seed, settings, config.section("evaluation"))
        m = result["metrics"]["volume_level"]
        print(f"[mode1 {args.variant} seed={seed}] BA={m['balanced_accuracy']:.4f} "
             f"AUC={m['roc_auc']:.4f} params={result['trainable_parameters']}", flush=True)


def cmd_mode2(args) -> None:
    from encoderbench.linear_head import run_linear_probe
    config = load_config()
    cache = _cache_path(args.variant)
    out_dir = _out_root(args.variant) / "linear"
    for seed in _seeds(args.seeds):
        result = run_linear_probe(cache, out_dir, POSITIVE, seed, config.section("evaluation"))
        m = result["metrics"]["volume_level"]
        print(f"[mode2 {args.variant} seed={seed}] BA={m['balanced_accuracy']:.4f} "
             f"AUC={m['roc_auc']:.4f} params={result['trainable_parameters']} "
             f"C={result['selection']['C']}", flush=True)


def cmd_report(args) -> None:
    lines = ["## MASS/bsnip2 — SPM12 MNI-registered ('wmr') whole-head input, vs. the unregistered baseline\n",
            "Tests whether the lack of inter-subject deformable registration in MASS's own "
            "preprocessing (see improved_report.md's Analysis section) is really capping the "
            "unregistered-native-token numbers. 469/509 subjects have a matching `wmr` file "
            "(SPM12-warped to MNI, 121x145x121, 1.5mm isotropic — already matching MASS's own "
            "target spacing); the other 40 are dropped, same train/val/test split otherwise. "
            "Two variants: `nocrop` feeds the registered volume straight to MASS's own "
            "clip_zscore+resize; `crop` first runs it through MASS's own body-crop geometry step "
            "(controls for wmr's larger background proportion, ~26% nonzero, vs. what MASS's own "
            "un-registered pipeline produces). mode1/mode2 only (frozen probes — the fastest, "
            "most direct test of the hypothesis before committing to a full 6-mode/3-seed run).\n"]
    lines.append("| variant | mode | test BA (mean±SD) | test AUC (mean±SD) | seeds completed |")
    lines.append("|---|---|---:|---:|---:|")
    for variant in VARIANTS:
        for number, (subdir, prefix) in ((1, ("attention", "probe")), (2, ("linear", "linear"))):
            mode_dir = _out_root(variant) / subdir
            bas, aucs, completed = [], [], []
            for seed in SEEDS:
                path = mode_dir / f"{prefix}_seed_{seed}_summary.json"
                if path.is_file():
                    data = json.loads(path.read_text())
                    bas.append(data["metrics"]["volume_level"]["balanced_accuracy"])
                    aucs.append(data["metrics"]["volume_level"]["roc_auc"])
                    completed.append(seed)
            if bas:
                import statistics
                ba_mean = statistics.mean(bas)
                ba_sd = statistics.stdev(bas) if len(bas) > 1 else 0.0
                auc_mean = statistics.mean(aucs)
                auc_sd = statistics.stdev(aucs) if len(aucs) > 1 else 0.0
                lines.append(f"| {variant} | {number} | {ba_mean:.3f}±{ba_sd:.3f} "
                             f"| {auc_mean:.3f}±{auc_sd:.3f} | {completed} |")
            else:
                lines.append(f"| {variant} | {number} | — | — | none |")
    lines.append("")
    lines.append("Reference points already established: unregistered MASS native-token mode1 = "
                 "0.668±0.017 BA / 0.723±0.024 AUC, mode2 = 0.650±0.000 / 0.718±0.000 "
                 "(`improved_report.md`); smri raw-voxel exact-LR (registered, full resolution) "
                 "= 0.819 AUC.")
    SECTION_OUT.parent.mkdir(parents=True, exist_ok=True)
    SECTION_OUT.write_text("\n".join(lines) + "\n")
    print(f"Wrote {SECTION_OUT}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("mode1", "mode2"):
        p = sub.add_parser(name)
        p.add_argument("--variant", required=True, choices=VARIANTS)
        p.add_argument("--seeds", default="all")
    sub.add_parser("report")
    args = parser.parse_args()
    if args.cmd == "mode1":
        cmd_mode1(args)
    elif args.cmd == "mode2":
        cmd_mode2(args)
    elif args.cmd == "report":
        cmd_report(args)


if __name__ == "__main__":
    main()
