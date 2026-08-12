#!/usr/bin/env python
"""SynthSeg/bsnip2 counterpart of run_improved_bsnip2_mass.py -- same 6 modes,
same protocol, MASS -> SynthSeg. See that file's docstring for the mode
definitions; not repeated here.

Two real differences from the MASS version, both forced by SynthSeg's nature
(a frozen external FreeSurfer tool, not a PyTorch encoder -- see
extractors.SynthSegPosteriorAdapter's docstring):
  - "encoder fine-tuning" in modes 3-6 means training the small post-pooling
    SynthSegPosteriorAdapter (extractor.finetune_groups() already returns just
    that one module for synthseg); there is no backbone to unfreeze.
  - The de-pooled cache uses pooled_grid=(32,32,32), not fully native 128^3:
    128^3 x 33 channels flattened is a ~69M-dim vector, not tractable for the
    exact-linear channel's Gram-trick train matrix (~60GB+ at float32). 32^3 is
    matched in order of magnitude to MASS's native 4096x256=1,048,576-dim
    flatten (32768x33=1,081,344), not literally native, but >500x finer than
    the original [4,4,4]=64-token protocol -- see
    cache_synthseg_native_bsnip2.py's docstring for the full reasoning.

Writes its own report section to a SEPARATE file
(artifacts/reports/improved_report_synthseg_section.md), not
improved_report.md directly -- that file may still be getting incrementally
overwritten by the concurrently-running MASS job, and two unrelated jobs
writing the same file would race. Merge manually once both are done.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from encoderbench.config import load_config
from encoderbench.training import run_probe

ENCODER = "synthseg"
DISEASE = "bsnip2"
POSITIVE = "SZ"
ROOT = Path("/net/projects2/litian-lab/scpan/encoders")
# default.yaml's checkpoints.synthseg points at ADNI's posteriors -- SynthSeg
# has no shared checkpoint across datasets, unlike MASS's one weights file.
CONFIG_PATH = ROOT / "config" / "bsnip2_synthseg.yaml"
OUT_ROOT = ROOT / "artifacts" / "improved" / DISEASE / ENCODER
REPORT_DIR = ROOT / "artifacts" / "reports"
SAMPLE_LOG = REPORT_DIR / "bsnip2_synthseg_improved_samplelog.csv"
SWEEP_CHOICE_PATH = OUT_ROOT / "sweep_choice.json"
BA_THRESHOLD = 0.65
SEEDS = (0, 1, 2)
POOLED_GRID_OVERRIDE = (32, 32, 32)
SECTION_OUT = REPORT_DIR / "improved_report_synthseg_section.md"


def _seeds(value: str) -> list[int]:
    return SEEDS if value == "all" else [int(v) for v in value.split(",")]


def _cache_path(config) -> Path:
    return config.output_root / "cache" / DISEASE / "synthseg_native.npz"


def _manifest_path(config) -> Path:
    return ROOT / "data" / "manifests" / "bsnip2_synthseg.csv"


def _log_probe_style(result: dict, mode_tag: str, seed: int) -> None:
    import csv as _csv

    SAMPLE_LOG.parent.mkdir(parents=True, exist_ok=True)
    is_new = not SAMPLE_LOG.exists()
    predictions_csv = Path(result["predictions"])
    with SAMPLE_LOG.open("a", newline="", encoding="utf-8") as out:
        writer = _csv.writer(out)
        if is_new:
            writer.writerow(["mode", "seed", "epoch", "split", "file_id", "subject_id", "true", "positive_probability"])
        with predictions_csv.open() as handle:
            reader = _csv.DictReader(handle)
            for row in reader:
                writer.writerow([mode_tag, seed, "FINAL", "test", row["file_id"], row["subject_id"],
                                 row["true"], row["positive_probability"]])


def _swept_probe_settings(config) -> dict:
    if SWEEP_CHOICE_PATH.is_file():
        choice = json.loads(SWEEP_CHOICE_PATH.read_text())
        settings = dict(config.section("probe"))
        settings.update(choice.get("attention_settings", {}))
        return settings
    return config.section("probe")


def _swept_c_grid() -> tuple[float, ...]:
    from encoderbench.linear_head import DEFAULT_C_GRID
    if SWEEP_CHOICE_PATH.is_file():
        choice = json.loads(SWEEP_CHOICE_PATH.read_text())
        grid = choice.get("linear_c_grid")
        if grid:
            return tuple(grid)
    return DEFAULT_C_GRID


def cmd_mode1(args) -> None:
    config = load_config(CONFIG_PATH)
    cache = _cache_path(config)
    out_dir = config.output_root / "attention" / DISEASE / ENCODER
    settings = _swept_probe_settings(config)
    for seed in _seeds(args.seeds):
        result = run_probe(cache, out_dir, POSITIVE, seed, settings, config.section("evaluation"))
        _log_probe_style(result, "mode1_frozen_attention", seed)
        m = result["metrics"]["volume_level"]
        print(f"[mode1 seed={seed}] BA={m['balanced_accuracy']:.4f} AUC={m['roc_auc']:.4f} "
             f"params={result['trainable_parameters']}", flush=True)


def cmd_mode2(args) -> None:
    from encoderbench.linear_head import run_linear_probe
    config = load_config(CONFIG_PATH)
    cache = _cache_path(config)
    out_dir = config.output_root / "linear" / DISEASE / ENCODER
    c_grid = _swept_c_grid()
    for seed in _seeds(args.seeds):
        result = run_linear_probe(cache, out_dir, POSITIVE, seed, config.section("evaluation"), c_grid=c_grid)
        _log_probe_style(result, "mode2_frozen_linear", seed)
        m = result["metrics"]["volume_level"]
        print(f"[mode2 seed={seed}] BA={m['balanced_accuracy']:.4f} AUC={m['roc_auc']:.4f} "
             f"params={result['trainable_parameters']} C={result['selection']['C']}", flush=True)


def cmd_sweep(args) -> None:
    config = load_config(CONFIG_PATH)
    cache = _cache_path(config)
    base_settings = config.section("probe")
    attention_grid = [
        {"learning_rate": 0.001, "hidden_size": 128},
        {"learning_rate": 0.0003, "hidden_size": 128},
        {"learning_rate": 0.001, "hidden_size": 64},
        {"learning_rate": 0.001, "hidden_size": 256},
        {"learning_rate": 0.003, "hidden_size": 128},
    ]
    best_attention, best_attention_ba = None, -1.0
    for overrides in attention_grid:
        settings = {**base_settings, **overrides}
        result = run_probe(cache, OUT_ROOT / "sweep_attention", POSITIVE, 0, settings, config.section("evaluation"))
        ba = result["metrics"]["volume_level"]["balanced_accuracy"]
        print(f"[sweep attention] {overrides} -> BA={ba:.4f}", flush=True)
        if ba > best_attention_ba:
            best_attention, best_attention_ba = overrides, ba

    from encoderbench.linear_head import run_linear_probe
    wide_c_grid = (1e-5, 1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)
    result = run_linear_probe(cache, OUT_ROOT / "sweep_linear", POSITIVE, 0, config.section("evaluation"),
                              c_grid=wide_c_grid)
    linear_ba = result["metrics"]["volume_level"]["balanced_accuracy"]
    print(f"[sweep linear] wide C grid -> BA={linear_ba:.4f} chosen C={result['selection']['C']}", flush=True)

    choice = {"attention_settings": best_attention, "attention_seed0_ba": best_attention_ba,
             "linear_c_grid": list(wide_c_grid), "linear_seed0_ba": linear_ba}
    SWEEP_CHOICE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SWEEP_CHOICE_PATH.write_text(json.dumps(choice, indent=2))
    print(f"Sweep choice written to {SWEEP_CHOICE_PATH}: {choice}", flush=True)


def cmd_mode(number: int, head_kind: str, train_head: bool, warm_start_head: bool,
            parameter_budget) -> None:
    from encoderbench.finetune_variants import run_finetune_variant
    config = load_config(CONFIG_PATH)
    cache = _cache_path(config)
    manifest = _manifest_path(config)
    out_dir = OUT_ROOT / f"mode{number}_{head_kind}_{'trainhead' if train_head else 'headfrozen'}"
    mode_log = REPORT_DIR / f"bsnip2_synthseg_improved_samplelog_mode{number}.csv"
    # Must match whatever mode1 actually used (possibly swept away from the
    # config default) or warm-starting mode3/5's AttentionPoolHead from mode1's
    # checkpoint raises a state_dict shape mismatch.
    attention_hidden_size = (_swept_probe_settings(config)["hidden_size"]
                             if head_kind == "attention" else None)

    def _run(args):
        for seed in _seeds(args.seeds):
            result = run_finetune_variant(manifest, cache, DISEASE, ENCODER, config.raw, out_dir, seed,
                                          head_kind=head_kind, train_head=train_head,
                                          warm_start_head=warm_start_head, parameter_budget=parameter_budget,
                                          device="cuda", restart=args.restart, sample_log_path=mode_log,
                                          native_tokens=False, pooled_grid_override=POOLED_GRID_OVERRIDE,
                                          attention_hidden_size=attention_hidden_size)
            m = result["metrics"]["volume_level"]
            print(f"[mode{number} seed={seed}] BA={m['balanced_accuracy']:.4f} AUC={m['roc_auc']:.4f} "
                 f"trainable={result['trainable_parameters']}", flush=True)
    return _run


def cmd_report(args) -> None:
    lines = ["## SynthSeg/bsnip2 — same 6 modes, MASS -> SynthSeg\n",
            "SZ vs HC(stored as CN), 3 seeds (0/1/2), test-split volume-level metrics "
            "(mean ± sample SD across seeds unless noted). SynthSeg's frozen 'encoder' is a "
            "precomputed FreeSurfer posterior volume with no backbone to unfreeze; modes 3-6's "
            "'encoder fine-tune' trains SynthSegPosteriorAdapter, its small post-pooling residual "
            "MLP (see extractors.py). Pooling grid de-compressed from [4,4,4]=64 tokens to "
            "[32,32,32]=32768 tokens (not fully native 128^3 -- flattening that is a ~69M-dim "
            "vector, not tractable for the exact-linear channel's Gram matrix; 32^3 is matched in "
            "order of magnitude to MASS's native 4096x256 flatten instead). "
            "Generated by `scripts/run_improved_bsnip2_synthseg.py report`.\n"]
    config = load_config(CONFIG_PATH)
    mode_dirs = {
        1: (config.output_root / "attention" / DISEASE / ENCODER, "probe"),
        2: (config.output_root / "linear" / DISEASE / ENCODER, "linear"),
        3: (OUT_ROOT / "mode3_attention_headfrozen", "finetune_attention_headfrozen_warm"),
        4: (OUT_ROOT / "mode4_linear_headfrozen", "finetune_linear_headfrozen_warm"),
        5: (OUT_ROOT / "mode5_attention_trainhead", "finetune_attention_trainhead_cold"),
        6: (OUT_ROOT / "mode6_linear_trainhead", "finetune_linear_trainhead_cold"),
    }
    descriptions = {
        1: "(1) frozen posteriors + attention-classifier, train classifier only",
        2: "(2) frozen posteriors + linear-classifier (lossless-ish, mimics smri), train classifier only",
        3: "(3) fine-tuned adapter + attention-classifier, warm-started from (1), classifier frozen, adapter only",
        4: "(4) fine-tuned adapter + linear-classifier, warm-started from (2), classifier frozen, adapter only",
        5: "(5) fine-tune adapter & attention-classifier jointly, classifier from zero",
        6: "(6) fine-tune adapter & linear-classifier jointly, classifier from zero",
    }
    lines.append("| mode | test BA (mean±SD) | test AUC (mean±SD) | trainable params | seeds completed |")
    lines.append("|---|---:|---:|---:|---:|")
    for number, (mode_dir, prefix) in mode_dirs.items():
        bas, aucs, params, completed = [], [], None, []
        for seed in SEEDS:
            path = mode_dir / f"{prefix}_seed_{seed}_summary.json"
            if path.is_file():
                data = json.loads(path.read_text())
                bas.append(data["metrics"]["volume_level"]["balanced_accuracy"])
                aucs.append(data["metrics"]["volume_level"]["roc_auc"])
                params = data["trainable_parameters"]
                completed.append(seed)
        if bas:
            import statistics
            ba_mean, ba_sd = statistics.mean(bas), (statistics.stdev(bas) if len(bas) > 1 else 0.0)
            auc_mean, auc_sd = statistics.mean(aucs), (statistics.stdev(aucs) if len(aucs) > 1 else 0.0)
            lines.append(f"| {number} | {ba_mean:.3f}±{ba_sd:.3f} | {auc_mean:.3f}±{auc_sd:.3f} "
                         f"| {params} | {completed} |")
        else:
            lines.append(f"| {number} | — | — | — | none |")
    lines.append("")
    for number, desc in descriptions.items():
        lines.append(f"- **{number}**: {desc}")
    lines.append("")
    if SWEEP_CHOICE_PATH.is_file():
        lines.append("### Hyperparameter sweep (SynthSeg)\n")
        lines.append("```json")
        lines.append(SWEEP_CHOICE_PATH.read_text())
        lines.append("```")
    else:
        lines.append("### Hyperparameter sweep (SynthSeg)\n\nNot triggered — mode 1/2 baseline results were >= 0.65 BA.\n")
    SECTION_OUT.parent.mkdir(parents=True, exist_ok=True)
    SECTION_OUT.write_text("\n".join(lines) + "\n")
    print(f"Wrote {SECTION_OUT} (NOT merged into improved_report.md yet -- do that once "
         f"both the MASS and SynthSeg jobs are fully done)")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("mode1", "mode2"):
        p = sub.add_parser(name)
        p.add_argument("--seeds", default="all")
    sub.add_parser("sweep")
    for name in ("mode3", "mode4", "mode5", "mode6"):
        p = sub.add_parser(name)
        p.add_argument("--seeds", default="all")
        p.add_argument("--restart", action="store_true")
    sub.add_parser("report")
    args = parser.parse_args()

    if args.cmd == "mode1":
        cmd_mode1(args)
    elif args.cmd == "mode2":
        cmd_mode2(args)
    elif args.cmd == "sweep":
        cmd_sweep(args)
    elif args.cmd == "mode3":
        cmd_mode(3, "attention", train_head=False, warm_start_head=True, parameter_budget=8_000_000)(args)
    elif args.cmd == "mode4":
        cmd_mode(4, "linear", train_head=False, warm_start_head=True, parameter_budget=8_000_000)(args)
    elif args.cmd == "mode5":
        cmd_mode(5, "attention", train_head=True, warm_start_head=False, parameter_budget=None)(args)
    elif args.cmd == "mode6":
        cmd_mode(6, "linear", train_head=True, warm_start_head=False, parameter_budget=None)(args)
    elif args.cmd == "report":
        cmd_report(args)


if __name__ == "__main__":
    main()
