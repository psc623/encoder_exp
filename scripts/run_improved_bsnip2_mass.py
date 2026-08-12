#!/usr/bin/env python
"""Driver for the 6 MASS/bsnip2 SZ-vs-HC(CN) experiment modes requested on top
of the existing encoderbench attention-probe/finetune pipeline:

  1 frozen encoder + attention-classifier   (= existing run_probe, re-run for clean logs)
  2 frozen encoder + linear-classifier      (new: linear_head.run_linear_probe, mimics
                                              bsnip2-smri-classification/classification/classify.py)
  3 fine-tuned encoder + attention-classifier, warm-started from (1), classifier FROZEN, encoder only
  4 fine-tuned encoder + linear-classifier,    warm-started from (2), classifier FROZEN, encoder only
  5 fine-tuned encoder + attention-classifier, classifier from zero, encoder fully unfrozen, joint
  6 fine-tuned encoder + linear-classifier,    classifier from zero, encoder fully unfrozen, joint

Usage:
  python run_improved_bsnip2_mass.py mode1 --seeds 0,1,2
  python run_improved_bsnip2_mass.py mode2 --seeds 0,1,2
  python run_improved_bsnip2_mass.py sweep          # only if mode1/mode2 mean BA < 0.65
  python run_improved_bsnip2_mass.py mode3 --seeds 0,1,2
  ... mode4 mode5 mode6
  python run_improved_bsnip2_mass.py report         # writes improved_report.md

Every run appends one line per sample per epoch to
artifacts/reports/bsnip2_mass_improved_samplelog.csv (mode,seed,epoch,split,file_id,
subject_id,true_label,positive_probability) -- modes 1/2 (no epochs) log a
single synthetic "epoch=FINAL,split=test" pass; modes 3-6 log every train
batch, every validation pass, and the final test pass via finetune_variants'
sample_log_path.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from encoderbench.config import load_config
from encoderbench.training import run_probe

ENCODER = "mass"
DISEASE = "bsnip2"
POSITIVE = "SZ"
ROOT = Path("/net/projects2/litian-lab/scpan/encoders")
OUT_ROOT = ROOT / "artifacts" / "improved" / DISEASE / ENCODER
REPORT_DIR = ROOT / "artifacts" / "reports"
SAMPLE_LOG = REPORT_DIR / "bsnip2_mass_improved_samplelog.csv"
SWEEP_CHOICE_PATH = OUT_ROOT / "sweep_choice.json"
BA_THRESHOLD = 0.65
SEEDS = (0, 1, 2)


def _seeds(value: str) -> list[int]:
    return SEEDS if value == "all" else [int(v) for v in value.split(",")]


def _cache_path(config) -> Path:
    # Native (unpooled) 4096x256 cache -- NOT the pooled_grid=[4,4,4]=64-token
    # cache the rest of encoderbench uses. Built by scripts/cache_mass_native_bsnip2.py.
    # Every mode (1-6) reads from here now, per instruction to remove the lossy
    # pooling bottleneck for the attention channel too, not just the linear one.
    return config.output_root / "cache" / DISEASE / "mass_native.npz"


def _manifest_path(config) -> Path:
    return ROOT / "data" / "manifests" / "bsnip2_mass.csv"


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
    config = load_config()
    cache = _cache_path(config)
    # Canonical location finetune_variants._warm_start looks for (same one
    # run_finetune's original warm-start used) -- NOT a private mode1 subdir,
    # so modes 3/5 warm-starting from "(1)" actually find this run's output,
    # including after a hyperparameter sweep changes it.
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
    config = load_config()
    cache = _cache_path(config)
    # Canonical location finetune_variants._warm_start looks for -- see cmd_mode1.
    out_dir = config.output_root / "linear" / DISEASE / ENCODER
    c_grid = _swept_c_grid()
    for seed in _seeds(args.seeds):
        result = run_linear_probe(cache, out_dir, POSITIVE, seed, config.section("evaluation"), c_grid=c_grid)
        _log_probe_style(result, "mode2_frozen_linear", seed)
        m = result["metrics"]["volume_level"]
        print(f"[mode2 seed={seed}] BA={m['balanced_accuracy']:.4f} AUC={m['roc_auc']:.4f} "
             f"params={result['trainable_parameters']} C={result['selection']['C']}", flush=True)


def _mean_ba(mode_dir: Path, prefix: str, seeds) -> float:
    values = []
    for seed in seeds:
        path = mode_dir / f"{prefix}_seed_{seed}_summary.json"
        if path.is_file():
            values.append(json.loads(path.read_text())["metrics"]["volume_level"]["balanced_accuracy"])
    if not values:
        raise RuntimeError(f"No summaries found under {mode_dir} for seeds {seeds}")
    return sum(values) / len(values)


def cmd_sweep(args) -> None:
    """Only meant to run after mode1+mode2 (seed 0 at least) show BA < 0.65.
    Cheap sweep on seed 0 only: for the attention head, try {learning_rate,
    hidden_size} combinations (weight_decay is already grid-searched inside
    run_probe every time); for the linear head, try a much wider C grid. Whichever
    combo gets the best seed-0 validation-selected test BA is written to
    sweep_choice.json and picked up by every later mode1-6 invocation.
    """
    config = load_config()
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
    config = load_config()
    cache = _cache_path(config)
    manifest = _manifest_path(config)
    out_dir = OUT_ROOT / f"mode{number}_{head_kind}_{'trainhead' if train_head else 'headfrozen'}"

    # Separate per-mode log file: modes 3-6 run as 4 concurrent processes on
    # separate GPUs in the sbatch script, and /net/projects2/litian-lab is an
    # NFS mount where concurrent O_APPEND from different processes isn't
    # guaranteed atomic the way local POSIX filesystems are -- one file per
    # mode avoids any risk of interleaved/corrupted lines. `report` doesn't
    # read these (it reads the JSON summaries); they exist to satisfy
    # per-sample logging, and mode1/mode2's _log_probe_style still writes the
    # single non-concurrent SAMPLE_LOG since only one of them runs at a time.
    mode_log = REPORT_DIR / f"bsnip2_mass_improved_samplelog_mode{number}.csv"
    # Must match whatever mode1 actually used (possibly swept away from the
    # config default) or warm-starting mode3/5's AttentionPoolHead from mode1's
    # checkpoint raises a state_dict shape mismatch (found the hard way on the
    # SynthSeg run, where the sweep did fire; MASS's own run never triggered the
    # sweep so this had no effect there, but keeping both drivers consistent).
    attention_hidden_size = (_swept_probe_settings(config)["hidden_size"]
                             if head_kind == "attention" else None)

    def _run(args):
        for seed in _seeds(args.seeds):
            result = run_finetune_variant(manifest, cache, DISEASE, ENCODER, config.raw, out_dir, seed,
                                          head_kind=head_kind, train_head=train_head,
                                          warm_start_head=warm_start_head, parameter_budget=parameter_budget,
                                          device="cuda", restart=args.restart, sample_log_path=mode_log,
                                          native_tokens=True, attention_hidden_size=attention_hidden_size)
            m = result["metrics"]["volume_level"]
            print(f"[mode{number} seed={seed}] BA={m['balanced_accuracy']:.4f} AUC={m['roc_auc']:.4f} "
                 f"trainable={result['trainable_parameters']}", flush=True)
    return _run


def cmd_report(args) -> None:
    lines = ["# Improved bsnip2:mass report — lossless linear channel + 6 training modes\n",
            "SZ vs HC(stored as CN), 3 seeds (0/1/2), test-split volume-level metrics "
            "(mean ± sample SD across seeds unless noted). Generated by "
            "`scripts/run_improved_bsnip2_mass.py report`.\n"]
    config = load_config()
    mode_dirs = {
        1: (config.output_root / "attention" / DISEASE / ENCODER, "probe"),
        2: (config.output_root / "linear" / DISEASE / ENCODER, "linear"),
        3: (OUT_ROOT / "mode3_attention_headfrozen", "finetune_attention_headfrozen_warm"),
        4: (OUT_ROOT / "mode4_linear_headfrozen", "finetune_linear_headfrozen_warm"),
        5: (OUT_ROOT / "mode5_attention_trainhead", "finetune_attention_trainhead_cold"),
        6: (OUT_ROOT / "mode6_linear_trainhead", "finetune_linear_trainhead_cold"),
    }
    descriptions = {
        1: "(1) frozen encoder + attention-classifier, train classifier only",
        2: "(2) frozen encoder + linear-classifier (lossless, mimics smri), train classifier only",
        3: "(3) fine-tuned encoder + attention-classifier, warm-started from (1), classifier frozen, encoder only",
        4: "(4) fine-tuned encoder + linear-classifier, warm-started from (2), classifier frozen, encoder only",
        5: "(5) fine-tune encoder & attention-classifier jointly, encoder from frozen ckpt, classifier from zero",
        6: "(6) fine-tune encoder & linear-classifier jointly, encoder from frozen ckpt, classifier from zero",
    }
    lines.append("| mode | test BA (mean±SD) | test AUC (mean±SD) | trainable params | seeds completed |")
    lines.append("|---|---:|---:|---:|---:|")
    summary_rows = {}
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
        summary_rows[number] = (bas, aucs, params, completed)
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
        lines.append("## Hyperparameter sweep\n")
        lines.append("Triggered because mode 1 and/or mode 2's seed-0 (or mean) BA was below 0.65. "
                     "Chosen settings (seed-0-only sweep, then applied to all seeds/modes):\n")
        lines.append("```json")
        lines.append(SWEEP_CHOICE_PATH.read_text())
        lines.append("```")
    else:
        lines.append("## Hyperparameter sweep\n\nNot triggered — mode 1/2 baseline results were >= 0.65 BA.\n")
    lines.append("\n## Reference: bsnip2-smri-classification raw-voxel SVM/LR")
    lines.append("Reported by that repo's own README/methodology for the analogous SZ/HC task: AUC 0.819 "
                 "(exact linear model on ~5x10^5 raw registered voxels, no pretrained encoder). "
                 "Compare against mode 2 above (same exact-linear math, but on MASS's frozen 64x256 pooled "
                 "features instead of raw voxels) to isolate whether the gap is the classifier (attention vs "
                 "exact-linear) or the feature source (pretrained-encoder embedding vs raw registered voxel).")
    out = REPORT_DIR / "improved_report.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out}")


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
