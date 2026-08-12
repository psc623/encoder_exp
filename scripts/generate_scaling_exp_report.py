#!/usr/bin/env python
"""Aggregate the A/B x dataset1/dataset2 MASS scaling comparison into
/net/projects2/litian-lab/scpan/encoders/report/scaling_exp/: one markdown
report, one loss/balanced-accuracy curve SVG per run, and copies of every
summary JSON + sbatch log referenced.

No matplotlib in any of this repo's environments (med/med310/pyenv312), so
plots are hand-rolled dependency-free SVG polylines -- simple line charts,
no external plotting library needed.
"""
from __future__ import annotations

import json
import math
import shutil
import statistics
from pathlib import Path

ROOT = Path("/net/projects2/litian-lab/scpan/encoders")
OUT_DIR = ROOT / "report" / "scaling_exp"
LOG_DIR = Path("/net/projects2/litian-lab/scpan/logs")

CELLS = {
    "A1": {"kind": "probe", "dir": ROOT / "artifacts/attention/ad/mass_native_d1",
          "dataset": "dataset1 (ADNI_processed_clean)", "setting": "A: frozen probe"},
    "B1": {"kind": "finetune", "dir": ROOT / "artifacts/finetune/ad/mass_native_d1_warmstart",
          "dataset": "dataset1 (ADNI_processed_clean)",
          "setting": "B: warm-started joint finetune"},
    "A2": {"kind": "probe", "dir": ROOT / "artifacts/attention/ad/mass_native_d2",
          "dataset": "dataset2 (ADNI_full_screen, not skull-stripped)", "setting": "A: frozen probe"},
    "B2": {"kind": "finetune", "dir": ROOT / "artifacts/finetune/ad/mass_native_d2_warmstart",
          "dataset": "dataset2 (ADNI_full_screen, not skull-stripped)",
          "setting": "B: warm-started joint finetune"},
    "A2ss": {"kind": "probe", "dir": ROOT / "artifacts/attention/ad/mass_native_d2_ss",
            "dataset": "dataset2 (ADNI_full_screen, HD-BET skull-stripped)",
            "setting": "A: frozen probe"},
    "B2ss": {"kind": "finetune", "dir": ROOT / "artifacts/finetune/ad/mass_native_d2_ss_warmstart",
            "dataset": "dataset2 (ADNI_full_screen, HD-BET skull-stripped)",
            "setting": "B: warm-started joint finetune"},
    "A2ss70": {"kind": "probe", "dir": ROOT / "artifacts/attention/ad/mass_native_d2ss70",
              "dataset": "dataset2 (same 431 subj/images as A2ss, reshuffled 70/15/15)",
              "setting": "A: frozen probe"},
    "B2ss70": {"kind": "finetune", "dir": ROOT / "artifacts/finetune/ad/mass_native_d2ss70_warmstart",
              "dataset": "dataset2 (same 431 subj/images as A2ss, reshuffled 70/15/15)",
              "setting": "B: warm-started joint finetune, encoder_lr=5e-4 (raised from 1e-5)"},
    "A3": {"kind": "probe", "dir": ROOT / "artifacts/attention/ad/mass_native_d3",
          "dataset": "dataset3 (ADNI_full, every visit, CN/AD only, skull-stripped)",
          "setting": "A: frozen probe"},
    "B3": {"kind": "finetune", "dir": ROOT / "artifacts/finetune/ad/mass_native_d3_warmstart",
          "dataset": "dataset3 (ADNI_full, every visit, CN/AD only, skull-stripped)",
          "setting": "B: warm-started joint finetune, encoder_lr=5e-4 (raised from 1e-5)"},
    "A4": {"kind": "probe", "dir": ROOT / "artifacts/attention/ad/mass_native_d4",
          "dataset": "dataset4 (dataset2 1.5T + ADNI_add_full 3T, 819 subj, 70/15/15)",
          "setting": "A: frozen probe"},
    "B4": {"kind": "finetune", "dir": ROOT / "artifacts/finetune/ad/mass_native_d4_warmstart",
          "dataset": "dataset4 (dataset2 1.5T + ADNI_add_full 3T, 819 subj, 70/15/15)",
          "setting": "B: warm-started joint finetune, encoder_lr=5e-4 (raised from 1e-5)"},
}
SEEDS = (0, 1, 2)


def _svg_polyline(points: list[tuple[float, float]], width: int, height: int,
                  margin: int, color: str, y_min: float, y_max: float, x_max: float) -> str:
    # Non-finite y (e.g. val_loss=inf from a numerically diverged epoch, see
    # the A4/B4 write-up) would otherwise poison the whole axis scale and
    # collapse every other point to the same pixel row -- break the line into
    # separate finite-only segments instead of pretending those epochs are 0.
    if not points or y_max <= y_min:
        return ""
    span = y_max - y_min
    segments: list[list[str]] = [[]]
    for x, y in points:
        if not math.isfinite(y):
            if segments[-1]:
                segments.append([])
            continue
        px = margin + (x / max(x_max, 1e-9)) * (width - 2 * margin)
        py = height - margin - ((y - y_min) / span) * (height - 2 * margin)
        segments[-1].append(f"{px:.1f},{py:.1f}")
    parts = []
    for coords in segments:
        if len(coords) >= 2:
            parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" '
                        f'points="{" ".join(coords)}"/>')
    return "\n".join(parts)


def plot_history(history: list[dict], title: str, out_path: Path,
                 selected_epoch: int | None = None) -> None:
    width, height, margin = 640, 360, 50
    epochs = [entry["epoch"] for entry in history]
    x_max = max(epochs) if epochs else 1
    train_key = "train_loss" if history and "train_loss" in history[0] else None
    val_loss = [entry["val_loss"] for entry in history]
    val_ba = [entry.get("val_balanced_accuracy", entry.get("balanced_accuracy", 0.0)) for entry in history]
    loss_values = val_loss + ([entry[train_key] for entry in history] if train_key else [])
    finite_loss_values = [v for v in loss_values if math.isfinite(v)]
    loss_min, loss_max = ((min(finite_loss_values), max(finite_loss_values))
                          if finite_loss_values else (0.0, 1.0))
    loss_min = min(loss_min, 0.0)
    diverged_epoch = next((entry["epoch"] for entry in history
                           if not math.isfinite(entry["val_loss"])), None)

    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
          f'viewBox="0 0 {width} {height}" font-family="monospace" font-size="11">']
    svg.append(f'<rect width="{width}" height="{height}" fill="white"/>')
    svg.append(f'<text x="{margin}" y="20" font-size="13" fill="black">{title}</text>')

    if train_key:
        train_points = [(entry["epoch"], entry[train_key]) for entry in history]
        svg.append(_svg_polyline(train_points, width, height, margin, "#1f77b4",
                                 loss_min, loss_max, x_max))
    val_loss_points = [(entry["epoch"], entry["val_loss"]) for entry in history]
    svg.append(_svg_polyline(val_loss_points, width, height, margin, "#d62728",
                             loss_min, loss_max, x_max))

    ba_min, ba_max = 0.0, 1.0
    ba_points = [(entry["epoch"], entry.get("val_balanced_accuracy", entry.get("balanced_accuracy", 0.0)))
                for entry in history]
    svg.append(_svg_polyline(ba_points, width, height, margin, "#2ca02c", ba_min, ba_max, x_max))

    if selected_epoch is not None:
        px = margin + (selected_epoch / max(x_max, 1e-9)) * (width - 2 * margin)
        svg.append(f'<line x1="{px:.1f}" y1="{margin}" x2="{px:.1f}" y2="{height - margin}" '
                  f'stroke="gray" stroke-dasharray="4,3"/>')
        svg.append(f'<text x="{px + 3:.1f}" y="{margin + 10}" fill="gray">selected epoch {selected_epoch}</text>')

    if diverged_epoch is not None:
        px = margin + (diverged_epoch / max(x_max, 1e-9)) * (width - 2 * margin)
        svg.append(f'<line x1="{px:.1f}" y1="{margin}" x2="{px:.1f}" y2="{height - margin}" '
                  f'stroke="#d62728" stroke-dasharray="2,2"/>')
        svg.append(f'<text x="{px + 3:.1f}" y="{margin + 22}" fill="#d62728">'
                  f'val_loss diverged (inf) from epoch {diverged_epoch}</text>')

    legend_y = height - 15
    if train_key:
        svg.append(f'<text x="{margin}" y="{legend_y}" fill="#1f77b4">-- train loss</text>')
    svg.append(f'<text x="{margin + 130}" y="{legend_y}" fill="#d62728">-- val loss</text>')
    svg.append(f'<text x="{margin + 250}" y="{legend_y}" fill="#2ca02c">-- val balanced accuracy (0-1 scale)</text>')
    svg.append("</svg>")
    out_path.write_text("\n".join(svg))


def fmt(values: list[float]) -> str:
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{mean:.3f} ± {sd:.3f}"


def collect_cell(name: str, spec: dict) -> dict | None:
    cell_dir = spec["dir"]
    kind = spec["kind"]
    prefix = "probe" if kind == "probe" else "finetune"
    summaries = []
    for seed in SEEDS:
        path = cell_dir / f"{prefix}_seed_{seed}_summary.json"
        if not path.is_file():
            return None
        summaries.append(json.loads(path.read_text()))

    plot_dir = OUT_DIR / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for summary in summaries:
        seed = summary["seed"]
        history = summary.get("history", [])
        if kind == "probe":
            best_wd = summary["selection"].get("weight_decay")
            history = [entry for entry in history if entry.get("weight_decay") == best_wd]
        selected_epoch = summary["selection"].get("epoch")
        plot_history(history, f"{name} seed={seed} ({spec['setting']}, {spec['dataset']})",
                    plot_dir / f"{name}_seed{seed}.svg", selected_epoch)

    vba = [s["metrics"]["volume_level"]["balanced_accuracy"] for s in summaries]
    vauc = [s["metrics"]["volume_level"]["roc_auc"] for s in summaries
           if s["metrics"]["volume_level"].get("roc_auc") is not None]
    sba = [s["metrics"]["subject_level"]["balanced_accuracy"] for s in summaries]
    sauc = [s["metrics"]["subject_level"]["roc_auc"] for s in summaries
           if s["metrics"]["subject_level"].get("roc_auc") is not None]
    return {"summaries": summaries, "volume_ba": fmt(vba), "volume_auc": fmt(vauc) if vauc else "n/a",
           "subject_ba": fmt(sba), "subject_auc": fmt(sauc) if sauc else "n/a",
           "selected_epochs": [s["selection"]["epoch"] for s in summaries],
           "trainable_parameters": summaries[0].get("trainable_parameters"),
           "head_warm_started_from": [s.get("head_warm_started_from") for s in summaries]}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, spec in CELLS.items():
        result = collect_cell(name, spec)
        if result is not None:
            results[name] = result
            print(f"{name}: done ({len(result['summaries'])} seeds)")
        else:
            print(f"{name}: not complete yet, skipping")

    log_dir = OUT_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ("scaling_exp_d1_*.log", "scaling_exp_d2_*.log"):
        for log in LOG_DIR.glob(pattern):
            shutil.copy2(log, log_dir / log.name)

    lines = ["# MASS A/B x dataset scaling comparison\n",
            "Native tokens (no 4x4x4 pooling) throughout. A = frozen encoder + probe head "
            "(300 epochs, cached features, cheap). B = head warm-started from that same A "
            "run's probe, then encoder+head jointly finetuned (30 epochs, patience 15, "
            "unbounded parameter budget -- same 9,345,888-parameter upstream MASS stack as "
            "every other run in this series). dataset1 = ADNI_processed_clean (548 vol / 416 "
            "subj, external skull-strip, non-uniform spacing). dataset2 = ADNI_full_screen "
            "(431 subj, MASS-native preprocessed, uniform 1.5mm spacing) -- same data as the "
            "Group 1 run in ADNI_MASS_scaling_report.md.\n",
            "**A2/B2 vs A2ss/B2ss**: A2/B2 diagnosed a large dataset1-vs-dataset2 gap that "
            "persisted even at the frozen-probe level (no joint-training instability possible), "
            "traced to dataset2 never being skull-stripped -- MASS's own recipe doesn't skull-strip, "
            "so its body-crop field of view came out 3.77x larger by volume than dataset1's "
            "(measured directly: ~200x241x242mm vs ~135x166x137mm mean bounding box), diluting the "
            "brain's share of the fixed 128^3 input grid and contaminating MASS's own whole-array "
            "percentile-clip intensity normalization with skull/scalp/neck intensities. A2ss/B2ss "
            "add HD-BET skull-stripping (already vendored + weight-cached in this repo, reused "
            "unchanged from src/encoderbench/bsnip2/hdbet_batch.py) ahead of MASS's reorient/"
            "resample, with the crop switched to the skull-strip mask's own nonzero bounding box "
            "instead of MASS's intensity-threshold body crop.\n",
            "**A3/B3**: dataset3 expands dataset2 to every ADNI visit (screening + m6/m12/.../m48) "
            "for the same 431 CN/AD subjects (MCI dropped), 1,649 volumes total -- same "
            "HD-BET-skull-stripped MASS-native recipe as A2ss/B2ss, and the 431 already-processed "
            "screening volumes were reused unchanged (idempotent, keyed by subject_id+file_id) "
            "rather than reprocessed. B3 also raises `encoder_learning_rate` from 1e-5 to 5e-4: "
            "directly measuring B1/B2/B2ss's finetuned encoder weights against the original MASS "
            "checkpoint found only 0.36-0.88% relative L2 change at 1e-5 (best epoch found at "
            "5-19 of the 30-epoch budget, well under it) -- the encoder was barely moving at all, "
            "which is why B tracked A so closely in every earlier cell. B3 tests whether letting "
            "the encoder actually adapt (not just nudge) helps once it has ~4x more training data "
            "to adapt against.\n",
            "**A4/B4**: dataset4 adds 388 new baseline CN/AD subjects from ADNI_add_full "
            "(ADNI2-4, confirmed 100% field_strength=3.0T across sampled metadata, zero "
            "subject_id overlap with dataset2's 431 subjects) to dataset2 (ADNI1, 1.5T), for "
            "819 total subjects. Both pools are HD-BET-skull-stripped + MASS-native preprocessed "
            "the same way as A2ss/B2ss (idempotent, shared output dirs). Each pool is "
            "independently split at test_fraction=0.15 / validation_fraction=0.1765-of-remaining "
            "(-> ~70/15/15) and concatenated -- since the two pools are exactly the 1.5T/3T "
            "partition with no subject overlap, this is equivalent to a joint "
            "(field_strength x diagnosis) stratified split of the combined pool without needing "
            "a second stratification key in manifest.stratified_subject_split, and it guarantees "
            "every split contains both field strengths (avoiding a field-strength-confounds-with-"
            "split failure mode). This replaces the previous default test_fraction=0.5 used for "
            "dataset1/dataset2/dataset3 (train was consistently smaller than test, ~42-44%), "
            "chosen deliberately here to give B4's joint finetune (9.3M trainable params, "
            "data-hungry) more to train on now that the subject pool has nearly doubled. "
            "dataset2's 431 subjects are reshuffled from scratch at these new fractions rather "
            "than reusing A2ss/B2ss's split, so A4/B4 is not a subject-level-identical comparison "
            "against A2ss/B2ss (different test set membership) -- deliberate tradeoff, chosen "
            "over freezing the old split, which could not reach 70/15/15 (dataset2's old 216-"
            "subject test set alone was already 26% of the combined 819). B4 uses "
            "encoder_learning_rate=5e-4, same as B3.\n",
            "**A2ss70/B2ss70**: A4/B4 changed two things at once relative to A2ss/B2ss -- more "
            "subjects (819 vs 431) and a different split ratio (70/15/15 vs A2ss/B2ss's original "
            "~42/50/8, from the manifest builder's test_fraction=0.5 default). A2ss70/B2ss70 "
            "holds the subject pool fixed at dataset2's same 431 subjects and same already-"
            "preprocessed images as A2ss/B2ss, and changes only the split ratio (same reshuffled "
            "70/15/15 manifest used as dataset4's 1.5T half), isolating how much of any A4-vs-"
            "A2ss gap is attributable to the split ratio alone versus the added ADNI_add_full "
            "training data.\n",
            "| cell | dataset | setting | volume BA | volume AUC | subject BA | subject AUC | "
            "selected epochs | trainable params |",
            "|---|---|---|---:|---:|---:|---:|---|---:|"]
    for name, spec in CELLS.items():
        if name not in results:
            lines.append(f"| {name} | {spec['dataset']} | {spec['setting']} | "
                         f"(not complete) | | | | | |")
            continue
        r = results[name]
        lines.append(f"| {name} | {spec['dataset']} | {spec['setting']} | {r['volume_ba']} | "
                     f"{r['volume_auc']} | {r['subject_ba']} | {r['subject_auc']} | "
                     f"{r['selected_epochs']} | {r['trainable_parameters']:,} |"
                     if r['trainable_parameters'] else "| n/a |")

    lines.append("\n## Per-cell detail\n")
    for name, spec in CELLS.items():
        if name not in results:
            continue
        r = results[name]
        lines.append(f"### {name} -- {spec['setting']}, {spec['dataset']}\n")
        lines.append(f"Head warm-started from: {r['head_warm_started_from']}\n")
        lines.append("Per-seed loss/balanced-accuracy curves:\n")
        for seed in SEEDS:
            lines.append(f"- seed {seed}: `plots/{name}_seed{seed}.svg`")
        lines.append("")

    (OUT_DIR / "scaling_exp_report.md").write_text("\n".join(lines))
    print(f"wrote {OUT_DIR / 'scaling_exp_report.md'}")


if __name__ == "__main__":
    main()
