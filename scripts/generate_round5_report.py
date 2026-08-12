#!/usr/bin/env python
"""Round-5 report: B2ss70 and B4 under the lowered-rate / longer-budget schedule,
side by side with the round-4 runs they are meant to be compared against.

Round 5 changes only B's optimisation schedule (encoder_lr 1e-4 -> 3e-5,
head_lr 1e-3 -> 3e-4, max_epochs 40 -> 120, patience/guard 15 -> 40, plus an
explicit overfit stop). The A cells were not rerun; round-5 B warm-starts from
the round-4 A checkpoints, so any difference here is the schedule.

Headline metrics are **balanced accuracy at the fixed 0.5 threshold and AUC**,
as requested. The validation-fitted-threshold number is still computed by the
training code and is carried in a secondary column rather than dropped.

The question this run exists to answer: round 4 drove training loss to exactly
0 by epoch 17-20 in all ten runs, so the epoch budget was never binding --
memorisation was. If lowering both rates works, training loss should reach 0
much later and peak validation AUC should be higher. If the curves just stretch
without the peak moving, the ceiling is the 301/573 training subjects rather
than the optimiser.

Plots are hand-rolled SVG (no matplotlib in any environment here), same as the
other generators, with one extra rule marking where the overfit stop fired.
"""
from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path

import numpy as np

ROOT = Path("/net/projects2/litian-lab/scpan/encoders")
OUT_DIR = ROOT / "report" / "scaling_exp_improve"
PLOT_DIR = OUT_DIR / "plots_round5"
LOG_DIR = Path("/net/projects2/litian-lab/scpan/logs")

CELLS = {
    "B2ss70": {"round5": ROOT / "artifacts/finetune/ad/improve5_d2ss70_warmstart",
               "round4": ROOT / "artifacts/finetune/ad/improve4_d2ss70_warmstart",
               "dataset": "dataset2 (431 subj, 70/15/15)"},
    "B4": {"round5": ROOT / "artifacts/finetune/ad/improve5_d4_warmstart",
           "round4": ROOT / "artifacts/finetune/ad/improve4_d4_warmstart",
           "dataset": "dataset4 (819 subj, 70/15/15)"},
}
WIDTH, HEIGHT, MARGIN = 760, 420, 58


def _polyline(points, color, y_min, y_max, x_max, dash=None):
    if not points or y_max <= y_min:
        return ""
    span = y_max - y_min
    segments = [[]]
    for x, y in points:
        if not math.isfinite(y):
            if segments[-1]:
                segments.append([])
            continue
        px = MARGIN + (x / max(x_max, 1e-9)) * (WIDTH - 2 * MARGIN)
        py = HEIGHT - MARGIN - ((y - y_min) / span) * (HEIGHT - 2 * MARGIN)
        segments[-1].append(f"{px:.1f},{py:.1f}")
    style = f' stroke-dasharray="{dash}"' if dash else ""
    return "\n".join(f'<polyline fill="none" stroke="{color}" stroke-width="1.8"{style} '
                     f'points="{" ".join(c)}"/>' for c in segments if len(c) >= 2)


def _vline(x, x_max, color, label, label_y, dash="4,3"):
    px = MARGIN + (x / max(x_max, 1e-9)) * (WIDTH - 2 * MARGIN)
    anchor = "end" if px > WIDTH * 0.72 else "start"
    return (f'<line x1="{px:.1f}" y1="{MARGIN}" x2="{px:.1f}" y2="{HEIGHT - MARGIN}" '
            f'stroke="{color}" stroke-dasharray="{dash}"/>'
            f'<text x="{px + (-4 if anchor == "end" else 4):.1f}" y="{label_y}" '
            f'fill="{color}" text-anchor="{anchor}">{label}</text>')


def plot_history(history, title, out_path, selected_epoch, subtitle, overfit_epoch=None):
    if not history:
        return
    x_max = max(e["epoch"] for e in history)
    losses = [e["val_loss"] for e in history] + [e["train_loss"] for e in history]
    finite = [v for v in losses if math.isfinite(v)]
    loss_min, loss_max = (min(finite), max(finite)) if finite else (0.0, 1.0)
    loss_min = min(loss_min, 0.0)
    loss_max = max(loss_max, loss_min + 1e-6)
    loss_min_entry = min((e for e in history if math.isfinite(e["val_loss"])),
                         key=lambda e: e["val_loss"], default=None)
    # Where training loss effectively reaches zero -- the quantity round 5 is
    # trying to push later, so it gets its own marker.
    memorised = next((e["epoch"] for e in history if e["train_loss"] < 1e-4), None)

    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
           f'viewBox="0 0 {WIDTH} {HEIGHT}" font-family="monospace" font-size="11">',
           f'<rect width="{WIDTH}" height="{HEIGHT}" fill="white"/>',
           f'<text x="{MARGIN}" y="20" font-size="13" fill="black">{title}</text>',
           f'<text x="{MARGIN}" y="36" font-size="10" fill="#555">{subtitle}</text>',
           f'<rect x="{MARGIN}" y="{MARGIN}" width="{WIDTH - 2 * MARGIN}" '
           f'height="{HEIGHT - 2 * MARGIN}" fill="none" stroke="#ddd"/>']
    for f in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = HEIGHT - MARGIN - f * (HEIGHT - 2 * MARGIN)
        svg.append(f'<text x="{MARGIN - 6}" y="{y + 3:.1f}" text-anchor="end" fill="#888">'
                   f'{loss_min + f * (loss_max - loss_min):.2f}</text>')
        svg.append(f'<text x="{WIDTH - MARGIN + 6}" y="{y + 3:.1f}" fill="#888">{f:.2f}</text>')
        x = MARGIN + f * (WIDTH - 2 * MARGIN)
        svg.append(f'<text x="{x:.1f}" y="{HEIGHT - MARGIN + 14}" text-anchor="middle" '
                   f'fill="#888">{int(f * x_max)}</text>')

    svg.append(_polyline([(e["epoch"], e["train_loss"]) for e in history],
                         "#1f77b4", loss_min, loss_max, x_max))
    svg.append(_polyline([(e["epoch"], e["val_loss"]) for e in history],
                         "#d62728", loss_min, loss_max, x_max))
    svg.append(_polyline([(e["epoch"], e.get("val_auc", float("nan"))) for e in history],
                         "#2ca02c", 0.0, 1.0, x_max))
    svg.append(_polyline([(e["epoch"], e["val_balanced_accuracy"]) for e in history],
                         "#9467bd", 0.0, 1.0, x_max, dash="3,2"))

    if loss_min_entry is not None:
        svg.append(_vline(loss_min_entry["epoch"], x_max, "#d62728",
                          f'val_loss min ep{loss_min_entry["epoch"]}', MARGIN + 26, "2,2"))
    svg.append(_vline(selected_epoch, x_max, "#333", f"selected ep{selected_epoch}", MARGIN + 12))
    if memorised is not None:
        svg.append(_vline(memorised, x_max, "#1f77b4",
                          f"train_loss~0 ep{memorised}", MARGIN + 40, "1,3"))
    if overfit_epoch is not None:
        svg.append(_vline(overfit_epoch, x_max, "#ff7f0e",
                          f"overfit stop ep{overfit_epoch}", MARGIN + 54, "6,3"))

    legend = [("#1f77b4", "train loss (left)"), ("#d62728", "val loss (left)"),
              ("#2ca02c", "val AUC (right)"), ("#9467bd", "val balanced acc @0.5 (right)")]
    for i, (color, label) in enumerate(legend):
        svg.append(f'<text x="{MARGIN + (i % 2) * 330}" y="{HEIGHT - 22 + (i // 2) * 13}" '
                   f'fill="{color}">-- {label}</text>')
    svg.append("</svg>")
    out_path.write_text("\n".join(svg))


def fmt(values):
    if not values:
        return "n/a"
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{statistics.mean(values):.3f} ± {sd:.3f}"


def seeds_present(directory: Path) -> list[int]:
    found = set()
    for path in directory.glob("finetune_seed_*_summary.json"):
        stem = path.name[len("finetune_seed_"):].split("_")[0]
        if stem.isdigit():
            found.add(int(stem))
    return sorted(found)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    rows, detail = [], []
    for name, spec in CELLS.items():
        if not spec["round5"].is_dir():
            print(f"{name}: round-5 directory missing, skipping")
            continue
        seeds = seeds_present(spec["round5"])
        if not seeds:
            print(f"{name}: no round-5 seed finished yet, skipping")
            continue
        ba5, auc5, baf5, sel, memo, ofs, ran = [], [], [], [], [], [], []
        for seed in seeds:
            summary = json.loads((spec["round5"] / f"finetune_seed_{seed}_summary.json").read_text())
            history, selection = summary["history"], summary["selection"]
            at_half = summary.get("metrics_at_half", summary["metrics"])
            ba5.append(at_half["volume_level"]["balanced_accuracy"])
            auc5.append(summary["metrics"]["volume_level"]["roc_auc"])
            baf5.append(summary["metrics"]["volume_level"]["balanced_accuracy"])
            sel.append(selection["epoch"])
            ran.append(history[-1]["epoch"])
            memo.append(next((e["epoch"] for e in history if e["train_loss"] < 1e-4), None))
            ofs.append(summary.get("overfit_stop_epoch"))
            plot_history(
                history, f"{name} seed={seed} -- round 5 (encoder_lr 3e-5, head_lr 3e-4)",
                PLOT_DIR / f"{name}_round5_seed{seed}.svg", selection["epoch"],
                f'ran {history[-1]["epoch"]} of 120 epochs; val_loss min ep'
                f'{selection.get("val_loss_min_epoch")}; encoder drift '
                f'{summary.get("encoder_relative_l2_change", float("nan")) * 100:.2f}%',
                summary.get("overfit_stop_epoch"))
        # Round-4 counterpart over the same seeds, for a like-for-like row.
        ba4, auc4, memo4, sel4 = [], [], [], []
        for seed in seeds:
            path = spec["round4"] / f"finetune_seed_{seed}_summary.json"
            if not path.is_file():
                continue
            previous = json.loads(path.read_text())
            ba4.append(previous.get("metrics_at_half", previous["metrics"])["volume_level"]["balanced_accuracy"])
            auc4.append(previous["metrics"]["volume_level"]["roc_auc"])
            sel4.append(previous["selection"]["epoch"])
            memo4.append(next((e["epoch"] for e in previous["history"] if e["train_loss"] < 1e-4), None))
        rows.append((name, spec["dataset"], seeds, fmt(ba4), fmt(ba5), fmt(auc4), fmt(auc5)))
        detail.append((name, sel4, sel, memo4, memo, ran, ofs, fmt(baf5)))
        print(f"{name}: seeds {seeds} done")

    if not rows:
        print("nothing to report yet")
        return

    lines = [
        "# Round 5 -- lowered learning rates, longer budget (B cells only)\n",
        "Same data, manifests, splits and feature caches as every earlier round. The A cells "
        "were **not** rerun: round-5 B warm-starts from the round-4 A checkpoints, so the only "
        "difference from round 4 is B's optimisation schedule.\n",
        "| | round 4 | round 5 |", "|---|---|---|",
        "| encoder_learning_rate | 1e-4 | **3e-5** |",
        "| head_learning_rate | 1e-3 | **3e-4** |",
        "| max_epochs | 40 | **120** |",
        "| patience / select_guard | 15 | **40** |",
        "| overfit_stop_window | -- | **10** |\n",
        "Both rates had to move together. In round 4 the training loss was already "
        "0.0096-0.2569 at epochs 1-2 *while the encoder was still frozen at lr=0*: the "
        "33,539-parameter head, warm-started from a converged probe, fits the training split on "
        "its own within two epochs, and once it does the loss gradient vanishes for the encoder "
        "too. Lowering only the encoder rate would have left memorisation at epoch ~18 exactly "
        "where it was.\n",
        "## Headline: balanced accuracy at the fixed 0.5 threshold, and AUC\n",
        "| cell | dataset | seeds | round4 BA@0.5 | **round5 BA@0.5** | round4 AUC | **round5 AUC** |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for name, dataset, seeds, ba4, ba5, auc4, auc5 in rows:
        lines.append(f"| {name} | {dataset} | {seeds} | {ba4} | **{ba5}** | {auc4} | **{auc5}** |")

    lines += ["\n## Did the longer, gentler schedule actually delay memorisation?\n",
              "`train_loss~0` is the first epoch whose training loss falls below 1e-4. In round 4 "
              "that happened at epoch 17-20 in every run, which is what this round set out to "
              "push later. If it moved and peak AUC did not, the ceiling is the training-set "
              "size, not the optimiser.\n",
              "| cell | round4 train_loss~0 | round5 train_loss~0 | round4 selected | "
              "round5 selected | round5 epochs run | overfit stop fired |",
              "|---|---|---|---|---|---|---|"]
    for name, sel4, sel5, memo4, memo5, ran, ofs, _ in detail:
        lines.append(f"| {name} | {memo4} | {memo5} | {sel4} | {sel5} | {ran} | {ofs} |")

    lines += ["\nSecondary (validation-fitted threshold, kept for continuity with round 4):\n",
              "| cell | round5 BA @fitted |", "|---|---:|"]
    for name, *_rest, baf5 in detail:
        lines.append(f"| {name} | {baf5} |")

    lines += ["\n## Curves\n",
              "One plot per cell per seed in `plots_round5/`. Four series: training loss and "
              "validation loss on the left scale, validation AUC and validation balanced "
              "accuracy (at 0.5) on a fixed 0-1 right scale. Four vertical rules: the selected "
              "epoch, the validation-loss minimum, the epoch where training loss reaches ~0, and "
              "where the overfit stop fired if it did.\n"]
    for name, *_ in detail:
        for path in sorted(PLOT_DIR.glob(f"{name}_round5_seed*.svg")):
            lines.append(f"- `plots_round5/{path.name}`")
        lines.append("")

    (OUT_DIR / "scaling_round5_report.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_DIR / 'scaling_round5_report.md'} and {len(list(PLOT_DIR.glob('*.svg')))} plots")


if __name__ == "__main__":
    main()
