#!/usr/bin/env python
"""Append the A2mc/B2mc (3-class CN/MCI/AD) results to
report/scaling_exp/scaling_exp_report.md as a clearly separate section --
not numerically comparable to the binary A/B cells above it. Reuses
plot_history from generate_scaling_exp_report.py unchanged.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_scaling_exp_report import plot_history  # noqa: E402

ROOT = Path("/net/projects2/litian-lab/scpan/encoders")
OUT_DIR = ROOT / "report" / "scaling_exp"
CELLS = {
    "A2mc": {"kind": "probe", "dir": ROOT / "artifacts/attention/ad3/mass_native_d2_mc"},
    "B2mc": {"kind": "finetune", "dir": ROOT / "artifacts/finetune/ad3/mass_native_d2_mc_warmstart"},
}
CLASSES = ("CN", "MCI", "AD")
SEEDS = (0, 1, 2)


def fmt(values: list[float]) -> str:
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{mean:.3f} ± {sd:.3f}"


def main() -> None:
    plot_dir = OUT_DIR / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, spec in CELLS.items():
        prefix = "probe" if spec["kind"] == "probe" else "finetune"
        summaries = [json.loads((spec["dir"] / f"{prefix}_seed_{s}_summary.json").read_text())
                    for s in SEEDS]
        for summary in summaries:
            seed = summary["seed"]
            history = summary.get("history", [])
            if spec["kind"] == "probe":
                best_wd = summary["selection"].get("weight_decay")
                history = [entry for entry in history if entry.get("weight_decay") == best_wd]
            plot_history(history, f"{name} seed={seed} (3-class CN/MCI/AD)",
                        plot_dir / f"{name}_seed{seed}.svg", summary["selection"].get("epoch"))
        vba = [s["metrics"]["volume_level"]["balanced_accuracy"] for s in summaries]
        recalls = {cls: [s["metrics"]["volume_level"]["per_class"][cls]["recall"] for s in summaries]
                  for cls in CLASSES}
        results[name] = {"summaries": summaries, "volume_ba": fmt(vba), "recalls": recalls}

    a, b = results["A2mc"], results["B2mc"]
    recall_rows = "\n".join(
        f"| {cls} | {fmt(a['recalls'][cls])} | {fmt(b['recalls'][cls])} |" for cls in CLASSES
    )
    warm_starts = [s.get("head_warm_started_from") for s in b["summaries"]]
    epochs_a = [s["selection"]["epoch"] for s in a["summaries"]]
    epochs_b = [s["selection"]["epoch"] for s in b["summaries"]]

    section = f"""

---

## A2mc/B2mc -- 3-class (CN/MCI/AD), dataset2, NOT numerically comparable to A/B above

MCI added back into dataset2 (screening-only, HD-BET skull-stripped, same
recipe as A2ss/B2ss); 431 already-processed CN/AD screening volumes reused
unchanged, ~411 new MCI screening volumes processed the same way (842
subjects total: 231 CN / 411 MCI / 200 AD). New standalone scripts (3-class
metrics/head machinery didn't exist before this): `run_probe_adni_mass_
multiclass.py` (A setting -- frozen probe, 300-epoch weight-decay grid,
mirrors `training.run_probe`) and `run_finetune_adni_mass_multiclass.py`
(B setting -- warm-started joint finetune, mirrors `finetune.run_finetune`).
`encoder_learning_rate` deliberately kept at the *original* B2 value (1e-5,
not B3's 5e-4 fix) per direct instruction -- everything else held equal to
A2ss/B2ss except the label scheme and MCI inclusion, so this isolates "does
adding MCI help" from "does the higher LR help" (a follow-up combining both
was declined; 3-class work stops here).

This is a genuinely different task (3-way macro-balanced-accuracy) from every
A/B cell above (binary), so the numbers are not on the same scale and should
not be read as "better/worse" than the AD-vs-CN cells.

| cell | volume BA | selected epochs |
|---|---:|---|
| A2mc (frozen probe) | {a['volume_ba']} | {epochs_a} |
| B2mc (warm-started joint finetune, encoder_lr=1e-5) | {b['volume_ba']} | {epochs_b} |

### Per-class recall (volume level, mean ± sample SD)

| class | A2mc | B2mc |
|---|---:|---:|
{recall_rows}

MCI is the clear weak point in both settings (A2mc seed 1: MCI recall
exactly 0 -- the model never predicted MCI at all, falling back to CN/AD
only). This tracks the wider AD-imaging literature: MCI is a heterogeneous,
boundary-blurred transitional state rather than a fixed pathology, and is
consistently the hardest of the three classes to separate on structural MRI
alone. B2mc raises MCI recall substantially over A2mc in every seed (mean
{fmt(b['recalls']['MCI'])} vs {fmt(a['recalls']['MCI'])}) but at the cost of
AD recall in most seeds -- overall balanced accuracy stays flat
({a['volume_ba']} -> {b['volume_ba']}), consistent with the encoder barely
moving at this learning rate (same mechanism diagnosed for B1/B2/B2ss)
redistributing the decision boundary rather than genuinely improving it.

Head warm-started from: {warm_starts}.

### Per-seed curves

- A2mc: `plots/A2mc_seed0.svg`, `plots/A2mc_seed1.svg`, `plots/A2mc_seed2.svg`
- B2mc: `plots/B2mc_seed0.svg`, `plots/B2mc_seed1.svg`, `plots/B2mc_seed2.svg`

Artifacts: `artifacts/attention/ad3/mass_native_d2_mc/probe_seed_*_summary.json`,
`artifacts/finetune/ad3/mass_native_d2_mc_warmstart/finetune_seed_*_summary.json`,
`artifacts/cache/ad3/mass_native_d2_mc.npz`.
"""
    report_path = OUT_DIR / "scaling_exp_report.md"
    with report_path.open("a") as handle:
        handle.write(section)
    print(f"appended A2mc/B2mc section to {report_path}")


if __name__ == "__main__":
    main()
