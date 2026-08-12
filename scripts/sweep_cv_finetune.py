#!/usr/bin/env python
"""Finetune hyperparameter sweep for the repeated-split MASS/bsnip2 re-run.

Selection is by **validation** balanced accuracy only (the `selection` block each
run writes), never by test -- the whole point of the repeated-split protocol is
that test is scored once per repeat and never steers a choice.

Which knobs, and why these: the three failure modes diagnosed on the previous
(frozen-split) run were
  * mode 4 barely trained at all -- it early-stopped at epoch 1-3 with an
    identical validation BA across all three seeds, i.e. with the head frozen at
    its warm start and the encoder on lr 1e-5 the predictions hardly moved.
    -> sweep `encoder_learning_rate` upward.
  * mode 6 diverged -- validation loss was literally `inf` for all three seeds,
    a ~1e6-parameter linear head trained by plain SGD on a few hundred volumes
    with only the encoder's weight decay applied to it.
    -> sweep the new `head_weight_decay` (and a lower `head_learning_rate`).
  * modes 3/5 showed validation-minus-test gaps of 0.05-0.08 -- selection noise
    from a 38-sample validation split. Already addressed by the protocol change
    (72-sample validation, 5 repeats), not by a hyperparameter.

Sweeps run on mode 4 (isolates encoder_learning_rate: head is frozen, so the
encoder is the only thing moving) and mode 6 (isolates head regularization: the
head is the thing that blew up), over 2 repeats each to keep the cost bounded.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_cv_bsnip2_mass import MODES, OUT_ROOT, TUNED_PATH, run_mode  # noqa: E402

SWEEP_REPEATS = [0, 1]

# Ordered so the cheapest hypothesis (just raise the encoder lr) is tested first.
CONFIGS = {
    "base": {},
    "enc3e5": {"encoder_learning_rate": 3e-5},
    "enc1e4": {"encoder_learning_rate": 1e-4},
    "enc1e4_hwd1": {"encoder_learning_rate": 1e-4, "head_weight_decay": 1.0},
    "enc1e4_hwd10_hlr1e4": {"encoder_learning_rate": 1e-4, "head_weight_decay": 10.0,
                            "head_learning_rate": 1e-4},
    "enc3e4_hwd1": {"encoder_learning_rate": 3e-4, "head_weight_decay": 1.0},
    # --- round 2 -------------------------------------------------------------
    # Round 1 fixed *what* the optimizer does (encoder lr, head decay) but not
    # *how many times* it does it. With gradient_accumulation=8 over 284 training
    # volumes there are only ceil(284/8)=36 optimizer steps per epoch, so the
    # 12-epoch budget is 432 steps total -- for mode 6's ~2.1e6-parameter linear
    # head initialised at zero, that is nowhere near enough to approach the ridge
    # optimum mode 2 obtains in closed form, which is exactly the shape of the
    # 6-vs-2 deficit. Accumulating over fewer micro-batches costs essentially
    # nothing (the number of forward/backward passes per epoch is unchanged; only
    # the number of optimizer steps rises), so these configs buy 4x the steps for
    # free and optionally double the epoch budget on top.
    "steps4x": {"encoder_learning_rate": 1e-4, "head_weight_decay": 1.0,
                "gradient_accumulation": 2},
    "steps4x_ep24_hlr3e3": {"encoder_learning_rate": 1e-4, "head_weight_decay": 1.0,
                            "gradient_accumulation": 2, "max_epochs": 24,
                            "head_learning_rate": 3e-3},
    "steps4x_ep24_hwd10_hlr1e2": {"encoder_learning_rate": 1e-4, "head_weight_decay": 10.0,
                                  "gradient_accumulation": 2, "max_epochs": 24,
                                  "head_learning_rate": 1e-2},
}


def _validation_scores(mode: int, variant: str) -> tuple[list[float], list[float]]:
    """(validation BA, test BA) per repeat. Test is collected only so the report
    can show it afterwards -- it is never used to pick the winner."""
    spec = MODES[mode]
    head = spec["head"]
    prefix = (f"finetune_{head}_{'trainhead' if spec['train_head'] else 'headfrozen'}_"
              f"{'warm' if spec['warm'] else 'cold'}")
    val, test = [], []
    for repeat in SWEEP_REPEATS:
        path = OUT_ROOT / f"mode{mode}_{variant}" / f"{prefix}_seed_0_rep{repeat}_summary.json"
        if path.is_file():
            data = json.loads(path.read_text())
            val.append(data["selection"]["balanced_accuracy"])
            test.append(data["metrics"]["volume_level"]["balanced_accuracy"])
    return val, test


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default="4,6")
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--collect-only", action="store_true")
    args = ap.parse_args()

    modes = [int(v) for v in args.modes.split(",")]
    names = [n.strip() for n in args.configs.split(",") if n.strip()]

    if not args.collect_only:
        for name in names:
            for mode in modes:
                print(f"\n##### sweep config={name} mode={mode} #####", flush=True)
                try:
                    run_mode(mode, SWEEP_REPEATS, settings_override=CONFIGS[name],
                             variant=f"sweep_{name}")
                except Exception as exc:  # keep sweeping; a diverged config is a result
                    print(f"!!! config={name} mode={mode} failed: {type(exc).__name__}: {exc}",
                          flush=True)

    print("\n===== sweep summary (winner chosen by VALIDATION BA) =====", flush=True)
    table = {}
    for mode in modes:
        for name in names:
            val, test = _validation_scores(mode, f"sweep_{name}")
            if val:
                table[(mode, name)] = (statistics.mean(val), statistics.mean(test), len(val))
                print(f"mode{mode:<2} {name:<22} val BA {statistics.mean(val):.4f}  "
                      f"(test BA {statistics.mean(test):.4f}, n={len(val)})", flush=True)
            else:
                print(f"mode{mode:<2} {name:<22} no completed repeats", flush=True)

    # One shared setting for all four finetune modes: sum validation BA across the
    # swept modes so the choice is not driven by whichever mode happens to be noisier.
    totals = {}
    for name in names:
        scores = [table[(mode, name)][0] for mode in modes if (mode, name) in table]
        if len(scores) == len(modes):
            totals[name] = sum(scores)
    if totals:
        winner = max(totals, key=totals.get)
        payload = {"chosen": CONFIGS[winner], "chosen_name": winner,
                   "selected_by": "sum of mean validation BA over swept modes",
                   "swept_modes": modes, "repeats": SWEEP_REPEATS,
                   "validation_ba": {f"mode{m}/{n}": table[(m, n)][0] for (m, n) in table},
                   "test_ba_not_used_for_selection": {f"mode{m}/{n}": table[(m, n)][1]
                                                      for (m, n) in table}}
        TUNED_PATH.parent.mkdir(parents=True, exist_ok=True)
        TUNED_PATH.write_text(json.dumps(payload, indent=2))
        print(f"\nwinner: {winner} -> {CONFIGS[winner]}")
        print(f"written to {TUNED_PATH}")
    else:
        print("\nno config completed every swept mode; not writing a winner")


if __name__ == "__main__":
    main()
