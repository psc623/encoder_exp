#!/usr/bin/env python
"""Exit 0 if mode1 and mode2's mean test balanced accuracy are both >= 0.65,
exit 1 otherwise (signals sbatch_improved_bsnip2_mass.sh to run the sweep)."""
import json
import statistics
import sys
from pathlib import Path

ARTIFACTS = Path("/net/projects2/litian-lab/scpan/encoders/artifacts")
THRESHOLD = 0.65
CHECKS = [
    (ARTIFACTS / "attention" / "bsnip2" / "mass", "probe"),
    (ARTIFACTS / "linear" / "bsnip2" / "mass", "linear"),
]

ok = True
for mode_dir, prefix in CHECKS:
    bas = []
    for seed in (0, 1, 2):
        path = mode_dir / f"{prefix}_seed_{seed}_summary.json"
        if path.is_file():
            bas.append(json.loads(path.read_text())["metrics"]["volume_level"]["balanced_accuracy"])
    if not bas:
        print(f"{mode_dir}: no summaries found")
        ok = False
        continue
    mean_ba = statistics.mean(bas)
    print(f"{mode_dir}: mean BA over {len(bas)} seed(s) = {mean_ba:.4f}")
    if mean_ba < THRESHOLD:
        ok = False

sys.exit(0 if ok else 1)
