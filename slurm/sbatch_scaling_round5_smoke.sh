#!/bin/bash
#SBATCH --job-name=round5_smoke
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --exclude=j005-ds
#SBATCH --output=/net/projects2/litian-lab/scpan/logs/round5_smoke_%j.log
#
# Exercises the round-5 finetune path end to end on the real dataset2 volumes
# before the full rerun is allowed to start. The login node has no GPU, so the
# probe half of round 3 could be smoke-tested locally but the finetune half --
# which is where the round-5 changes are riskiest (deferred selection reading
# per-epoch checkpoints off disk, then deleting them; the validation-fitted
# threshold; label smoothing through the autocast path) -- could not.
#
# Six epochs, one seed, into a scratch directory that is deleted afterwards.
# The full rerun is submitted with --dependency=afterok on this job, so a
# failure here stops the rerun instead of burning hours discovering it.
set -uo pipefail
cd /net/projects2/litian-lab/scpan/encoders
source /opt/conda/etc/profile.d/conda.sh
conda activate /net/projects2/litian-lab/scpan/encoders/med310
export PYTHONUNBUFFERED=1
export PYTHONPATH=/net/projects2/litian-lab/scpan/encoders/src:${PYTHONPATH:-}

echo "=== round-5 smoke on $(hostname) at $(date -u) job=$SLURM_JOB_ID ==="
nvidia-smi -L

SCRATCH=$(mktemp -d /net/projects2/litian-lab/scpan/encoders/artifacts/.round5_smoke_XXXXXX)
trap 'rm -rf "$SCRATCH"' EXIT

python - "$SCRATCH" <<'PY'
import copy, glob, os, sys
from encoderbench.config import load_config
from encoderbench.finetune import run_finetune

out = sys.argv[1]
config = copy.deepcopy(load_config("config/adni_mass_scaling_exp_round5.yaml").raw)
# Short budget; the point is to exercise every branch, not to converge.
config["finetune"].update({"max_epochs": 8, "patience": 5, "select_guard": 5,
                           "encoder_freeze_epochs": 2, "overfit_stop_window": 3})
config["evaluation"]["bootstrap_samples"] = 50

result = run_finetune("data/manifests/adni_full_mass_d2_reshuffled_ss.csv",
                      "artifacts/cache/ad/mass_native_d2ss70.npz", "ad", "mass", config, out,
                      seed=0, device="cuda", layer=3, native_tokens=True,
                      warm_start_dir="artifacts/attention/ad/improve4_d2ss70")

selection, history = result["selection"], result["history"]
print("\n=== round-5 finetune smoke ===")
print("selected epoch:", selection["epoch"], "|", selection["selection_rule"])
print("threshold fitted on validation:", round(result["decision_threshold"], 4))
print("test BA @fitted", round(result["metrics"]["volume_level"]["balanced_accuracy"], 3),
      "| @0.5", round(result["metrics_at_half"]["volume_level"]["balanced_accuracy"], 3))
print("encoder relative L2 change: %.3f%%" % (result["encoder_relative_l2_change"] * 100))
print("warm-start baseline val AUC:", round(result["warm_start_baseline"]["val_auc"], 4))
print("min train_loss (label-smoothing floor):", round(min(e["train_loss"] for e in history), 4))

failures = []
if selection["epoch"] > selection["val_loss_min_epoch"] + 5:
    failures.append("guard violated")
if not all("val_probability" in entry for entry in history):
    failures.append("validation probability trace missing")
if len(history[0]["val_probability"]) != 65:
    failures.append(f"trace has {len(history[0]['val_probability'])} subjects, expected 65")
leftover = glob.glob(os.path.join(out, "*_epoch*.pt"))
if leftover:
    failures.append(f"{len(leftover)} per-epoch checkpoints were not cleaned up")
if not all("test_probability_diagnostic_only" in entry for entry in history):
    failures.append("diagnostic test trace missing")
if result["metrics"]["decision_threshold"] == result["metrics_at_half"]["decision_threshold"]:
    print("note: fitted threshold happened to equal 0.5 for this seed")

if failures:
    print("SMOKE FAILED: " + "; ".join(failures))
    raise SystemExit(1)
print("encoder_lr at last epoch: %.2e | head_lr: %.2e"
      % (history[-1]["encoder_lr"], history[-1]["head_lr"]))
print("overfit_stop_epoch:", result.get("overfit_stop_epoch"))
if abs(history[-1]["head_lr"]) > 3.1e-4:
    print("SMOKE FAILED: head_lr was not lowered")
    raise SystemExit(1)
print("ROUND-5 SMOKE OK")
PY
STATUS=$?

if [ "$STATUS" -ne 0 ]; then
  echo "=== round-5 smoke FAILED (exit $STATUS); the dependent rerun will not start ===" >&2
  exit "$STATUS"
fi
echo "=== round-5 smoke passed at $(date -u) ==="
