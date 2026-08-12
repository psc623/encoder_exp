#!/bin/bash
#SBATCH --job-name=scaling_round5
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --exclude=j005-ds
#SBATCH --output=/net/projects2/litian-lab/scpan/logs/scaling_round5_%j.log
#
# Round 5: B2ss70 and B4 only, seeds 0-4. The A cells are NOT rerun -- round-5 B
# warm-starts from the round-4 A checkpoints, so the only thing that changes is
# B's optimisation schedule and the comparison against round 4 is clean.
#
# DATA IS UNCHANGED: same manifests, same frozen splits, same feature caches as
# rounds 1-4.
#
# What changes (config/adni_mass_scaling_exp_round5.yaml carries the full
# rationale and the measurements behind it):
#   encoder_learning_rate  1e-4 -> 3e-5
#   head_learning_rate     1e-3 -> 3e-4   (same 3.3x, 10:1 ratio preserved)
#   max_epochs             40   -> 120
#   patience/select_guard  15   -> 40
#   overfit_stop_window    (new) 10
#
# Why both rates: round 4 reached training loss 0.0096-0.2569 at epochs 1-2
# while the encoder was still frozen at lr=0. The 33,539-parameter head,
# warm-started from a converged probe, fits 301 training volumes on its own
# within two epochs, and once it does the loss gradient vanishes for the encoder
# too. Lowering only the encoder rate would have left memorisation at epoch ~18
# untouched and measured nothing.
#
# Why more epochs: all ten round-4 B runs hit training loss exactly 0 by epoch
# 17-20 and stopped at 17-27 of a 40 cap, so the budget was never binding --
# memorisation was. At gradient_accumulation=16 that is ~19 optimiser steps per
# epoch, ~340 steps total.
#
# The overfit stop (training loss falling while validation loss rises, averaged
# over 10 epochs vs the 10 before) is a safety net, not the primary rule:
# replayed against all ten round-4 curves it fires at epoch 20, after every
# epoch that was actually selected (8-18).
#
# Self-chaining, same pattern as the other scaling sbatch scripts.
set -uo pipefail
cd /net/projects2/litian-lab/scpan/encoders
source /opt/conda/etc/profile.d/conda.sh
conda activate /net/projects2/litian-lab/scpan/encoders/med310
export PYTHONUNBUFFERED=1
export PYTHONPATH=/net/projects2/litian-lab/scpan/encoders/src:${PYTHONPATH:-}

SCRIPT_PATH=/net/projects2/litian-lab/scpan/encoders/slurm/sbatch_scaling_round5.sh
MAX_HOPS=25
HOP_FILE=/net/projects2/litian-lab/scpan/logs/.scaling_round5_hops
HOP=$(cat "$HOP_FILE" 2>/dev/null || echo 0)

CONFIG=config/adni_mass_scaling_exp_round5.yaml
D2_MANIFEST=data/manifests/adni_full_mass_d2_reshuffled_ss.csv
D2_CACHE=artifacts/cache/ad/mass_native_d2ss70.npz
D2_A_DIR=artifacts/attention/ad/improve4_d2ss70
D2_B_DIR=artifacts/finetune/ad/improve5_d2ss70_warmstart
D4_MANIFEST=data/manifests/adni_full_mass4_ss.csv
D4_CACHE=artifacts/cache/ad/mass_native_d4.npz
D4_A_DIR=artifacts/attention/ad/improve4_d4
D4_B_DIR=artifacts/finetune/ad/improve5_d4_warmstart

echo "=== hop $HOP started on $(hostname) at $(date -u) job=$SLURM_JOB_ID ==="
nvidia-smi -L

SEEDS=$(python - "$CONFIG" <<'PY'
import sys, yaml
print(",".join(str(s) for s in yaml.safe_load(open(sys.argv[1]))["finetune"]["seeds"]))
PY
) || { echo "ERROR: could not read the seed list from $CONFIG" >&2; exit 1; }
echo "--- seeds: $SEEDS ---"

for REQUIRED in "$D2_MANIFEST" "$D2_CACHE" "$D4_MANIFEST" "$D4_CACHE" "$D2_A_DIR" "$D4_A_DIR"; do
  if [ ! -e "$REQUIRED" ]; then
    echo "ERROR: required input missing, refusing to rebuild it: $REQUIRED" >&2
    exit 1
  fi
done
echo "--- reusing round-4 A checkpoints as warm start; data untouched ---"

missing_seeds () {
  local DIR=$1 OUT=""
  for SEED in ${SEEDS//,/ }; do
    [ -f "$DIR/finetune_seed_${SEED}_summary.json" ] || OUT="${OUT:+$OUT,}$SEED"
  done
  echo "$OUT"
}

STATUS=0
run_cell () {
  local NAME=$1 MANIFEST=$2 CACHE=$3 A_DIR=$4 B_DIR=$5 TODO
  TODO=$(missing_seeds "$B_DIR")
  if [ -z "$TODO" ]; then echo "--- $NAME: all seeds done ---"; return 0; fi
  echo "--- $NAME: seeds $TODO, encoder_lr=3e-5 head_lr=3e-4, up to 120 epochs, warm-started from $A_DIR ---"
  encoderbench --config "$CONFIG" finetune ad mass --manifest "$MANIFEST" \
    --cache "$CACHE" --layer 3 --native-tokens --warm-start-dir "$A_DIR" \
    --seeds "$TODO" --device cuda --out-dir "$B_DIR"
}

[ "$STATUS" -eq 0 ] && { run_cell "B2ss70" "$D2_MANIFEST" "$D2_CACHE" "$D2_A_DIR" "$D2_B_DIR" || STATUS=1; }
[ "$STATUS" -eq 0 ] && { run_cell "B4"     "$D4_MANIFEST" "$D4_CACHE" "$D4_A_DIR" "$D4_B_DIR" || STATUS=1; }

ALL_DONE=1
[ -n "$(missing_seeds "$D2_B_DIR")" ] && ALL_DONE=0
[ -n "$(missing_seeds "$D4_B_DIR")" ] && ALL_DONE=0

if [ "$STATUS" -eq 0 ] && [ "$ALL_DONE" -eq 1 ]; then
  echo "=== round-5 complete at $(date -u), seeds $SEEDS ==="
  rm -f "$HOP_FILE"
  exit 0
fi

NEXT_HOP=$((HOP + 1))
if [ "$NEXT_HOP" -ge "$MAX_HOPS" ]; then
  echo "ERROR: hit $MAX_HOPS hops without finishing -- needs human attention" >&2
  exit 1
fi
echo "$NEXT_HOP" > "$HOP_FILE"
echo "=== not finished (exit $STATUS, all done=$ALL_DONE), self-resubmitting (hop $NEXT_HOP) ==="
sbatch --dependency=afterany:$SLURM_JOB_ID "$SCRIPT_PATH"
