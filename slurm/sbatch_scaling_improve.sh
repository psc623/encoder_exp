#!/bin/bash
#SBATCH --job-name=scaling_improve
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --exclude=j005-ds
#SBATCH --output=/net/projects2/litian-lab/scpan/logs/scaling_improve_%j.log
#
# Improved-protocol rerun of A2ss70 / B2ss70 / A4 / B4.
#
# DATA IS UNCHANGED. Same manifests, same frozen splits, same already-built
# feature caches as the original report/scaling_exp run -- no preprocessing, no
# skull-stripping, no re-caching happens here. Only the training and epoch-
# selection protocol changed, via config/adni_mass_scaling_exp_improve.yaml
# (which documents each change together with the measurement that motivated
# it) and src/encoderbench/selection.py (the shared selection rule).
#
# Results go to fresh artifact directories (improve_*) so the original run
# stays on disk untouched and the two protocols can be compared directly.
#
# Recap of what changed:
#   selection : val AUC, trailing 3-epoch mean, nothing past val_loss min +
#               select_guard. Replaces argmax of raw val balanced accuracy,
#               which was picking epoch 279 when val_loss bottomed at epoch 15.
#   probe     : early stopping (patience 50 on val_loss) instead of a fixed
#               300 epochs; weight-decay grid 5 -> 3 values; training loss is
#               now recorded per epoch (it never was before, so the A curves
#               could not show memorisation).
#   finetune  : encoder_lr 5e-4 -> 1e-4 plus 3 head-only warm-up epochs
#               (5e-4 moved the encoder 28.8-38.7% in relative L2 and wiped out
#               the warm start inside epoch 1); gradient_accumulation 8 -> 16;
#               max_epochs 30 -> 40; val_loss clamped so saturation no longer
#               reads as `inf`; epoch-0 warm-start baseline and final encoder
#               drift both recorded into the summary JSON.
#
# Self-chaining, same pattern as the other scaling sbatch scripts.
set -uo pipefail
cd /net/projects2/litian-lab/scpan/encoders
source /opt/conda/etc/profile.d/conda.sh
conda activate /net/projects2/litian-lab/scpan/encoders/med310
export PYTHONUNBUFFERED=1
export PYTHONPATH=/net/projects2/litian-lab/scpan/encoders/src:${PYTHONPATH:-}

SCRIPT_PATH=/net/projects2/litian-lab/scpan/encoders/slurm/sbatch_scaling_improve.sh
MAX_HOPS=25
HOP_FILE=/net/projects2/litian-lab/scpan/logs/.scaling_improve_hops
HOP=$(cat "$HOP_FILE" 2>/dev/null || echo 0)

CONFIG=config/adni_mass_scaling_exp_improve.yaml

D2_MANIFEST=data/manifests/adni_full_mass_d2_reshuffled_ss.csv
D2_CACHE=artifacts/cache/ad/mass_native_d2ss70.npz
D2_A_DIR=artifacts/attention/ad/improve_d2ss70
D2_B_DIR=artifacts/finetune/ad/improve_d2ss70_warmstart

D4_MANIFEST=data/manifests/adni_full_mass4_ss.csv
D4_CACHE=artifacts/cache/ad/mass_native_d4.npz
D4_A_DIR=artifacts/attention/ad/improve_d4
D4_B_DIR=artifacts/finetune/ad/improve_d4_warmstart

echo "=== hop $HOP started on $(hostname) at $(date -u) job=$SLURM_JOB_ID ==="
nvidia-smi -L

# Fail loudly rather than silently rebuilding: this rerun is only meaningful if
# it reads exactly the data the original run did.
for REQUIRED in "$D2_MANIFEST" "$D2_CACHE" "$D4_MANIFEST" "$D4_CACHE"; do
  if [ ! -f "$REQUIRED" ]; then
    echo "ERROR: required input missing, refusing to rebuild it: $REQUIRED" >&2
    exit 1
  fi
done
echo "--- reusing existing manifests and feature caches (data unchanged) ---"
ls -la "$D2_CACHE" "$D4_CACHE"

STATUS=0

run_probe_cell () {
  local NAME=$1 CACHE=$2 DIR=$3
  local DONE=1
  for SEED in 0 1 2; do
    [ -f "$DIR/probe_seed_${SEED}_summary.json" ] || DONE=0
  done
  if [ "$DONE" -eq 1 ]; then
    echo "--- $NAME already complete, reusing ---"
    return 0
  fi
  echo "--- $NAME: frozen probe, improved protocol (wd grid x3, patience 50, AUC selection) ---"
  encoderbench --config "$CONFIG" probe ad mass --cache "$CACHE" \
    --seeds all --device cuda --out-dir "$DIR"
}

run_finetune_cell () {
  local NAME=$1 MANIFEST=$2 CACHE=$3 A_DIR=$4 B_DIR=$5
  local DONE=1
  for SEED in 0 1 2; do
    [ -f "$B_DIR/finetune_seed_${SEED}_summary.json" ] || DONE=0
  done
  if [ "$DONE" -eq 1 ]; then
    echo "--- $NAME already complete, reusing ---"
    return 0
  fi
  echo "--- $NAME: joint finetune, encoder_lr=1e-4 + 3 head-only warm-up epochs, warm-started from $A_DIR ---"
  encoderbench --config "$CONFIG" finetune ad mass --manifest "$MANIFEST" \
    --cache "$CACHE" --layer 3 --native-tokens --warm-start-dir "$A_DIR" \
    --seeds all --device cuda --out-dir "$B_DIR"
}

[ "$STATUS" -eq 0 ] && { run_probe_cell    "A2ss70" "$D2_CACHE" "$D2_A_DIR" || STATUS=1; }
[ "$STATUS" -eq 0 ] && { run_finetune_cell "B2ss70" "$D2_MANIFEST" "$D2_CACHE" "$D2_A_DIR" "$D2_B_DIR" || STATUS=1; }
[ "$STATUS" -eq 0 ] && { run_probe_cell    "A4"     "$D4_CACHE" "$D4_A_DIR" || STATUS=1; }
[ "$STATUS" -eq 0 ] && { run_finetune_cell "B4"     "$D4_MANIFEST" "$D4_CACHE" "$D4_A_DIR" "$D4_B_DIR" || STATUS=1; }

ALL_DONE=1
for SEED in 0 1 2; do
  [ -f "$D2_A_DIR/probe_seed_${SEED}_summary.json" ]    || ALL_DONE=0
  [ -f "$D2_B_DIR/finetune_seed_${SEED}_summary.json" ] || ALL_DONE=0
  [ -f "$D4_A_DIR/probe_seed_${SEED}_summary.json" ]    || ALL_DONE=0
  [ -f "$D4_B_DIR/finetune_seed_${SEED}_summary.json" ] || ALL_DONE=0
done

if [ "$STATUS" -eq 0 ] && [ "$ALL_DONE" -eq 1 ]; then
  echo "--- generating report/scaling_exp_improve/scaling_improve_report.md ---"
  python scripts/generate_scaling_improve_report.py
  echo "=== improved-protocol rerun complete at $(date -u) ==="
  rm -f "$HOP_FILE"
  exit 0
fi

NEXT_HOP=$((HOP + 1))
if [ "$NEXT_HOP" -ge "$MAX_HOPS" ]; then
  echo "ERROR: hit $MAX_HOPS resubmission hops without finishing -- stopping, needs human attention" >&2
  exit 1
fi
echo "$NEXT_HOP" > "$HOP_FILE"
echo "=== not finished (exit $STATUS, all done=$ALL_DONE), self-resubmitting (hop $NEXT_HOP) ==="
sbatch --dependency=afterany:$SLURM_JOB_ID "$SCRIPT_PATH"
