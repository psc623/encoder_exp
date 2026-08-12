#!/bin/bash
#SBATCH --job-name=scaling_exp_d2
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --exclude=j005-ds
#SBATCH --output=/net/projects2/litian-lab/scpan/logs/scaling_exp_d2_%j.log
#
# A2/B2 cells of the A/B x dataset1/dataset2 scaling comparison, on dataset 2
# (ADNI_full_screen: the 431-subject screening-only AD/CN cohort from
# ADNI_full, MASS-native preprocessed -- same data as the "Group 1" run in
# ADNI_MASS_scaling_report.md, whose from-scratch joint-training cell is what
# motivated this whole A/B comparison in the first place).
#
# Reuses the native-token cache already built for Group 1
# (artifacts/cache/ad/mass_full_native.npz) instead of rebuilding it.
#
# A2: frozen probe, 300 epochs.
# B2: encoder+head jointly finetuned, head warm-started from A2's own probe
#     (--warm-start-dir, isolated from both the historical ad/mass probe and
#     from A1/B1's dataset1 checkpoints).
#
# Self-chaining, same pattern as sbatch_scaling_exp_dataset1.sh.
set -uo pipefail
cd /net/projects2/litian-lab/scpan/encoders
source /opt/conda/etc/profile.d/conda.sh
conda activate /net/projects2/litian-lab/scpan/encoders/med310
export PYTHONUNBUFFERED=1

SCRIPT_PATH=/net/projects2/litian-lab/scpan/encoders/slurm/sbatch_scaling_exp_dataset2.sh
MAX_HOPS=20
HOP_FILE=/net/projects2/litian-lab/scpan/logs/.scaling_exp_d2_hops
HOP=$(cat "$HOP_FILE" 2>/dev/null || echo 0)

CONFIG=config/adni_mass_scaling_exp.yaml
MANIFEST=data/manifests/adni_full_mass.csv
CACHE=artifacts/cache/ad/mass_full_native.npz
A2_DIR=artifacts/attention/ad/mass_native_d2
B2_DIR=artifacts/finetune/ad/mass_native_d2_warmstart

echo "=== hop $HOP started on $(hostname) at $(date -u) job=$SLURM_JOB_ID ==="
nvidia-smi -L

if [ ! -f "$CACHE" ]; then
  echo "--- cache missing, building it ---"
  encoderbench --config "$CONFIG" cache ad mass --manifest "$MANIFEST" \
    --layer 3 --native-tokens --device cuda --out "$CACHE" || exit 1
else
  echo "--- reusing existing cache: $CACHE ---"
fi

A2_DONE=1
for SEED in 0 1 2; do
  [ -f "$A2_DIR/probe_seed_${SEED}_summary.json" ] || A2_DONE=0
done
if [ "$A2_DONE" -eq 1 ]; then
  echo "--- A2 already complete, reusing: $A2_DIR ---"
else
  echo "--- A2: frozen probe (native tokens, 300 epochs) ---"
  encoderbench --config "$CONFIG" probe ad mass --cache "$CACHE" \
    --seeds all --device cuda --out-dir "$A2_DIR" || exit 1
fi

echo "--- B2: joint finetune, head warm-started from A2 (native tokens, 30 epochs) ---"
encoderbench --config "$CONFIG" finetune ad mass --manifest "$MANIFEST" \
  --cache "$CACHE" --layer 3 --native-tokens --warm-start-dir "$A2_DIR" \
  --seeds all --device cuda --out-dir "$B2_DIR"
STATUS=$?

B2_DONE=1
for SEED in 0 1 2; do
  [ -f "$B2_DIR/finetune_seed_${SEED}_summary.json" ] || B2_DONE=0
done

if [ "$STATUS" -eq 0 ] && [ "$B2_DONE" -eq 1 ]; then
  echo "=== dataset2 (A2+B2) complete at $(date -u) ==="
  rm -f "$HOP_FILE"
  exit 0
fi

NEXT_HOP=$((HOP + 1))
if [ "$NEXT_HOP" -ge "$MAX_HOPS" ]; then
  echo "ERROR: hit $MAX_HOPS resubmission hops without finishing -- stopping, needs human attention" >&2
  exit 1
fi
echo "$NEXT_HOP" > "$HOP_FILE"
echo "=== not finished (finetune exit $STATUS, B2 done=$B2_DONE), self-resubmitting (hop $NEXT_HOP) ==="
sbatch --dependency=afterany:$SLURM_JOB_ID "$SCRIPT_PATH"
