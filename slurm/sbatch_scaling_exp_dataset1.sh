#!/bin/bash
#SBATCH --job-name=scaling_exp_d1
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --exclude=j005-ds
#SBATCH --output=/net/projects2/litian-lab/scpan/logs/scaling_exp_d1_%j.log
#
# A1/B1 cells of the A/B x dataset1/dataset2 scaling comparison, on dataset 1
# (ADNI_processed_clean, data/manifests/adni.csv, the original 548-volume/
# 416-subject frozen AD/CN cohort). Both cells now use native tokens (no
# 4x4x4 pooling) via config/adni_mass_scaling_exp.yaml, so this re-runs A1
# rather than reusing Group 2's pooled-grid probe number.
#
# A1: frozen probe, 300 epochs (cheap -- cached features, no encoder pass).
# B1: encoder+head jointly finetuned, head warm-started from A1's own probe
#     (via --warm-start-dir, not the default disease/encoder-keyed path, so
#     this never touches the historical artifacts/attention/ad/mass/ probe),
#     30 epochs / patience 15.
#
# Self-chaining like the other scaling sbatch scripts: resumes via
# finetune.py's own latest-checkpoint mechanism if this 12h window isn't
# enough, hop-capped at 20.
set -uo pipefail
cd /net/projects2/litian-lab/scpan/encoders
source /opt/conda/etc/profile.d/conda.sh
conda activate /net/projects2/litian-lab/scpan/encoders/med310
export PYTHONUNBUFFERED=1

SCRIPT_PATH=/net/projects2/litian-lab/scpan/encoders/slurm/sbatch_scaling_exp_dataset1.sh
MAX_HOPS=20
HOP_FILE=/net/projects2/litian-lab/scpan/logs/.scaling_exp_d1_hops
HOP=$(cat "$HOP_FILE" 2>/dev/null || echo 0)

CONFIG=config/adni_mass_scaling_exp.yaml
MANIFEST=data/manifests/adni.csv
CACHE=artifacts/cache/ad/mass_native_d1.npz
A1_DIR=artifacts/attention/ad/mass_native_d1
B1_DIR=artifacts/finetune/ad/mass_native_d1_warmstart

echo "=== hop $HOP started on $(hostname) at $(date -u) job=$SLURM_JOB_ID ==="
nvidia-smi -L

if [ -f "$CACHE" ]; then
  echo "--- cache already exists, reusing: $CACHE ---"
else
  echo "--- caching MASS native-token features (layer 3) for dataset1 ---"
  encoderbench --config "$CONFIG" cache ad mass --manifest "$MANIFEST" \
    --layer 3 --native-tokens --device cuda --out "$CACHE" || exit 1
fi

A1_DONE=1
for SEED in 0 1 2; do
  [ -f "$A1_DIR/probe_seed_${SEED}_summary.json" ] || A1_DONE=0
done
if [ "$A1_DONE" -eq 1 ]; then
  echo "--- A1 already complete, reusing: $A1_DIR ---"
else
  echo "--- A1: frozen probe (native tokens, 300 epochs) ---"
  encoderbench --config "$CONFIG" probe ad mass --cache "$CACHE" \
    --seeds all --device cuda --out-dir "$A1_DIR" || exit 1
fi

echo "--- B1: joint finetune, head warm-started from A1 (native tokens, 30 epochs) ---"
encoderbench --config "$CONFIG" finetune ad mass --manifest "$MANIFEST" \
  --cache "$CACHE" --layer 3 --native-tokens --warm-start-dir "$A1_DIR" \
  --seeds all --device cuda --out-dir "$B1_DIR"
STATUS=$?

B1_DONE=1
for SEED in 0 1 2; do
  [ -f "$B1_DIR/finetune_seed_${SEED}_summary.json" ] || B1_DONE=0
done

if [ "$STATUS" -eq 0 ] && [ "$B1_DONE" -eq 1 ]; then
  echo "=== dataset1 (A1+B1) complete at $(date -u) ==="
  rm -f "$HOP_FILE"
  exit 0
fi

NEXT_HOP=$((HOP + 1))
if [ "$NEXT_HOP" -ge "$MAX_HOPS" ]; then
  echo "ERROR: hit $MAX_HOPS resubmission hops without finishing -- stopping, needs human attention" >&2
  exit 1
fi
echo "$NEXT_HOP" > "$HOP_FILE"
echo "=== not finished (finetune exit $STATUS, B1 done=$B1_DONE), self-resubmitting (hop $NEXT_HOP) ==="
sbatch --dependency=afterany:$SLURM_JOB_ID "$SCRIPT_PATH"
