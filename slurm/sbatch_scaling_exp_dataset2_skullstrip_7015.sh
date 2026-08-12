#!/bin/bash
#SBATCH --job-name=scaling_exp_d2ss70
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --exclude=j005-ds
#SBATCH --output=/net/projects2/litian-lab/scpan/logs/scaling_exp_d2ss70_%j.log
#
# A2ss70/B2ss70: re-runs A2ss/B2ss on the SAME 431 dataset2 subjects and the
# SAME already HD-BET-skull-stripped + MASS-preprocessed volumes (no
# reprocessing -- reuses the shared ADNI_full_skullstripped/ADNI_full_
# preprocessed_ss output dirs untouched), but with dataset2 reshuffled to
# ~70/15/15 (data/manifests/adni_full_mass_d2_reshuffled_ss.csv, already built
# for the A4/B4 merge: test_fraction=0.15, validation_fraction=0.1765) instead
# of A2ss/B2ss's original test_fraction=0.5 split.
#
# Purpose: A4/B4 changed two things at once relative to A2ss/B2ss -- more
# subjects (819 vs 431) AND a different split ratio (70/15/15 vs ~42/50/8).
# This cell holds subject count fixed at dataset2's 431 and only changes the
# split ratio, isolating how much of A4's jump over A2ss is "split ratio"
# vs "more training data by adding the new 3T ADNI_add_full subjects".
#
# B2ss70 uses encoder_learning_rate=5e-4, same as B3/B4 (config/
# adni_mass_scaling_exp.yaml's shared default, unchanged).
#
# The feature cache bakes in the split (encoderbench.cache.FeatureCache.
# splits), so A2ss/B2ss's existing cache (mass_native_d2_ss.npz) can't be
# reused as-is -- a fresh cache keyed to the reshuffled manifest is built
# below, but no image reprocessing is needed since the manifest already
# points at existing preprocessed files.
#
# Self-chaining, same pattern as the other scaling sbatch scripts.
set -uo pipefail
cd /net/projects2/litian-lab/scpan/encoders
source /opt/conda/etc/profile.d/conda.sh
conda activate /net/projects2/litian-lab/scpan/encoders/med310
export PYTHONUNBUFFERED=1
export PYTHONPATH=/net/projects2/litian-lab/scpan/encoders/src:${PYTHONPATH:-}

SCRIPT_PATH=/net/projects2/litian-lab/scpan/encoders/slurm/sbatch_scaling_exp_dataset2_skullstrip_7015.sh
MAX_HOPS=20
HOP_FILE=/net/projects2/litian-lab/scpan/logs/.scaling_exp_d2ss70_hops
HOP=$(cat "$HOP_FILE" 2>/dev/null || echo 0)

FINAL_MANIFEST=data/manifests/adni_full_mass_d2_reshuffled_ss.csv
CONFIG=config/adni_mass_scaling_exp.yaml
CACHE=artifacts/cache/ad/mass_native_d2ss70.npz
A_DIR=artifacts/attention/ad/mass_native_d2ss70
B_DIR=artifacts/finetune/ad/mass_native_d2ss70_warmstart

echo "=== hop $HOP started on $(hostname) at $(date -u) job=$SLURM_JOB_ID ==="
nvidia-smi -L

STATUS=0
echo "--- caching MASS native-token features (layer 3), dataset2 reshuffled 70/15/15 ---"
if [ -f "$CACHE" ]; then
  echo "--- cache already exists, reusing ---"
else
  encoderbench --config "$CONFIG" cache ad mass --manifest "$FINAL_MANIFEST" \
    --layer 3 --native-tokens --device cuda --out "$CACHE" || STATUS=1
fi

if [ "$STATUS" -eq 0 ]; then
  A_DONE=1
  for SEED in 0 1 2; do
    [ -f "$A_DIR/probe_seed_${SEED}_summary.json" ] || A_DONE=0
  done
  if [ "$A_DONE" -eq 1 ]; then
    echo "--- A2ss70 already complete, reusing ---"
  else
    echo "--- A2ss70: frozen probe (native tokens, dataset2 @ 70/15/15, 300 epochs) ---"
    encoderbench --config "$CONFIG" probe ad mass --cache "$CACHE" \
      --seeds all --device cuda --out-dir "$A_DIR" || STATUS=1
  fi
fi

if [ "$STATUS" -eq 0 ]; then
  echo "--- B2ss70: joint finetune, head warm-started from A2ss70 (native tokens, 30 epochs, encoder_lr=5e-4) ---"
  encoderbench --config "$CONFIG" finetune ad mass --manifest "$FINAL_MANIFEST" \
    --cache "$CACHE" --layer 3 --native-tokens --warm-start-dir "$A_DIR" \
    --seeds all --device cuda --out-dir "$B_DIR"
  STATUS=$?
fi

B_DONE=1
for SEED in 0 1 2; do
  [ -f "$B_DIR/finetune_seed_${SEED}_summary.json" ] || B_DONE=0
done

if [ "$STATUS" -eq 0 ] && [ "$B_DONE" -eq 1 ]; then
  echo "--- regenerating scaling_exp_report.md with A2ss70/B2ss70 ---"
  python scripts/generate_scaling_exp_report.py && python scripts/append_multiclass_to_scaling_report.py
  echo "=== dataset2-skullstrip-7015 (A2ss70+B2ss70) complete at $(date -u) ==="
  rm -f "$HOP_FILE"
  exit 0
fi

NEXT_HOP=$((HOP + 1))
if [ "$NEXT_HOP" -ge "$MAX_HOPS" ]; then
  echo "ERROR: hit $MAX_HOPS resubmission hops without finishing -- stopping, needs human attention" >&2
  exit 1
fi
echo "$NEXT_HOP" > "$HOP_FILE"
echo "=== not finished (exit $STATUS, B done=$B_DONE), self-resubmitting (hop $NEXT_HOP) ==="
sbatch --dependency=afterany:$SLURM_JOB_ID "$SCRIPT_PATH"
