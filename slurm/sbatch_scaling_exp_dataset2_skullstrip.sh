#!/bin/bash
#SBATCH --job-name=scaling_exp_d2_ss
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --exclude=j005-ds
#SBATCH --output=/net/projects2/litian-lab/scpan/logs/scaling_exp_d2_ss_%j.log
#
# Re-does A2/B2 (dataset2 = ADNI_full_screen) with HD-BET skull-stripping
# added ahead of MASS's own reorient/resample/crop -- diagnosis (see
# report/scaling_exp/) found dataset2's frozen-probe ceiling (A2, no joint
# training instability possible) was already ~0.67 BA vs dataset1's ~0.80,
# and traced it to dataset2's un-skull-stripped field of view being 3.77x
# larger by volume than dataset1's (skull/neck/scalp diluting the brain's
# share of the fixed 128^3 input grid, plus contaminating MASS's own
# whole-array percentile-clip intensity normalization).
#
# Pipeline: raw ADNI_full images (data/manifests/adni_full_mass_raw.csv)
#   -> HD-BET skull-strip (already vendored + weight-cached in this repo,
#      reused unchanged from src/encoderbench/bsnip2/hdbet_batch.py, the same
#      code BrainIAC's and medsiglip's BSNIP2 pipelines already use)
#   -> MASS's own reorient(RAS)/resample(1.5mm) (unchanged) + a nonzero-bbox
#      crop instead of MASS's intensity-threshold body crop (preprocess_mass_
#      native.py --skull-stripped -- once input is truly skull-stripped,
#      nonzero already *is* the brain, no heuristic threshold needed)
#   -> native-token cache -> A2ss probe (300 epochs) -> B2ss finetune
#      (warm-started from A2ss, 30 epochs/patience 15, unbounded budget)
#
# Written to *_ss-suffixed output paths throughout so the original A2/B2
# (no skull-strip) results in artifacts/{attention,finetune}/ad/mass_native_d2*
# are preserved untouched for direct before/after comparison.
#
# Self-chaining, same pattern as the other scaling sbatch scripts.
set -uo pipefail
cd /net/projects2/litian-lab/scpan/encoders
source /opt/conda/etc/profile.d/conda.sh
conda activate /net/projects2/litian-lab/scpan/encoders/med310
export PYTHONUNBUFFERED=1
export PYTHONPATH=/net/projects2/litian-lab/scpan/encoders/src:${PYTHONPATH:-}

SCRIPT_PATH=/net/projects2/litian-lab/scpan/encoders/slurm/sbatch_scaling_exp_dataset2_skullstrip.sh
MAX_HOPS=20
HOP_FILE=/net/projects2/litian-lab/scpan/logs/.scaling_exp_d2_ss_hops
HOP=$(cat "$HOP_FILE" 2>/dev/null || echo 0)

RAW_MANIFEST=data/manifests/adni_full_mass_raw.csv
SS_DIR=/net/projects2/litian-lab/scpan/dataset/ADNI_full/ADNI_full_skullstripped
SS_MANIFEST=data/manifests/adni_full_mass_raw_ss.csv
OUT_DIR=/net/projects2/litian-lab/scpan/dataset/ADNI_full/ADNI_full_preprocessed_ss
FINAL_MANIFEST=data/manifests/adni_full_mass_ss.csv
AUDIT=artifacts/audits/ad/adni_full_mass_ss_preprocess_audit.json
CONFIG=config/adni_mass_scaling_exp.yaml
CACHE=artifacts/cache/ad/mass_native_d2_ss.npz
A2SS_DIR=artifacts/attention/ad/mass_native_d2_ss
B2SS_DIR=artifacts/finetune/ad/mass_native_d2_ss_warmstart

echo "=== hop $HOP started on $(hostname) at $(date -u) job=$SLURM_JOB_ID ==="
nvidia-smi -L

if [ ! -f "$SS_MANIFEST" ]; then
  echo "--- HD-BET skull-stripping (raw ADNI_full images) ---"
  python -m encoderbench.bsnip2.hdbet_batch --manifest "$RAW_MANIFEST" \
    --out-dir "$SS_DIR" --out-manifest "$SS_MANIFEST" --device 0 --mode fast || exit 1
else
  echo "--- skull-stripped manifest already exists, reusing: $SS_MANIFEST ---"
fi

echo "--- MASS-native reorient/resample/nonzero-crop over skull-stripped input ---"
python scripts/preprocess_mass_native.py --manifest "$SS_MANIFEST" --skull-stripped \
  --out-dir "$OUT_DIR" --out-manifest "$FINAL_MANIFEST" --audit "$AUDIT"
STATUS=$?
if [ "$STATUS" -ne 0 ]; then
  echo "=== preprocessing not finished (exit $STATUS) ==="
else
  echo "--- caching MASS native-token features (layer 3), skull-stripped dataset2 ---"
  if [ -f "$CACHE" ]; then
    echo "--- cache already exists, reusing ---"
  else
    encoderbench --config "$CONFIG" cache ad mass --manifest "$FINAL_MANIFEST" \
      --layer 3 --native-tokens --device cuda --out "$CACHE" || STATUS=1
  fi
fi

if [ "$STATUS" -eq 0 ]; then
  A2SS_DONE=1
  for SEED in 0 1 2; do
    [ -f "$A2SS_DIR/probe_seed_${SEED}_summary.json" ] || A2SS_DONE=0
  done
  if [ "$A2SS_DONE" -eq 1 ]; then
    echo "--- A2ss already complete, reusing ---"
  else
    echo "--- A2ss: frozen probe (native tokens, skull-stripped, 300 epochs) ---"
    encoderbench --config "$CONFIG" probe ad mass --cache "$CACHE" \
      --seeds all --device cuda --out-dir "$A2SS_DIR" || STATUS=1
  fi
fi

if [ "$STATUS" -eq 0 ]; then
  echo "--- B2ss: joint finetune, head warm-started from A2ss (native tokens, 30 epochs) ---"
  encoderbench --config "$CONFIG" finetune ad mass --manifest "$FINAL_MANIFEST" \
    --cache "$CACHE" --layer 3 --native-tokens --warm-start-dir "$A2SS_DIR" \
    --seeds all --device cuda --out-dir "$B2SS_DIR"
  STATUS=$?
fi

B2SS_DONE=1
for SEED in 0 1 2; do
  [ -f "$B2SS_DIR/finetune_seed_${SEED}_summary.json" ] || B2SS_DONE=0
done

if [ "$STATUS" -eq 0 ] && [ "$B2SS_DONE" -eq 1 ]; then
  echo "=== dataset2-skullstrip (A2ss+B2ss) complete at $(date -u) ==="
  rm -f "$HOP_FILE"
  exit 0
fi

NEXT_HOP=$((HOP + 1))
if [ "$NEXT_HOP" -ge "$MAX_HOPS" ]; then
  echo "ERROR: hit $MAX_HOPS resubmission hops without finishing -- stopping, needs human attention" >&2
  exit 1
fi
echo "$NEXT_HOP" > "$HOP_FILE"
echo "=== not finished (exit $STATUS, B2ss done=$B2SS_DONE), self-resubmitting (hop $NEXT_HOP) ==="
sbatch --dependency=afterany:$SLURM_JOB_ID "$SCRIPT_PATH"
