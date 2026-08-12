#!/bin/bash
#SBATCH --job-name=scaling_exp_d3
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --exclude=j005-ds
#SBATCH --output=/net/projects2/litian-lab/scpan/logs/scaling_exp_d3_%j.log
#
# A3/B3 cells: dataset3 = ADNI_full, every visit (screening + m6/m12/.../m48),
# CN/AD only (MCI dropped per instruction), skull-stripped via the same
# HD-BET + MASS-native nonzero-crop recipe validated for A2ss/B2ss. dataset2
# (screening-only, 431 subjects) is a strict subject+file_id subset of
# dataset3's raw manifest, and both the HD-BET batch script and
# preprocess_mass_native.py are idempotent keyed by (subject_id, file_id) --
# pointed at the SAME output directories used for A2ss/B2ss, this
# automatically reuses the 431 already-skull-stripped/preprocessed screening
# volumes and only does new work for the additional non-screening visits, no
# separate "what's already done" bookkeeping needed.
#
# Pipeline: raw manifest (data/manifests/adni_full_mass3_raw.csv, built by
# `build_manifest_adni_full_mass.py --groups CN,AD --all-visits`)
#   -> HD-BET skull-strip (same ADNI_full_skullstripped output dir as A2ss/B2ss)
#   -> MASS reorient(RAS)/resample(1.5mm) + nonzero-bbox crop (same
#      ADNI_full_preprocessed_ss output dir as A2ss/B2ss)
#   -> native-token cache -> A3 probe (300 epochs) -> B3 finetune
#      (warm-started from A3, 30 epochs/patience 15, unbounded budget)
#
# Self-chaining, same pattern as the other scaling sbatch scripts.
set -uo pipefail
cd /net/projects2/litian-lab/scpan/encoders
source /opt/conda/etc/profile.d/conda.sh
conda activate /net/projects2/litian-lab/scpan/encoders/med310
export PYTHONUNBUFFERED=1
export PYTHONPATH=/net/projects2/litian-lab/scpan/encoders/src:${PYTHONPATH:-}

SCRIPT_PATH=/net/projects2/litian-lab/scpan/encoders/slurm/sbatch_scaling_exp_dataset3.sh
MAX_HOPS=20
HOP_FILE=/net/projects2/litian-lab/scpan/logs/.scaling_exp_d3_hops
HOP=$(cat "$HOP_FILE" 2>/dev/null || echo 0)

RAW_MANIFEST=data/manifests/adni_full_mass3_raw.csv
# Same output dirs as A2ss/B2ss on purpose -- idempotent reuse of the 431
# already-processed screening volumes, see header comment.
SS_DIR=/net/projects2/litian-lab/scpan/dataset/ADNI_full/ADNI_full_skullstripped
SS_MANIFEST=data/manifests/adni_full_mass3_raw_ss.csv
OUT_DIR=/net/projects2/litian-lab/scpan/dataset/ADNI_full/ADNI_full_preprocessed_ss
FINAL_MANIFEST=data/manifests/adni_full_mass3_ss.csv
AUDIT=artifacts/audits/ad/adni_full_mass3_ss_preprocess_audit.json
CONFIG=config/adni_mass_scaling_exp.yaml
CACHE=artifacts/cache/ad/mass_native_d3.npz
A3_DIR=artifacts/attention/ad/mass_native_d3
B3_DIR=artifacts/finetune/ad/mass_native_d3_warmstart

echo "=== hop $HOP started on $(hostname) at $(date -u) job=$SLURM_JOB_ID ==="
nvidia-smi -L

if [ ! -f "$RAW_MANIFEST" ]; then
  echo "--- building dataset3 raw manifest (all visits, CN/AD only) ---"
  python scripts/build_manifest_adni_full_mass.py --groups CN,AD --all-visits --out "$RAW_MANIFEST" || exit 1
else
  echo "--- raw manifest already exists, reusing: $RAW_MANIFEST ---"
fi

if [ ! -f "$SS_MANIFEST" ] || [ "$(wc -l < "$SS_MANIFEST")" -lt "$(wc -l < "$RAW_MANIFEST")" ]; then
  echo "--- HD-BET skull-stripping (idempotent -- reuses the 431 screening volumes already done for A2ss/B2ss) ---"
  python -m encoderbench.bsnip2.hdbet_batch --manifest "$RAW_MANIFEST" \
    --out-dir "$SS_DIR" --out-manifest "$SS_MANIFEST" --device 0 --mode fast || exit 1
else
  echo "--- skull-stripped manifest already covers all rows, reusing: $SS_MANIFEST ---"
fi

echo "--- MASS-native reorient/resample/nonzero-crop (idempotent) ---"
python scripts/preprocess_mass_native.py --manifest "$SS_MANIFEST" --skull-stripped \
  --out-dir "$OUT_DIR" --out-manifest "$FINAL_MANIFEST" --audit "$AUDIT"
STATUS=$?
if [ "$STATUS" -ne 0 ]; then
  echo "=== preprocessing not finished (exit $STATUS) ==="
else
  echo "--- caching MASS native-token features (layer 3), dataset3 ---"
  if [ -f "$CACHE" ]; then
    echo "--- cache already exists, reusing ---"
  else
    encoderbench --config "$CONFIG" cache ad mass --manifest "$FINAL_MANIFEST" \
      --layer 3 --native-tokens --device cuda --out "$CACHE" || STATUS=1
  fi
fi

if [ "$STATUS" -eq 0 ]; then
  A3_DONE=1
  for SEED in 0 1 2; do
    [ -f "$A3_DIR/probe_seed_${SEED}_summary.json" ] || A3_DONE=0
  done
  if [ "$A3_DONE" -eq 1 ]; then
    echo "--- A3 already complete, reusing ---"
  else
    echo "--- A3: frozen probe (native tokens, all-visit CN/AD, 300 epochs) ---"
    encoderbench --config "$CONFIG" probe ad mass --cache "$CACHE" \
      --seeds all --device cuda --out-dir "$A3_DIR" || STATUS=1
  fi
fi

if [ "$STATUS" -eq 0 ]; then
  echo "--- B3: joint finetune, head warm-started from A3 (native tokens, 30 epochs) ---"
  encoderbench --config "$CONFIG" finetune ad mass --manifest "$FINAL_MANIFEST" \
    --cache "$CACHE" --layer 3 --native-tokens --warm-start-dir "$A3_DIR" \
    --seeds all --device cuda --out-dir "$B3_DIR"
  STATUS=$?
fi

B3_DONE=1
for SEED in 0 1 2; do
  [ -f "$B3_DIR/finetune_seed_${SEED}_summary.json" ] || B3_DONE=0
done

if [ "$STATUS" -eq 0 ] && [ "$B3_DONE" -eq 1 ]; then
  echo "=== dataset3 (A3+B3) complete at $(date -u) ==="
  rm -f "$HOP_FILE"
  exit 0
fi

NEXT_HOP=$((HOP + 1))
if [ "$NEXT_HOP" -ge "$MAX_HOPS" ]; then
  echo "ERROR: hit $MAX_HOPS resubmission hops without finishing -- stopping, needs human attention" >&2
  exit 1
fi
echo "$NEXT_HOP" > "$HOP_FILE"
echo "=== not finished (exit $STATUS, B3 done=$B3_DONE), self-resubmitting (hop $NEXT_HOP) ==="
sbatch --dependency=afterany:$SLURM_JOB_ID "$SCRIPT_PATH"
