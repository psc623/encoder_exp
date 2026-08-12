#!/bin/bash
#SBATCH --job-name=scaling_exp_d4
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --exclude=j005-ds
#SBATCH --output=/net/projects2/litian-lab/scpan/logs/scaling_exp_d4_%j.log
#
# A4/B4 cells: dataset4 = dataset2 (431 subjects, ADNI1, 1.5T, screening,
# already HD-BET-skull-stripped + MASS-native preprocessed for A2ss/B2ss) plus
# 388 new baseline CN/AD subjects from ADNI_add_full (ADNI2-4, 3T, confirmed
# zero subject_id overlap with dataset2 and 100% field_strength=3.0 across all
# sampled records). Both pools are split independently at the SAME fractions
# (test 0.15, validation 0.1765-of-remaining -> ~70/15/15) and then
# concatenated: since dataset2 == 1.5T and the new portion == 3T with no
# subject overlap, this is equivalent to a joint (field_strength x diagnosis)
# stratified split of the combined 819-subject pool without extending
# manifest.stratified_subject_split for a second key. This also guarantees
# every split (train/val/test) contains both 1.5T and 3T subjects, avoiding
# the field-strength-confounds-with-split trap flagged for this merge.
#
# dataset2's existing A2ss/B2ss split is NOT reused here -- it's reshuffled
# from scratch at the new fractions (data/manifests/adni_full_mass_d2_
# reshuffled_ss.csv, already built), so A4/B4 is not a like-for-like subject-
# level comparison against A2ss/B2ss (different test set membership); it is a
# fresh, larger, better-balanced-ratio split by design, per instruction.
#
# B4 uses encoder_learning_rate=5e-4 (config/adni_mass_scaling_exp.yaml
# already carries this from B3 onward -- no override needed here).
#
# Pipeline: new-portion raw manifest (data/manifests/adni_add_full_mass_raw.csv,
# already built via `build_manifest_adni_full_mass.py --metadata-root/--image-root
# pointed at ADNI_add_full --test-fraction 0.15 --validation-fraction 0.1765`)
#   -> HD-BET skull-strip (same shared ADNI_full_skullstripped output dir as
#      A2ss/B2ss/A3/B3 -- idempotent, keyed by subject_id+file_id, zero
#      collision risk given zero subject overlap)
#   -> MASS reorient(RAS)/resample(1.5mm) + nonzero-bbox crop (same shared
#      ADNI_full_preprocessed_ss output dir)
#   -> merge with dataset2's reshuffled manifest -> native-token cache
#   -> A4 probe (300 epochs) -> B4 finetune (warm-started from A4, 30 epochs/
#      patience 15, unbounded budget, encoder_lr=5e-4)
#   -> regenerate report/scaling_exp/scaling_exp_report.md (A4/B4 already
#      added to generate_scaling_exp_report.py's CELLS dict) and re-append
#      the A2mc/B2mc 3-class section on top (append_multiclass_to_scaling_
#      report.py -- the generator overwrites the whole file, so the 3-class
#      section has to be re-appended every time, not just the first time)
#
# Self-chaining, same pattern as the other scaling sbatch scripts.
set -uo pipefail
cd /net/projects2/litian-lab/scpan/encoders
source /opt/conda/etc/profile.d/conda.sh
conda activate /net/projects2/litian-lab/scpan/encoders/med310
export PYTHONUNBUFFERED=1
export PYTHONPATH=/net/projects2/litian-lab/scpan/encoders/src:${PYTHONPATH:-}

SCRIPT_PATH=/net/projects2/litian-lab/scpan/encoders/slurm/sbatch_scaling_exp_dataset4.sh
MAX_HOPS=25
HOP_FILE=/net/projects2/litian-lab/scpan/logs/.scaling_exp_d4_hops
HOP=$(cat "$HOP_FILE" 2>/dev/null || echo 0)

NEW_RAW_MANIFEST=data/manifests/adni_add_full_mass_raw.csv
SS_DIR=/net/projects2/litian-lab/scpan/dataset/ADNI_full/ADNI_full_skullstripped
NEW_SS_MANIFEST=data/manifests/adni_add_full_mass_raw_ss.csv
OUT_DIR=/net/projects2/litian-lab/scpan/dataset/ADNI_full/ADNI_full_preprocessed_ss
NEW_FINAL_MANIFEST=data/manifests/adni_add_full_mass_ss.csv
AUDIT=artifacts/audits/ad/adni_add_full_mass_ss_preprocess_audit.json
OLD_RESHUFFLED_MANIFEST=data/manifests/adni_full_mass_d2_reshuffled_ss.csv
FINAL_MANIFEST=data/manifests/adni_full_mass4_ss.csv
CONFIG=config/adni_mass_scaling_exp.yaml
CACHE=artifacts/cache/ad/mass_native_d4.npz
A4_DIR=artifacts/attention/ad/mass_native_d4
B4_DIR=artifacts/finetune/ad/mass_native_d4_warmstart

echo "=== hop $HOP started on $(hostname) at $(date -u) job=$SLURM_JOB_ID ==="
nvidia-smi -L

if [ ! -f "$NEW_SS_MANIFEST" ] || [ "$(wc -l < "$NEW_SS_MANIFEST")" -lt "$(wc -l < "$NEW_RAW_MANIFEST")" ]; then
  echo "--- HD-BET skull-stripping new ADNI_add_full portion ---"
  python -m encoderbench.bsnip2.hdbet_batch --manifest "$NEW_RAW_MANIFEST" \
    --out-dir "$SS_DIR" --out-manifest "$NEW_SS_MANIFEST" --device 0 --mode fast || exit 1
else
  echo "--- new-portion skull-stripped manifest already exists, reusing: $NEW_SS_MANIFEST ---"
fi

echo "--- MASS-native reorient/resample/nonzero-crop over skull-stripped new portion ---"
python scripts/preprocess_mass_native.py --manifest "$NEW_SS_MANIFEST" --skull-stripped \
  --out-dir "$OUT_DIR" --out-manifest "$NEW_FINAL_MANIFEST" --audit "$AUDIT"
STATUS=$?
if [ "$STATUS" -ne 0 ]; then
  echo "=== new-portion preprocessing not finished (exit $STATUS) ==="
else
  echo "--- merging new portion + dataset2 reshuffled manifest -> dataset4 ---"
  python scripts/merge_adni_full_mass4_manifest.py \
    --new-portion "$NEW_FINAL_MANIFEST" --old-portion "$OLD_RESHUFFLED_MANIFEST" \
    --out "$FINAL_MANIFEST" || STATUS=1
fi

if [ "$STATUS" -eq 0 ]; then
  echo "--- caching MASS native-token features (layer 3), dataset4 ---"
  if [ -f "$CACHE" ]; then
    echo "--- cache already exists, reusing ---"
  else
    encoderbench --config "$CONFIG" cache ad mass --manifest "$FINAL_MANIFEST" \
      --layer 3 --native-tokens --device cuda --out "$CACHE" || STATUS=1
  fi
fi

if [ "$STATUS" -eq 0 ]; then
  A4_DONE=1
  for SEED in 0 1 2; do
    [ -f "$A4_DIR/probe_seed_${SEED}_summary.json" ] || A4_DONE=0
  done
  if [ "$A4_DONE" -eq 1 ]; then
    echo "--- A4 already complete, reusing ---"
  else
    echo "--- A4: frozen probe (native tokens, dataset4, 300 epochs) ---"
    encoderbench --config "$CONFIG" probe ad mass --cache "$CACHE" \
      --seeds all --device cuda --out-dir "$A4_DIR" || STATUS=1
  fi
fi

if [ "$STATUS" -eq 0 ]; then
  echo "--- B4: joint finetune, head warm-started from A4 (native tokens, 30 epochs, encoder_lr=5e-4) ---"
  encoderbench --config "$CONFIG" finetune ad mass --manifest "$FINAL_MANIFEST" \
    --cache "$CACHE" --layer 3 --native-tokens --warm-start-dir "$A4_DIR" \
    --seeds all --device cuda --out-dir "$B4_DIR"
  STATUS=$?
fi

B4_DONE=1
for SEED in 0 1 2; do
  [ -f "$B4_DIR/finetune_seed_${SEED}_summary.json" ] || B4_DONE=0
done

if [ "$STATUS" -eq 0 ] && [ "$B4_DONE" -eq 1 ]; then
  echo "--- regenerating scaling_exp_report.md with A4/B4 ---"
  python scripts/generate_scaling_exp_report.py && python scripts/append_multiclass_to_scaling_report.py
  echo "=== dataset4 (A4+B4) complete at $(date -u) ==="
  rm -f "$HOP_FILE"
  exit 0
fi

NEXT_HOP=$((HOP + 1))
if [ "$NEXT_HOP" -ge "$MAX_HOPS" ]; then
  echo "ERROR: hit $MAX_HOPS resubmission hops without finishing -- stopping, needs human attention" >&2
  exit 1
fi
echo "$NEXT_HOP" > "$HOP_FILE"
echo "=== not finished (exit $STATUS, B4 done=$B4_DONE), self-resubmitting (hop $NEXT_HOP) ==="
sbatch --dependency=afterany:$SLURM_JOB_ID "$SCRIPT_PATH"
