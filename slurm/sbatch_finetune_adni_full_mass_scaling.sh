#!/bin/bash
#SBATCH --job-name=adni_full_mass_scaling
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --exclude=j005-ds
#SBATCH --output=/net/projects2/litian-lab/scpan/logs/adni_full_mass_scaling_%j.log
#
# Full-parameter MASS finetune (no budget cap) + native tokens (no 4x4x4
# pooling) + attention head trained from scratch, on the full ADNI_full AD/CN
# baseline-screening cohort. Depends on sbatch_preprocess_adni_full_mass.sh
# having already produced data/manifests/adni_full_mass.csv.
#
# Self-chaining: finetune.py's own latest-checkpoint resume (see
# src/encoderbench/finetune.py's `latest_path` handling) already survives
# being killed mid-epoch -- this script never passes --restart, so relaunching
# it just continues training. If not all 3 seeds finish within one 12h window,
# this script resubmits itself via `sbatch --dependency=afterany`, hop-capped
# at 20 (240h) as a safety net. On full completion it generates the final
# report and stops.
set -uo pipefail
cd /net/projects2/litian-lab/scpan/encoders
source /opt/conda/etc/profile.d/conda.sh
conda activate /net/projects2/litian-lab/scpan/encoders/med310
export PYTHONUNBUFFERED=1

SCRIPT_PATH=/net/projects2/litian-lab/scpan/encoders/slurm/sbatch_finetune_adni_full_mass_scaling.sh
MAX_HOPS=20
HOP_FILE=/net/projects2/litian-lab/scpan/logs/.adni_full_mass_scaling_hops
HOP=$(cat "$HOP_FILE" 2>/dev/null || echo 0)

CONFIG=config/adni_full_mass_scaling.yaml
MANIFEST=data/manifests/adni_full_mass.csv
CACHE=artifacts/cache/ad/mass_full_native.npz
OUT_DIR=artifacts/finetune/ad/mass_full_scaling

echo "=== hop $HOP started on $(hostname) at $(date -u) job=$SLURM_JOB_ID ==="
nvidia-smi -L

if [ ! -f "$MANIFEST" ]; then
  echo "ERROR: $MANIFEST does not exist -- preprocessing must complete first" >&2
  exit 1
fi

if [ -f "$CACHE" ]; then
  echo "--- cache already exists, reusing: $CACHE ---"
else
  echo "--- caching MASS native-token features (layer 3) for normalization stats ---"
  encoderbench --config "$CONFIG" cache ad mass --manifest "$MANIFEST" \
    --layer 3 --native-tokens --device cuda --out "$CACHE" || exit 1
fi

echo "--- finetune (all seeds, resumes automatically if a checkpoint exists) ---"
encoderbench --config "$CONFIG" finetune ad mass --manifest "$MANIFEST" \
  --cache "$CACHE" --layer 3 --native-tokens --no-warm-start \
  --seeds all --device cuda --out-dir "$OUT_DIR"
STATUS=$?

DONE=1
for SEED in 0 1 2; do
  if [ ! -f "$OUT_DIR/finetune_seed_${SEED}_summary.json" ]; then
    DONE=0
  fi
done

if [ "$STATUS" -eq 0 ] && [ "$DONE" -eq 1 ]; then
  echo "=== all 3 seeds complete at $(date -u) ==="
  rm -f "$HOP_FILE"
  echo "--- generating report ---"
  python scripts/generate_adni_mass_scaling_report.py
  exit 0
fi

NEXT_HOP=$((HOP + 1))
if [ "$NEXT_HOP" -ge "$MAX_HOPS" ]; then
  echo "ERROR: hit $MAX_HOPS resubmission hops without finishing -- stopping, needs human attention" >&2
  exit 1
fi
echo "$NEXT_HOP" > "$HOP_FILE"
echo "=== not finished (finetune exit $STATUS, all-seeds-done=$DONE), self-resubmitting (hop $NEXT_HOP) ==="
sbatch --dependency=afterany:$SLURM_JOB_ID "$SCRIPT_PATH"
