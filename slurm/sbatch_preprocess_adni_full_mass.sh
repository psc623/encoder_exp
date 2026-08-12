#!/bin/bash
#SBATCH --job-name=adni_full_mass_preprocess
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/net/projects2/litian-lab/scpan/logs/adni_full_mass_preprocess_%j.log
#
# CPU-only: reorient/resample/body-crop via SimpleITK needs no GPU. Builds the
# AD/CN baseline-screening manifest from ADNI_full's raw XML metadata (once),
# then runs MASS's own preprocessing recipe (scripts/preprocess_mass_native.py)
# over every row.
#
# Self-chaining: general's walltime cap is 12h and this dataset (~800+ raw
# subjects) may not finish preprocessing in one window. Both steps are
# idempotent (manifest build is deterministic; preprocessing skips files that
# already exist), so if this run doesn't finish, it resubmits itself via
# `sbatch --dependency=afterany` and picks up exactly where it left off. A hop
# counter caps this at 20 resubmissions (240h) so a real bug fails loudly
# instead of looping forever. On full completion it submits the finetune job.
set -uo pipefail
cd /net/projects2/litian-lab/scpan/encoders
source /opt/conda/etc/profile.d/conda.sh
conda activate /net/projects2/litian-lab/scpan/encoders/med310
export PYTHONUNBUFFERED=1

SCRIPT_PATH=/net/projects2/litian-lab/scpan/encoders/slurm/sbatch_preprocess_adni_full_mass.sh
MAX_HOPS=20
HOP_FILE=/net/projects2/litian-lab/scpan/logs/.adni_full_mass_preprocess_hops
HOP=$(cat "$HOP_FILE" 2>/dev/null || echo 0)

RAW_MANIFEST=data/manifests/adni_full_mass_raw.csv
OUT_MANIFEST=data/manifests/adni_full_mass.csv
OUT_DIR=/net/projects2/litian-lab/scpan/dataset/ADNI_full/ADNI_full_preprocessed
AUDIT=artifacts/audits/ad/adni_full_mass_preprocess_audit.json

echo "=== hop $HOP started on $(hostname) at $(date -u) job=$SLURM_JOB_ID ==="

if [ ! -f "$RAW_MANIFEST" ]; then
  echo "--- building raw manifest from ADNI_full XML metadata ---"
  python scripts/build_manifest_adni_full_mass.py --out "$RAW_MANIFEST"
else
  echo "--- raw manifest already exists, reusing: $RAW_MANIFEST ---"
fi

echo "--- preprocessing (idempotent, resumes automatically) ---"
python scripts/preprocess_mass_native.py \
  --manifest "$RAW_MANIFEST" --out-dir "$OUT_DIR" \
  --out-manifest "$OUT_MANIFEST" --audit "$AUDIT"
STATUS=$?

if [ "$STATUS" -eq 0 ]; then
  echo "=== preprocessing complete at $(date -u) ==="
  rm -f "$HOP_FILE"
  echo "=== submitting finetune job ==="
  sbatch /net/projects2/litian-lab/scpan/encoders/slurm/sbatch_finetune_adni_full_mass_scaling.sh
  exit 0
fi

NEXT_HOP=$((HOP + 1))
if [ "$NEXT_HOP" -ge "$MAX_HOPS" ]; then
  echo "ERROR: hit $MAX_HOPS resubmission hops without finishing -- stopping, needs human attention" >&2
  exit 1
fi
echo "$NEXT_HOP" > "$HOP_FILE"
echo "=== not finished (preprocess exit $STATUS), self-resubmitting (hop $NEXT_HOP) ==="
sbatch --dependency=afterany:$SLURM_JOB_ID "$SCRIPT_PATH"
