#!/bin/bash
#SBATCH --job-name=bsnip2_mass
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --exclude=j005-ds
#SBATCH --output=/net/projects2/litian-lab/scpan/logs/bsnip2_mass_%j.log
#
# BSNIP2 raw whole-head T1 -> MASS's real native geometry pipeline: reorient
# to RAS, resample to MASS's own 1.5mm isotropic spacing, threshold body
# crop. Pure CPU/SimpleITK work, no GPU needed. Submitted with sbatch so it
# survives the submitting shell/VSCode disconnecting; idempotent, so
# re-submitting resumes rather than restarting.
set -o pipefail
cd /net/projects2/litian-lab/scpan/encoders
source /opt/conda/etc/profile.d/conda.sh
conda activate /net/projects2/litian-lab/scpan/encoders/med310
export PYTHONUNBUFFERED=1

MANIFEST=data/manifests/bsnip2_raw.csv
OUT_DIR=data/bsnip2_mass_final
FINAL_MANIFEST=data/manifests/bsnip2_mass_final.csv

echo "=== bsnip2 mass preprocessing on $(hostname) started $(date -u) ==="
N=$(python -c "import csv; print(len(list(csv.DictReader(open('$MANIFEST')))))")
echo "manifest rows: $N"

echo "--- reorient + resample + body crop (CPU, 32-way parallel) ---"
seq 0 $((N - 1)) | xargs -P 32 -I{} python -m encoderbench.bsnip2.preprocess_mass \
  --manifest "$MANIFEST" --out-dir "$OUT_DIR" --single-index {}

echo "--- merge ---"
python -m encoderbench.bsnip2.preprocess_mass --manifest "$MANIFEST" \
  --out-dir "$OUT_DIR" --out-manifest "$FINAL_MANIFEST" --merge

wc -l "$FINAL_MANIFEST" 2>/dev/null
echo "=== bsnip2 mass preprocessing done $(date -u) ==="
