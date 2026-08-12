#!/bin/bash
#SBATCH --job-name=wmr_crop
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --exclude=j005-ds
#SBATCH --output=/net/projects2/litian-lab/scpan/logs/wmr_crop_%j.log
#
# "Crop" wmr variant: registered (SPM12 MNI-warped) whole-head T1 first run
# through MASS's OWN geometry pipeline (preprocess_mass.py: reorient RAS ->
# resample 1.5mm isotropic [already true for wmr, so effectively a no-op] ->
# threshold body-crop), THEN MASS's clip_zscore + resize(128^3) -- controls
# for the background-proportion mismatch the no-crop variant has (wmr is
# ~26% nonzero vs MASS's own body-cropped inputs having much less background).
set -euo pipefail
cd /net/projects2/litian-lab/scpan/encoders
source /opt/conda/etc/profile.d/conda.sh
conda activate /net/projects2/litian-lab/scpan/encoders/med310
export PYTHONUNBUFFERED=1

MANIFEST=data/manifests/bsnip2_mass_wmr.csv
OUT_DIR=data/bsnip2_mass_wmr_crop
FINAL_MANIFEST=data/manifests/bsnip2_mass_wmr_crop_final.csv

echo "=== body-crop wmr volumes with MASS's own geometry pipeline (CPU, 32-way) ==="
N=$(python -c "import csv; print(len(list(csv.DictReader(open('$MANIFEST')))))")
echo "manifest rows: $N"
seq 0 $((N - 1)) | xargs -P 32 -I{} python -m encoderbench.bsnip2.preprocess_mass \
  --manifest "$MANIFEST" --out-dir "$OUT_DIR" --single-index {}

echo "=== merge ==="
python -m encoderbench.bsnip2.preprocess_mass --manifest "$MANIFEST" \
  --out-dir "$OUT_DIR" --out-manifest "$FINAL_MANIFEST" --merge

echo "=== cache (native tokens, cropped) ==="
python scripts/cache_mass_native_generic.py \
  --manifest "$FINAL_MANIFEST" \
  --out artifacts/cache/bsnip2/mass_wmr_crop_native.npz \
  --note "SPM12 MNI-registered wmr, body-cropped via MASS's own geometry pipeline"

echo "=== mode1 (frozen attention) ==="
python scripts/run_improved_bsnip2_mass_wmr.py mode1 --variant crop --seeds all

echo "=== mode2 (frozen linear) ==="
python scripts/run_improved_bsnip2_mass_wmr.py mode2 --variant crop --seeds all

python scripts/run_improved_bsnip2_mass_wmr.py report
echo "=== wmr_crop done ==="
