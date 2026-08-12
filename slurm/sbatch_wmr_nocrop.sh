#!/bin/bash
#SBATCH --job-name=wmr_nocrop
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --exclude=j005-ds
#SBATCH --output=/net/projects2/litian-lab/scpan/logs/wmr_nocrop_%j.log
#
# "No-crop" wmr variant: registered (SPM12 MNI-warped) whole-head T1 fed
# straight to MASS's own clip_zscore + resize(128^3), no body-crop -- tests
# the registration hypothesis with the smallest possible number of changed
# variables (no reorientation/resample either, wmr is already 1.5mm isotropic
# matching MASS's own target spacing).
set -euo pipefail
cd /net/projects2/litian-lab/scpan/encoders
source /opt/conda/etc/profile.d/conda.sh
conda activate /net/projects2/litian-lab/scpan/encoders/med310
export PYTHONUNBUFFERED=1

echo "=== cache (native tokens, no crop) ==="
python scripts/cache_mass_native_generic.py \
  --manifest data/manifests/bsnip2_mass_wmr.csv \
  --out artifacts/cache/bsnip2/mass_wmr_nocrop_native.npz \
  --note "SPM12 MNI-registered wmr, no body-crop, direct resize to 128^3"

echo "=== mode1 (frozen attention) ==="
python scripts/run_improved_bsnip2_mass_wmr.py mode1 --variant nocrop --seeds all

echo "=== mode2 (frozen linear) ==="
python scripts/run_improved_bsnip2_mass_wmr.py mode2 --variant nocrop --seeds all

python scripts/run_improved_bsnip2_mass_wmr.py report
echo "=== wmr_nocrop done ==="
