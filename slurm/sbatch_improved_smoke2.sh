#!/bin/bash
#SBATCH --job-name=improved_smoke2
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --exclude=j005-ds
#SBATCH --output=/net/projects2/litian-lab/scpan/logs/improved_smoke2_%j.log
set -euo pipefail
cd /net/projects2/litian-lab/scpan/encoders
source /opt/conda/etc/profile.d/conda.sh
conda activate /net/projects2/litian-lab/scpan/encoders/med310
export PYTHONUNBUFFERED=1
echo "=== mode4 seed 0 (manifest path fixed) ==="
python scripts/run_improved_bsnip2_mass.py mode4 --seeds 0
echo "=== mode6 seed 0 (exercises joint from-zero training path) ==="
python scripts/run_improved_bsnip2_mass.py mode6 --seeds 0
echo "=== smoke2 OK ==="
