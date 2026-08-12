#!/bin/bash
#SBATCH --job-name=improved_smoke
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --exclude=j005-ds
#SBATCH --output=/net/projects2/litian-lab/scpan/logs/improved_smoke_%j.log
#
# Cheap smoke test of the new linear_head.py / finetune_variants.py / driver
# script before committing the full 4xA100/12h run: mode1 seed 0, mode2 seed
# 0 only, then one mode4 seed-0 step check (encoder-only finetune with a
# warm-started linear head) to exercise the new fine-tune-variant code path
# without waiting for a full 12-epoch run.
set -euo pipefail
cd /net/projects2/litian-lab/scpan/encoders
source /opt/conda/etc/profile.d/conda.sh
conda activate /net/projects2/litian-lab/scpan/encoders/med310
export PYTHONUNBUFFERED=1
echo "=== mode1 seed 0 ==="
python scripts/run_improved_bsnip2_mass.py mode1 --seeds 0
echo "=== mode2 seed 0 ==="
python scripts/run_improved_bsnip2_mass.py mode2 --seeds 0
echo "=== mode4 seed 0 (exercises finetune_variants.py + linear warm-start) ==="
python scripts/run_improved_bsnip2_mass.py mode4 --seeds 0
echo "=== smoke test OK ==="
