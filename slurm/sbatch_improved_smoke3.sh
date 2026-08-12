#!/bin/bash
#SBATCH --job-name=improved_smoke3
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --exclude=j005-ds
#SBATCH --output=/net/projects2/litian-lab/scpan/logs/improved_smoke3_%j.log
set -euo pipefail
cd /net/projects2/litian-lab/scpan/encoders
source /opt/conda/etc/profile.d/conda.sh
conda activate /net/projects2/litian-lab/scpan/encoders/med310
export PYTHONUNBUFFERED=1
echo "=== mode1 seed 0 (now writes to canonical attention/ dir) ==="
python scripts/run_improved_bsnip2_mass.py mode1 --seeds 0
echo "=== mode2 seed 0 (now writes to canonical linear/ dir) ==="
python scripts/run_improved_bsnip2_mass.py mode2 --seeds 0
echo "=== mode4 seed 0 (warm-start fix + hard-fail-if-missing check) ==="
python scripts/run_improved_bsnip2_mass.py mode4 --seeds 0
echo "=== mode3 seed 0 (attention warm-start path, not tested before) ==="
python scripts/run_improved_bsnip2_mass.py mode3 --seeds 0
echo "=== mode6 seed 0 (configure_full_finetune fix) ==="
python scripts/run_improved_bsnip2_mass.py mode6 --seeds 0
echo "=== mode5 seed 0 (attention, full unfrozen, not tested before) ==="
python scripts/run_improved_bsnip2_mass.py mode5 --seeds 0
echo "=== report (sanity check it runs even with only seed-0 data) ==="
python scripts/run_improved_bsnip2_mass.py report
echo "=== smoke3 all 6 modes OK ==="
