#!/bin/bash
#SBATCH --job-name=improved_synthseg_smoke
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --exclude=j005-ds
#SBATCH --output=/net/projects2/litian-lab/scpan/logs/improved_synthseg_smoke_%j.log
set -euo pipefail
cd /net/projects2/litian-lab/scpan/encoders
source /opt/conda/etc/profile.d/conda.sh
conda activate /net/projects2/litian-lab/scpan/encoders/med310
export PYTHONUNBUFFERED=1
DRIVER=scripts/run_improved_bsnip2_synthseg.py
echo "=== mode1 seed 0 ==="
time python "$DRIVER" mode1 --seeds 0
echo "=== mode2 seed 0 ==="
time python "$DRIVER" mode2 --seeds 0
echo "=== mode4 seed 0 (linear warm-start, adapter-only finetune) ==="
time python "$DRIVER" mode4 --seeds 0
echo "=== mode3 seed 0 (attention warm-start, adapter-only finetune) ==="
time python "$DRIVER" mode3 --seeds 0
echo "=== mode6 seed 0 (linear from-zero joint, adapter unfrozen) ==="
time python "$DRIVER" mode6 --seeds 0
echo "=== mode5 seed 0 (attention from-zero joint, adapter unfrozen) ==="
time python "$DRIVER" mode5 --seeds 0
echo "=== report ==="
python "$DRIVER" report
echo "=== synthseg smoke all 6 modes OK ==="
