#!/bin/bash
#SBATCH --job-name=cv_stage2_sweep
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --exclude=j005-ds
#SBATCH --output=/net/projects2/litian-lab/scpan/logs/cv_stage2_sweep_%j.log
#
# Stage 2: finetune hyperparameter sweep, 6 configs x modes {4,6} x 2 repeats,
# chosen by validation BA only. Depends on stage 1 having written the per-repeat
# mode1/mode2 checkpoints that modes 3/4 warm-start from.
set -euo pipefail
cd /net/projects2/litian-lab/scpan/encoders
source /opt/conda/etc/profile.d/conda.sh
conda activate /net/projects2/litian-lab/scpan/encoders/med310
export PYTHONUNBUFFERED=1
python scripts/sweep_cv_finetune.py --modes 4,6
echo "=== stage2 sweep done ==="
