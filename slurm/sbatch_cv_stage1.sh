#!/bin/bash
#SBATCH --job-name=cv_stage1
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --exclude=j005-ds
#SBATCH --output=/net/projects2/litian-lab/scpan/logs/cv_stage1_%j.log
#
# Stage 1 of the repeated-split re-run: frozen baselines only (mode1 attention
# probe, mode2 exact-linear with inner 5-fold CV), 5 repeats each. These are
# the numbers modes 3/5 and 4/6 have to beat, and modes 3/4 warm-start from
# their per-repeat checkpoints, so they must exist before stage 3.
set -euo pipefail
cd /net/projects2/litian-lab/scpan/encoders
source /opt/conda/etc/profile.d/conda.sh
conda activate /net/projects2/litian-lab/scpan/encoders/med310
export PYTHONUNBUFFERED=1
DRIVER=scripts/run_cv_bsnip2_mass.py

echo "=== split sanity ==="
python "$DRIVER" splits

echo "=== mode2 (exact linear, inner 5-fold CV) -- fast, run first ==="
python "$DRIVER" run --mode 2 --repeats all

echo "=== mode1 (attention probe) ==="
python "$DRIVER" run --mode 1 --repeats all

python "$DRIVER" report
echo "=== stage1 done ==="
