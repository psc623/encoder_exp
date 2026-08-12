#!/bin/bash
#SBATCH --job-name=fix_synthseg_mode3
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --exclude=j005-ds
#SBATCH --output=/net/projects2/litian-lab/scpan/logs/fix_synthseg_mode3_%j.log
set -euo pipefail
cd /net/projects2/litian-lab/scpan/encoders
source /opt/conda/etc/profile.d/conda.sh
conda activate /net/projects2/litian-lab/scpan/encoders/med310
export PYTHONUNBUFFERED=1
python scripts/run_improved_bsnip2_synthseg.py mode3 --seeds all --restart
python scripts/run_improved_bsnip2_synthseg.py report
