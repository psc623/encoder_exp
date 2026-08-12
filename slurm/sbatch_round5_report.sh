#!/bin/bash
#SBATCH --job-name=round5_report
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/net/projects2/litian-lab/scpan/logs/round5_report_%j.log
#
# Builds report/scaling_exp_improve/scaling_round5_report.md and the per-seed
# curve SVGs under plots_round5/. Submitted with --dependency=afterok on the
# round-5 training job so the plots exist as soon as the run lands, without
# needing the training script (already queued, and therefore frozen in slurm's
# spool) to be edited.
set -uo pipefail
cd /net/projects2/litian-lab/scpan/encoders
source /opt/conda/etc/profile.d/conda.sh
conda activate /net/projects2/litian-lab/scpan/encoders/med310
export PYTHONUNBUFFERED=1
export PYTHONPATH=/net/projects2/litian-lab/scpan/encoders/src:${PYTHONPATH:-}
python scripts/generate_round5_report.py
