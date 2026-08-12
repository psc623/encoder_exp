#!/bin/bash
#SBATCH --job-name=biot_eeg_ad
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --exclude=j005-ds
#SBATCH --output=/net/projects2/litian-lab/scpan/logs/biot_eeg_ad_%j.log

set -euo pipefail

EXP_DIR=/net/projects2/litian-lab/scpan/encoders/EEG_exp
cd "$EXP_DIR"
source /opt/conda/etc/profile.d/conda.sh
conda activate /net/projects2/litian-lab/scpan/encoders/med310
export PYTHONUNBUFFERED=1

python train_biot_ac.py train \
  --regimes pretrained_frozen pretrained_full scratch \
  --n-splits 5 \
  --epochs 30 \
  --patience 7 \
  --batch-size 32 \
  --eval-batch-size 64 \
  --train-windows-per-subject 20 \
  --cache-workers 4 \
  --device cuda
