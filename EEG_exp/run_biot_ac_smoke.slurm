#!/bin/bash
#SBATCH --job-name=biot_ac_smoke
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --nodes=1
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/net/projects2/litian-lab/scpan/logs/biot_ac_smoke_%j.log

set -euo pipefail

EXP_DIR=/net/projects2/litian-lab/scpan/encoders/EEG_exp
cd "$EXP_DIR"
source /opt/conda/etc/profile.d/conda.sh
conda activate /net/projects2/litian-lab/scpan/encoders/med310
export PYTHONUNBUFFERED=1

# Pipeline validation only: this deliberately caps train/evaluation windows.
python train_biot_ac.py train \
  --regimes pretrained_frozen pretrained_last2 pretrained_full scratch \
  --folds 0 \
  --epochs 1 \
  --patience 1 \
  --batch-size 8 \
  --eval-batch-size 8 \
  --train-windows-per-subject 1 \
  --eval-windows-per-subject 1 \
  --cache-workers 4 \
  --output-dir outputs/smoke \
  --device cuda \
  --no-save-model
