#!/bin/bash
#SBATCH --job-name=ad_synthseg_finetune
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --exclude=j005-ds
#SBATCH --output=/net/projects2/litian-lab/scpan/logs/ad_synthseg_finetune_%j.log
#
# ADNI SynthSeg "finetune": SynthSeg's own segmentation network is a frozen
# external tool with no differentiable parameters, so this trains the small
# post-pooling residual adapter (SynthSegPosteriorAdapter, see extractors.py)
# instead -- same seeds (0/1/2), epochs, optimizer, and schedule as every other
# encoder's finetune (config/finetune section), just a different (much
# smaller, deliberately so -- see config/default.yaml's synthseg_adapter_
# hidden_size comment) trainable parameter group. Uses the ADNI SynthSeg cache
# already on disk (artifacts/cache/ad/synthseg.npz) only for normalization
# statistics, same as every other finetune run.
set -o pipefail
cd /net/projects2/litian-lab/scpan/encoders
source /opt/conda/etc/profile.d/conda.sh
conda activate /net/projects2/litian-lab/scpan/encoders/med310
export PYTHONUNBUFFERED=1

echo "=== ad synthseg finetune on $(hostname) started $(date -u) ==="
nvidia-smi -L

encoderbench finetune ad synthseg --seeds all --device cuda || exit 1

echo "=== ad synthseg finetune done $(date -u) ==="
