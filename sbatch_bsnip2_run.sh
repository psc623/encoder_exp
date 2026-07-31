#!/bin/bash
#SBATCH --job-name=bsnip2_run
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --exclude=j005-ds
#SBATCH --output=/net/projects2/litian-lab/scpan/logs/bsnip2_run_%x_%j.log
#
# BSNIP2 SZ/HC transfer of the frozen-encoder attention-probe + budgeted
# fine-tune protocol, unchanged from the ADNI AD/CN run (same probe grid,
# same 8M finetune budget, same seeds 0/1/2): cache once, then probe with
# and without shuffled labels (the shuffled run is the overfitting negative
# control -- reported, never used to roll back the real run), then finetune
# starting from the same registered pretrained checkpoint the probe used.
# Submitted with sbatch so it survives the submitting shell/VSCode
# disconnecting; cache/probe/finetune are each idempotent per encoder.
set -o pipefail
cd /net/projects2/litian-lab/scpan/encoders
source /opt/conda/etc/profile.d/conda.sh
conda activate /net/projects2/litian-lab/scpan/encoders/med310
set -a
source .env
set +a
export PYTHONUNBUFFERED=1

ENC=${ENCODER:?set ENCODER=<name>}
MANIFEST=data/manifests/bsnip2_${ENC}.csv
CACHE=artifacts/cache/bsnip2/${ENC}.npz

echo "=== bsnip2 $ENC on $(hostname) started $(date -u) ==="
nvidia-smi -L

echo "--- cache ---"
encoderbench cache bsnip2 "$ENC" --manifest "$MANIFEST" --out "$CACHE" --device cuda || exit 1

echo "--- probe (seeds 0,1,2) ---"
encoderbench probe bsnip2 "$ENC" --cache "$CACHE" --seeds all --device cuda || exit 1

echo "--- probe shuffled-label negative control (seeds 0,1,2) ---"
encoderbench probe bsnip2 "$ENC" --cache "$CACHE" --seeds all --shuffled-labels --device cuda || exit 1

echo "--- finetune (seeds 0,1,2) ---"
encoderbench finetune bsnip2 "$ENC" --manifest "$MANIFEST" --cache "$CACHE" --seeds all --device cuda || exit 1

echo "=== bsnip2 $ENC done $(date -u) ==="
