#!/bin/bash
#SBATCH --job-name=bsnip2_medsiglip
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --exclude=j005-ds
#SBATCH --output=/net/projects2/litian-lab/scpan/logs/bsnip2_medsiglip_%j.log
#
# BSNIP2 raw whole-head T1 -> skull-strip only (HD-BET), no atlas
# registration. medsiglip is a generic 2D vision-language encoder with no
# native 3D registration recipe of its own, but its shared normalize_volume
# computes intensity percentiles over the whole array -- on a raw whole head
# that window gets pulled by scalp/skull/eyes, something ADNI's already
# skull-stripped input never exposed it to. This is deliberately the
# lightest of the three BSNIP2 preprocessing jobs. Submitted with sbatch so
# it survives the submitting shell/VSCode disconnecting; idempotent.
set -o pipefail
cd /net/projects2/litian-lab/scpan/encoders
source /opt/conda/etc/profile.d/conda.sh
conda activate /net/projects2/litian-lab/scpan/encoders/med310
export PYTHONUNBUFFERED=1

MANIFEST=data/manifests/bsnip2_raw.csv
OUT_DIR=data/bsnip2_medsiglip_final
FINAL_MANIFEST=data/manifests/bsnip2_medsiglip_final.csv

echo "=== bsnip2 medsiglip preprocessing on $(hostname) started $(date -u) ==="
nvidia-smi -L

echo "--- HD-BET skull-strip (GPU, one batched process) ---"
python bsnip2_hdbet_batch.py --manifest "$MANIFEST" --out-dir "$OUT_DIR" \
  --out-manifest "$FINAL_MANIFEST" --device 0 --mode fast

wc -l "$FINAL_MANIFEST" 2>/dev/null
echo "=== bsnip2 medsiglip preprocessing done $(date -u) ==="
