#!/bin/bash
#SBATCH --job-name=finetune_rerun
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --exclude=j005-ds
#SBATCH --output=/net/projects2/litian-lab/scpan/logs/finetune_rerun_%x_%j.log
#
# Generic rerun of `encoderbench finetune` for one disease/encoder using the
# existing cache + manifest already on disk (no cache/probe rebuild needed --
# those already ran and their probe checkpoints are exactly what this rerun
# warm-starts the head from, per finetune.py's _warm_start_head). Every other
# setting (seeds, budgets, epochs) is unchanged from the original finetune
# runs, so results stay comparable except for the one thing being tested: a
# head initialized from probe instead of from scratch.
set -o pipefail
cd /net/projects2/litian-lab/scpan/encoders
source /opt/conda/etc/profile.d/conda.sh
conda activate /net/projects2/litian-lab/scpan/encoders/med310
export PYTHONUNBUFFERED=1

DISEASE=${DISEASE:?set DISEASE=ad|bsnip2}
ENC=${ENCODER:?set ENCODER=<name>}
CONFIG_ARG=()
if [ -n "${CONFIG:-}" ]; then CONFIG_ARG=(--config "$CONFIG"); fi
MANIFEST_ARG=()
if [ -n "${MANIFEST:-}" ]; then MANIFEST_ARG=(--manifest "$MANIFEST"); fi
CACHE_ARG=()
if [ -n "${CACHE:-}" ]; then CACHE_ARG=(--cache "$CACHE"); fi

echo "=== finetune rerun (warm-started head) $DISEASE:$ENC on $(hostname) started $(date -u) ==="
nvidia-smi -L

encoderbench "${CONFIG_ARG[@]}" finetune "$DISEASE" "$ENC" "${MANIFEST_ARG[@]}" "${CACHE_ARG[@]}" \
  --seeds all --device cuda || exit 1

echo "=== finetune rerun $DISEASE:$ENC done $(date -u) ==="
