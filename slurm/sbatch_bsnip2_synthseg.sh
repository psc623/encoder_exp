#!/bin/bash
#SBATCH --job-name=bsnip2_synthseg
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --exclude=j005-ds
#SBATCH --output=/net/projects2/litian-lab/scpan/logs/bsnip2_synthseg_%j.log
#
# BSNIP2 SZ/HC counterpart of the ADNI SynthSeg runs, same protocol as
# slurm/sbatch_bsnip2_run.sh (cache once, probe with/without shuffled labels, then
# finetune -- same seeds 0/1/2, same shared probe/finetune settings). Uses
# config/bsnip2_synthseg.yaml, which is byte-identical to config/default.yaml
# except checkpoints.synthseg (points at the BSNIP2 mri_synthseg posteriors
# instead of ADNI's) and manifests.bsnip2 (points at bsnip2_synthseg.csv);
# config.py's _validate() enforces every frozen field regardless, so the run
# stays comparable to ad:synthseg and to the other bsnip2:<encoder> runs.
# "Finetune" here trains SynthSegPosteriorAdapter, the same small post-pooling
# residual MLP used for ad:synthseg (see extractors.py), not SynthSeg's own
# segmentation network, which has no differentiable parameters.
set -o pipefail
cd /net/projects2/litian-lab/scpan/encoders
source /opt/conda/etc/profile.d/conda.sh
conda activate /net/projects2/litian-lab/scpan/encoders/med310
export PYTHONUNBUFFERED=1

CONFIG=config/bsnip2_synthseg.yaml
MANIFEST=data/manifests/bsnip2_synthseg.csv
CACHE=artifacts/cache/bsnip2/synthseg.npz

echo "=== bsnip2 synthseg on $(hostname) started $(date -u) ==="
nvidia-smi -L

echo "--- cache ---"
encoderbench --config "$CONFIG" cache bsnip2 synthseg --manifest "$MANIFEST" --out "$CACHE" --device cuda || exit 1

echo "--- probe (seeds 0,1,2) ---"
encoderbench --config "$CONFIG" probe bsnip2 synthseg --cache "$CACHE" --seeds all --device cuda || exit 1

echo "--- probe shuffled-label negative control (seeds 0,1,2) ---"
encoderbench --config "$CONFIG" probe bsnip2 synthseg --cache "$CACHE" --seeds all --shuffled-labels --device cuda || exit 1

echo "--- finetune (seeds 0,1,2) ---"
encoderbench --config "$CONFIG" finetune bsnip2 synthseg --manifest "$MANIFEST" --cache "$CACHE" --seeds all --device cuda || exit 1

echo "=== bsnip2 synthseg done $(date -u) ==="
