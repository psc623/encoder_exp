#!/bin/bash
#SBATCH --job-name=bsnip2_anatcl
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --exclude=j005-ds
#SBATCH --output=/net/projects2/litian-lab/scpan/logs/bsnip2_anatcl_%j.log
#
# AnatCL never ran on BSNIP2 before. Reuses BrainIAC's already registered +
# skull-stripped bsnp2 images (bsnip2_finalize_anatcl_manifest.py) as input --
# no new preprocessing needed. Same protocol as sbatch_bsnip2_run.sh: cache
# once, probe with/without shuffled labels, then finetune -- and finetune now
# warm-starts its head from this run's own probe checkpoint (finetune.py's
# _warm_start_head), so this AnatCL result is directly comparable to the other
# 4 encoders' warm-started finetune reruns from the start, no separate re-run
# needed later.
set -o pipefail
cd /net/projects2/litian-lab/scpan/encoders
source /opt/conda/etc/profile.d/conda.sh
conda activate /net/projects2/litian-lab/scpan/encoders/med310
export PYTHONUNBUFFERED=1

MANIFEST=data/manifests/bsnip2_anatcl.csv
CACHE=artifacts/cache/bsnip2/anatcl.npz

echo "=== bsnip2 anatcl on $(hostname) started $(date -u) ==="
nvidia-smi -L

echo "--- cache ---"
encoderbench cache bsnip2 anatcl --manifest "$MANIFEST" --out "$CACHE" --device cuda || exit 1

echo "--- probe (seeds 0,1,2) ---"
encoderbench probe bsnip2 anatcl --cache "$CACHE" --seeds all --device cuda || exit 1

echo "--- probe shuffled-label negative control (seeds 0,1,2) ---"
encoderbench probe bsnip2 anatcl --cache "$CACHE" --seeds all --shuffled-labels --device cuda || exit 1

echo "--- finetune (seeds 0,1,2, head warm-started from probe) ---"
encoderbench finetune bsnip2 anatcl --manifest "$MANIFEST" --cache "$CACHE" --seeds all --device cuda || exit 1

echo "=== bsnip2 anatcl done $(date -u) ==="
