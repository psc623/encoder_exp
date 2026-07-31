#!/bin/bash
set -euo pipefail
cd /net/projects2/litian-lab/scpan/encoders
source pyenv312/bin/activate
set -a
source .env
set +a
export PYTHONUNBUFFERED=1

echo "=== HOSTNAME: $(hostname) ==="
nvidia-smi -L

echo "=== CACHE brainiac (ad, registered manifest) ==="
encoderbench cache ad brainiac \
  --manifest data/manifests/adni_brainiac_registered.csv \
  --out artifacts/cache/ad/brainiac_registered.npz \
  --device cuda

echo "=== PROBE brainiac seeds 0,1,2 ==="
for seed in 0 1 2; do
  encoderbench probe ad brainiac \
    --cache artifacts/cache/ad/brainiac_registered.npz \
    --seeds "$seed" \
    --out-dir artifacts/attention/ad/brainiac_registered \
    --device cuda
done

echo "=== DONE ==="
