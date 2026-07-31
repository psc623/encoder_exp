#!/bin/bash
set -uo pipefail
cd /net/projects2/litian-lab/scpan/encoders
source pyenv312/bin/activate

TEMPLATE=/net/projects2/litian-lab/scpan/BrainIAC/src/preprocessing/atlases/temp_head.nii.gz
MANIFEST=data/manifests/adni.csv
OUT_DIR=data/brainiac_registered
OUT_MANIFEST=data/manifests/adni_brainiac_registered.csv

N=$(python -c "import csv; print(len(list(csv.DictReader(open('$MANIFEST')))))")
echo "=== registering $N volumes, $(date) ==="

seq 0 $((N - 1)) | xargs -P 32 -I{} python scripts/register_brainiac_template.py \
  --manifest "$MANIFEST" --template "$TEMPLATE" \
  --out-dir "$OUT_DIR" --out-manifest unused --single-index {}

echo "=== registration pass done, $(date) ==="
echo "=== merging manifest ==="
python scripts/register_brainiac_template.py \
  --manifest "$MANIFEST" --template "$TEMPLATE" \
  --out-dir "$OUT_DIR" --out-manifest "$OUT_MANIFEST" --merge
