#!/bin/bash
#SBATCH --job-name=bsnip2_brainiac
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --exclude=j005-ds
#SBATCH --output=/net/projects2/litian-lab/scpan/logs/bsnip2_brainiac_%j.log
#
# BSNIP2 raw whole-head T1 -> BrainIAC's real native pipeline: N4 + rigid
# registration to temp_head.nii.gz (CPU, xargs -P parallel, one process per
# volume) then HD-BET skull-strip (GPU, one batched process for all volumes).
# Submitted with sbatch so it survives the submitting shell/VSCode
# disconnecting; every step is idempotent (skips files that already exist),
# so re-submitting resumes rather than restarting.
set -o pipefail
cd /net/projects2/litian-lab/scpan/encoders
source /opt/conda/etc/profile.d/conda.sh
conda activate /net/projects2/litian-lab/scpan/encoders/med310
export PYTHONUNBUFFERED=1

MANIFEST=data/manifests/bsnip2_raw.csv
TEMPLATE=/net/projects2/litian-lab/scpan/BrainIAC/src/preprocessing/atlases/temp_head.nii.gz
REG_DIR=data/bsnip2_brainiac_registered
REG_MANIFEST=data/manifests/bsnip2_brainiac_registered.csv
SS_DIR=data/bsnip2_brainiac_final
FINAL_MANIFEST=data/manifests/bsnip2_brainiac_final.csv

echo "=== bsnip2 brainiac preprocessing on $(hostname) started $(date -u) ==="
nvidia-smi -L
N=$(python -c "import csv; print(len(list(csv.DictReader(open('$MANIFEST')))))")
echo "manifest rows: $N"

echo "--- step 1/3: N4 + rigid registration (CPU, 32-way parallel) ---"
seq 0 $((N - 1)) | xargs -P 32 -I{} python bsnip2_register_brainiac.py \
  --manifest "$MANIFEST" --template "$TEMPLATE" \
  --out-dir "$REG_DIR" --single-index {}
reg_status=$?

echo "--- step 1/3 merge ---"
python bsnip2_register_brainiac.py --manifest "$MANIFEST" --template "$TEMPLATE" \
  --out-dir "$REG_DIR" --out-manifest "$REG_MANIFEST" --merge

echo "--- step 2/3: HD-BET skull-strip (GPU, one batched process) ---"
python bsnip2_hdbet_batch.py --manifest "$REG_MANIFEST" --out-dir "$SS_DIR" \
  --out-manifest "$FINAL_MANIFEST" --device 0 --mode fast

echo "--- step 3/3 summary ---"
wc -l "$FINAL_MANIFEST" 2>/dev/null
echo "=== bsnip2 brainiac preprocessing done $(date -u) ==="
