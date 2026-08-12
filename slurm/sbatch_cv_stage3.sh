#!/bin/bash
#SBATCH --job-name=cv_stage3
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --exclude=j005-ds
#SBATCH --output=/net/projects2/litian-lab/scpan/logs/cv_stage3_%j.log
#
# Stage 3: the four finetune modes at the swept settings, 5 repeats each,
# under the repeated-split protocol. Reads artifacts/cv/bsnip2/mass/
# finetune_settings.json (written by stage 2) automatically via
# run_cv_bsnip2_mass.tuned_settings(). Ordered 4,6,3,5 so the two linear modes
# -- the ones that have to clear mode2's strong closed-form baseline -- land
# first and are visible earliest.
set -euo pipefail
cd /net/projects2/litian-lab/scpan/encoders
source /opt/conda/etc/profile.d/conda.sh
conda activate /net/projects2/litian-lab/scpan/encoders/med310
export PYTHONUNBUFFERED=1
DRIVER=scripts/run_cv_bsnip2_mass.py

echo "=== settings in use ==="
cat artifacts/cv/bsnip2/mass/finetune_settings.json 2>/dev/null || echo "(none -- config defaults)"

for MODE in 4 6 3 5; do
  echo "=== mode${MODE} x5 repeats ==="
  python "$DRIVER" run --mode "$MODE" --repeats all
  python "$DRIVER" report
done

echo "=== stage3 done ==="
python "$DRIVER" report
