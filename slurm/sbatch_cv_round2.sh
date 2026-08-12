#!/bin/bash
#SBATCH --job-name=cv_round2
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=11:00:00
#SBATCH --exclude=j005-ds
#SBATCH --output=/net/projects2/litian-lab/scpan/logs/cv_round2_%j.log
#
# Round 2: sweep the optimizer *step budget* (round 1 only tuned learning rates
# and decays), then re-run all four finetune modes at whichever config wins on
# validation. Motivation: mode 6 has to match a closed-form ridge optimum with
# SGD, and round 1 gave it only 432 optimizer steps to do it in.
#
# Round-1 winner (enc1e4_hwd1) stays in the comparison as the incumbent, so the
# winner is chosen over both rounds rather than only among the new configs.
set -euo pipefail
cd /net/projects2/litian-lab/scpan/encoders
source /opt/conda/etc/profile.d/conda.sh
conda activate /net/projects2/litian-lab/scpan/encoders/med310
export PYTHONUNBUFFERED=1
DRIVER=scripts/run_cv_bsnip2_mass.py

echo "=== round-2 sweep (new step-budget configs + round-1 incumbent) ==="
python scripts/sweep_cv_finetune.py --modes 4,6 \
  --configs enc1e4_hwd1,steps4x,steps4x_ep24_hlr3e3,steps4x_ep24_hwd10_hlr1e2

echo "=== settings selected ==="
cat artifacts/cv/bsnip2/mass/finetune_settings.json

echo "=== re-run all four finetune modes at the selected settings, 5 repeats ==="
for MODE in 4 6 3 5; do
  echo "--- mode${MODE} ---"
  python "$DRIVER" run --mode "$MODE" --repeats all
  python "$DRIVER" report
done

echo "=== round2 done ==="
python "$DRIVER" report
