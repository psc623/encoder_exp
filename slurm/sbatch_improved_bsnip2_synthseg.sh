#!/bin/bash
#SBATCH --job-name=improved_bsnip2_synthseg
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --exclude=j005-ds
#SBATCH --output=/net/projects2/litian-lab/scpan/logs/improved_bsnip2_synthseg_%j.log
#
# SynthSeg/bsnip2 counterpart of sbatch_improved_bsnip2_mass.sh -- same 6
# modes, same conditional sweep-if-BA<0.65 logic, same incremental-report-
# after-every-step. mode1/mode2 already measured near-chance in smoke testing
# (BA ~0.52/0.54), so the sweep branch is expected to fire this time. Writes
# to improved_report_synthseg_section.md, NOT improved_report.md directly (see
# run_improved_bsnip2_synthseg.py's docstring) -- merge manually once done.
set -euo pipefail
cd /net/projects2/litian-lab/scpan/encoders
source /opt/conda/etc/profile.d/conda.sh
conda activate /net/projects2/litian-lab/scpan/encoders/med310
export PYTHONUNBUFFERED=1
DRIVER=scripts/run_improved_bsnip2_synthseg.py
LOG_DIR=/net/projects2/litian-lab/scpan/logs
BA_CHECK=scripts/check_ba_threshold_synthseg.py

echo "=== mode1 (frozen attention) ==="
python "$DRIVER" mode1 --seeds all 2>&1 | tee "$LOG_DIR/improved_synthseg_mode1_${SLURM_JOB_ID}.log"
python "$DRIVER" report

echo "=== mode2 (frozen linear) ==="
python "$DRIVER" mode2 --seeds all 2>&1 | tee "$LOG_DIR/improved_synthseg_mode2_${SLURM_JOB_ID}.log"
python "$DRIVER" report

echo "=== checking whether mode1/mode2 mean BA < 0.65 ==="
if ! python "$BA_CHECK"; then
  echo "=== BA below 0.65 -> running hyperparameter sweep (seed 0 only) ==="
  python "$DRIVER" sweep 2>&1 | tee "$LOG_DIR/improved_synthseg_sweep_${SLURM_JOB_ID}.log"
  echo "=== re-running mode1/mode2 with swept hyperparameters ==="
  python "$DRIVER" mode1 --seeds all 2>&1 | tee -a "$LOG_DIR/improved_synthseg_mode1_${SLURM_JOB_ID}.log"
  python "$DRIVER" report
  python "$DRIVER" mode2 --seeds all 2>&1 | tee -a "$LOG_DIR/improved_synthseg_mode2_${SLURM_JOB_ID}.log"
  python "$DRIVER" report
else
  echo "=== BA >= 0.65 for both -> no sweep needed ==="
fi

echo "=== modes 3-6 (fine-tune), sequential, report updated after each ==="
FAILED=0
set +e
for MODE in mode3 mode4 mode5 mode6; do
  echo "--- $MODE ---"
  python "$DRIVER" "$MODE" --seeds all 2>&1 | tee "$LOG_DIR/improved_synthseg_${MODE}_${SLURM_JOB_ID}.log"
  if [ "${PIPESTATUS[0]}" -ne 0 ]; then
    echo "!!! $MODE failed, continuing with the rest"
    FAILED=1
  fi
  python "$DRIVER" report
done
set -e

if [ "$FAILED" -ne 0 ]; then
  echo "=== one or more modes failed; improved_report_synthseg_section.md reflects whatever completed ==="
  exit 1
fi
echo "=== all modes completed successfully ==="
