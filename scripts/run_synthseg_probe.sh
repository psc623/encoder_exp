#!/bin/bash
# Cache the SynthSeg posterior-probability encoder and train its attention probe.
# Requires the mri_synthseg batch run (freesurfer_install/submit_synthseg.sh) to
# have finished and populated dataset/ADNI_processed/synthseg_outputs/post/.
set -euo pipefail
cd /net/projects2/litian-lab/scpan/encoders

source /opt/conda/etc/profile.d/conda.sh
conda activate /net/projects2/litian-lab/scpan/encoders/med310
export PYTHONUNBUFFERED=1

echo "=== HOSTNAME: $(hostname) ==="

echo "=== CACHE synthseg (ad) ==="
encoderbench cache ad synthseg --device cpu

echo "=== PROBE synthseg seeds 0,1,2 ==="
for seed in 0 1 2; do
  encoderbench probe ad synthseg --seeds "$seed" --device cpu
done

echo "=== DONE ==="
