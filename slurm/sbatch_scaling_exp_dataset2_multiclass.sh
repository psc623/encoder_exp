#!/bin/bash
#SBATCH --job-name=scaling_exp_d2_mc
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --exclude=j005-ds
#SBATCH --output=/net/projects2/litian-lab/scpan/logs/scaling_exp_d2_mc_%j.log
#
# A2mc/B2mc: 3-class (CN/MCI/AD) version of A2ss/B2ss (dataset2 = ADNI_full
# screening only), MCI added back in. Everything else matches A2ss/B2ss
# exactly: same HD-BET-skull-stripped MASS-native recipe, same 30-epoch/
# patience-15/unbounded-budget finetune protocol, same encoder_learning_rate
# (1e-5, NOT the 5e-4 used for B3 -- kept separate per direct instruction so
# this run isolates "does adding MCI help" from "does the higher LR help").
#
# HD-BET and preprocess_mass_native.py are reused unchanged and idempotent,
# pointed at the SAME output dirs as A2ss/B2ss -- the 431 already-processed
# CN/AD screening volumes are reused as-is; only the ~411 new MCI screening
# volumes get processed.
#
# A2mc: frozen probe, 3-class, 300 epochs (new script: run_probe_adni_mass_
#   multiclass.py, parallel to training.run_probe).
# B2mc: encoder+head jointly finetuned, head warm-started from A2mc (new
#   --warm-start-dir support added to run_finetune_adni_mass_multiclass.py).
#
# Self-chaining, same pattern as the other scaling sbatch scripts.
set -uo pipefail
cd /net/projects2/litian-lab/scpan/encoders
source /opt/conda/etc/profile.d/conda.sh
conda activate /net/projects2/litian-lab/scpan/encoders/med310
export PYTHONUNBUFFERED=1
export PYTHONPATH=/net/projects2/litian-lab/scpan/encoders/src:${PYTHONPATH:-}

SCRIPT_PATH=/net/projects2/litian-lab/scpan/encoders/slurm/sbatch_scaling_exp_dataset2_multiclass.sh
MAX_HOPS=20
HOP_FILE=/net/projects2/litian-lab/scpan/logs/.scaling_exp_d2_mc_hops
HOP=$(cat "$HOP_FILE" 2>/dev/null || echo 0)

RAW_MANIFEST=data/manifests/adni_full_mass2_mc_raw.csv
# Same output dirs as A2ss/B2ss on purpose -- idempotent reuse of the 431
# already-processed CN/AD screening volumes.
SS_DIR=/net/projects2/litian-lab/scpan/dataset/ADNI_full/ADNI_full_skullstripped
SS_MANIFEST=data/manifests/adni_full_mass2_mc_raw_ss.csv
OUT_DIR=/net/projects2/litian-lab/scpan/dataset/ADNI_full/ADNI_full_preprocessed_ss
FINAL_MANIFEST=data/manifests/adni_full_mass2_mc_ss.csv
AUDIT=artifacts/audits/ad/adni_full_mass2_mc_ss_preprocess_audit.json
CONFIG=config/adni_mass_scaling_exp_multiclass.yaml
CACHE=artifacts/cache/ad3/mass_native_d2_mc.npz
A2MC_DIR=artifacts/attention/ad3/mass_native_d2_mc
B2MC_DIR=artifacts/finetune/ad3/mass_native_d2_mc_warmstart

echo "=== hop $HOP started on $(hostname) at $(date -u) job=$SLURM_JOB_ID ==="
nvidia-smi -L

if [ ! -f "$SS_MANIFEST" ] || [ "$(wc -l < "$SS_MANIFEST")" -lt "$(wc -l < "$RAW_MANIFEST")" ]; then
  echo "--- HD-BET skull-stripping (idempotent -- reuses the 431 CN/AD screening volumes) ---"
  python -m encoderbench.bsnip2.hdbet_batch --manifest "$RAW_MANIFEST" \
    --out-dir "$SS_DIR" --out-manifest "$SS_MANIFEST" --device 0 --mode fast || exit 1
else
  echo "--- skull-stripped manifest already covers all rows, reusing: $SS_MANIFEST ---"
fi

echo "--- MASS-native reorient/resample/nonzero-crop (idempotent) ---"
python scripts/preprocess_mass_native.py --manifest "$SS_MANIFEST" --skull-stripped \
  --out-dir "$OUT_DIR" --out-manifest "$FINAL_MANIFEST" --audit "$AUDIT"
STATUS=$?
if [ "$STATUS" -ne 0 ]; then
  echo "=== preprocessing not finished (exit $STATUS) ==="
else
  echo "--- caching MASS native-token features (layer 3), dataset2 3-class ---"
  if [ -f "$CACHE" ]; then
    echo "--- cache already exists, reusing ---"
  else
    encoderbench --config "$CONFIG" cache ad mass --manifest "$FINAL_MANIFEST" \
      --layer 3 --native-tokens --device cuda --out "$CACHE" || STATUS=1
  fi
fi

if [ "$STATUS" -eq 0 ]; then
  A2MC_DONE=1
  for SEED in 0 1 2; do
    [ -f "$A2MC_DIR/probe_seed_${SEED}_summary.json" ] || A2MC_DONE=0
  done
  if [ "$A2MC_DONE" -eq 1 ]; then
    echo "--- A2mc already complete, reusing ---"
  else
    echo "--- A2mc: frozen probe (3-class, native tokens, 300 epochs) ---"
    python scripts/run_probe_adni_mass_multiclass.py --config "$CONFIG" --cache "$CACHE" \
      --seeds 0,1,2 --device cuda --out-dir "$A2MC_DIR" || STATUS=1
  fi
fi

if [ "$STATUS" -eq 0 ]; then
  echo "--- B2mc: joint finetune, head warm-started from A2mc (3-class, native tokens, 30 epochs) ---"
  python scripts/run_finetune_adni_mass_multiclass.py --config "$CONFIG" --manifest "$FINAL_MANIFEST" \
    --cache "$CACHE" --layer 3 --native-tokens --warm-start-dir "$A2MC_DIR" \
    --seeds 0,1,2 --device cuda --out-dir "$B2MC_DIR"
  STATUS=$?
fi

B2MC_DONE=1
for SEED in 0 1 2; do
  [ -f "$B2MC_DIR/finetune_seed_${SEED}_summary.json" ] || B2MC_DONE=0
done

if [ "$STATUS" -eq 0 ] && [ "$B2MC_DONE" -eq 1 ]; then
  echo "=== dataset2-multiclass (A2mc+B2mc) complete at $(date -u) ==="
  rm -f "$HOP_FILE"
  exit 0
fi

NEXT_HOP=$((HOP + 1))
if [ "$NEXT_HOP" -ge "$MAX_HOPS" ]; then
  echo "ERROR: hit $MAX_HOPS resubmission hops without finishing -- stopping, needs human attention" >&2
  exit 1
fi
echo "$NEXT_HOP" > "$HOP_FILE"
echo "=== not finished (exit $STATUS, B2mc done=$B2MC_DONE), self-resubmitting (hop $NEXT_HOP) ==="
sbatch --dependency=afterany:$SLURM_JOB_ID "$SCRIPT_PATH"
