#!/bin/bash
#SBATCH --job-name=scaling_improve_topup
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --exclude=j005-ds
#SBATCH --output=/net/projects2/litian-lab/scpan/logs/scaling_improve_topup_%j.log
#
# Tops every improved-protocol cell up to the full seed list declared in
# config/adni_mass_scaling_exp_improve.yaml (currently [0,1,2,3,4]).
#
# Why more seeds: three seeds cannot resolve the differences this series is
# comparing. On the 65-subject dataset2 validation split the standard error of
# AUC is about 0.034; the B2ss70 selection ambiguity that motivated this
# top-up (raw AUC peaks at epoch 8 at 0.9505, the trailing-smoothed rule picks
# epoch 17 at 0.9300) is a gap of 0.02 -- well inside one standard error. Five
# seeds do not fix that, but they do say whether the *direction* of each
# A-vs-B and dataset2-vs-dataset4 delta is consistent or is sign noise.
#
# Only the missing seeds are computed. run_probe and run_finetune are both
# deterministic given a seed, so seeds already on disk are skipped rather than
# recomputed to identical bytes. Data, manifests, splits and feature caches are
# untouched, exactly as in sbatch_scaling_improve.sh.
#
# Submit AFTER sbatch_scaling_improve.sh so the two never contend for the GPU:
#   sbatch --dependency=afterany:<jobid> slurm/sbatch_scaling_improve_topup.sh
#
# Self-chaining, same pattern as the other scaling sbatch scripts.
set -uo pipefail
cd /net/projects2/litian-lab/scpan/encoders
source /opt/conda/etc/profile.d/conda.sh
conda activate /net/projects2/litian-lab/scpan/encoders/med310
export PYTHONUNBUFFERED=1
export PYTHONPATH=/net/projects2/litian-lab/scpan/encoders/src:${PYTHONPATH:-}

SCRIPT_PATH=/net/projects2/litian-lab/scpan/encoders/slurm/sbatch_scaling_improve_topup.sh
MAX_HOPS=25
HOP_FILE=/net/projects2/litian-lab/scpan/logs/.scaling_improve_topup_hops
HOP=$(cat "$HOP_FILE" 2>/dev/null || echo 0)

CONFIG=config/adni_mass_scaling_exp_improve.yaml
D2_MANIFEST=data/manifests/adni_full_mass_d2_reshuffled_ss.csv
D2_CACHE=artifacts/cache/ad/mass_native_d2ss70.npz
D2_A_DIR=artifacts/attention/ad/improve_d2ss70
D2_B_DIR=artifacts/finetune/ad/improve_d2ss70_warmstart
D4_MANIFEST=data/manifests/adni_full_mass4_ss.csv
D4_CACHE=artifacts/cache/ad/mass_native_d4.npz
D4_A_DIR=artifacts/attention/ad/improve_d4
D4_B_DIR=artifacts/finetune/ad/improve_d4_warmstart

echo "=== hop $HOP started on $(hostname) at $(date -u) job=$SLURM_JOB_ID ==="
nvidia-smi -L

# The seed list is the config's, not a literal here, so bumping the config is
# the only place a seed count has to change.
SEEDS=$(python - "$CONFIG" <<'PY'
import sys, yaml
raw = yaml.safe_load(open(sys.argv[1]))
probe, finetune = raw["probe"]["seeds"], raw["finetune"]["seeds"]
assert probe == finetune, f"probe seeds {probe} != finetune seeds {finetune}"
print(",".join(str(s) for s in probe))
PY
) || { echo "ERROR: could not read the seed list from $CONFIG" >&2; exit 1; }
echo "--- target seed list from $CONFIG: $SEEDS ---"

for REQUIRED in "$D2_MANIFEST" "$D2_CACHE" "$D4_MANIFEST" "$D4_CACHE"; do
  if [ ! -f "$REQUIRED" ]; then
    echo "ERROR: required input missing, refusing to rebuild it: $REQUIRED" >&2
    exit 1
  fi
done
echo "--- reusing existing manifests and feature caches (data unchanged) ---"

STATUS=0

# Returns the comma-separated seeds that have no summary JSON yet, or empty.
missing_seeds () {
  local DIR=$1 PREFIX=$2 OUT=""
  for SEED in ${SEEDS//,/ }; do
    if [ ! -f "$DIR/${PREFIX}_seed_${SEED}_summary.json" ]; then
      OUT="${OUT:+$OUT,}$SEED"
    fi
  done
  echo "$OUT"
}

topup_probe () {
  local NAME=$1 CACHE=$2 DIR=$3
  local TODO
  TODO=$(missing_seeds "$DIR" probe)
  if [ -z "$TODO" ]; then
    echo "--- $NAME: all seeds ($SEEDS) already present, nothing to do ---"
    return 0
  fi
  echo "--- $NAME: running missing probe seeds $TODO ---"
  encoderbench --config "$CONFIG" probe ad mass --cache "$CACHE" \
    --seeds "$TODO" --device cuda --out-dir "$DIR"
}

topup_finetune () {
  local NAME=$1 MANIFEST=$2 CACHE=$3 A_DIR=$4 B_DIR=$5
  local TODO
  TODO=$(missing_seeds "$B_DIR" finetune)
  if [ -z "$TODO" ]; then
    echo "--- $NAME: all seeds ($SEEDS) already present, nothing to do ---"
    return 0
  fi
  echo "--- $NAME: running missing finetune seeds $TODO (warm-started from $A_DIR) ---"
  encoderbench --config "$CONFIG" finetune ad mass --manifest "$MANIFEST" \
    --cache "$CACHE" --layer 3 --native-tokens --warm-start-dir "$A_DIR" \
    --seeds "$TODO" --device cuda --out-dir "$B_DIR"
}

# Probes first within each dataset: the finetune cells warm-start from them.
[ "$STATUS" -eq 0 ] && { topup_probe    "A2ss70" "$D2_CACHE" "$D2_A_DIR" || STATUS=1; }
[ "$STATUS" -eq 0 ] && { topup_finetune "B2ss70" "$D2_MANIFEST" "$D2_CACHE" "$D2_A_DIR" "$D2_B_DIR" || STATUS=1; }
[ "$STATUS" -eq 0 ] && { topup_probe    "A4"     "$D4_CACHE" "$D4_A_DIR" || STATUS=1; }
[ "$STATUS" -eq 0 ] && { topup_finetune "B4"     "$D4_MANIFEST" "$D4_CACHE" "$D4_A_DIR" "$D4_B_DIR" || STATUS=1; }

ALL_DONE=1
[ -n "$(missing_seeds "$D2_A_DIR" probe)" ]    && ALL_DONE=0
[ -n "$(missing_seeds "$D2_B_DIR" finetune)" ] && ALL_DONE=0
[ -n "$(missing_seeds "$D4_A_DIR" probe)" ]    && ALL_DONE=0
[ -n "$(missing_seeds "$D4_B_DIR" finetune)" ] && ALL_DONE=0

if [ "$STATUS" -eq 0 ] && [ "$ALL_DONE" -eq 1 ]; then
  echo "--- regenerating report/scaling_exp_improve/scaling_improve_report.md over all seeds ---"
  python scripts/generate_scaling_improve_report.py
  echo "=== top-up complete at $(date -u), every cell has seeds $SEEDS ==="
  rm -f "$HOP_FILE"
  exit 0
fi

NEXT_HOP=$((HOP + 1))
if [ "$NEXT_HOP" -ge "$MAX_HOPS" ]; then
  echo "ERROR: hit $MAX_HOPS resubmission hops without finishing -- stopping, needs human attention" >&2
  exit 1
fi
echo "$NEXT_HOP" > "$HOP_FILE"
echo "=== not finished (exit $STATUS, all done=$ALL_DONE), self-resubmitting (hop $NEXT_HOP) ==="
sbatch --dependency=afterany:$SLURM_JOB_ID "$SCRIPT_PATH"
