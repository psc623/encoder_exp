#!/bin/bash
set -euo pipefail
cd /net/projects2/litian-lab/scpan/encoders

echo "=== HOSTNAME: $(hostname) ==="
nvidia-smi -L
python3 -V

VENV=pyenv312
if [ ! -x "$VENV/bin/python" ]; then
  echo "=== creating venv ($VENV) with system python3.12 ==="
  python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"
python -m pip install --upgrade pip -q

if ! python -c "import encoderbench" 2>/dev/null; then
  echo "=== installing encoderbench + deps ==="
  python -m pip install -e '.[test]' -q
fi

set -a
source .env
set +a
export PYTHONUNBUFFERED=1

echo "=== sanity: torch / cuda ==="
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda available:", torch.cuda.is_available())
PY

echo "=== CACHE anatcl (ad) ==="
encoderbench cache ad anatcl --device cuda

echo "=== PROBE anatcl seeds 0,1,2 ==="
for seed in 0 1 2; do
  encoderbench probe ad anatcl --seeds "$seed" --device cuda
done

echo "=== DONE ==="
