#!/usr/bin/env bash
# CV01 B1–M3 operator script. Run from repository root.
# Nothing runs on GPU until an operator explicitly calls smoke/pilot/full.
set -euo pipefail
ACTION="${1:-}"
if [[ "$ACTION" != test && "$ACTION" != preflight && ! "$ACTION" =~ ^(smoke|pilot|full|resume)-(b[12]|m[123])$ ]]; then
  echo "Usage: bash scripts/cp2_mm_run.sh {test|preflight|{smoke,pilot,full,resume}-{b1,b2,m1,m2,m3}}"
  exit 2
fi
if [[ ! -f src/cp2/mm_runner.py || ! -f src/master_workflow.py ]]; then
  echo "ERROR: run from root of CP2 implementation branch"; exit 2
fi
export PYTHONPATH="$PWD/src"
export PYTHONNOUSERSITE=1
export CP2_WORKSPACE="${CP2_WORKSPACE:-$HOME/kelompok-7-private/cp2_data}"
export CP2_OUT="${CP2_OUT:-$HOME/kelompok-7-private/cp2_runs}"
if [[ "$ACTION" == "test" ]]; then
  python -m pytest -q tests
  exit 0
fi
if [[ "$ACTION" == "preflight" ]]; then
  python - <<'PY'
import torch
from pathlib import Path
print("PyTorch:", torch.__version__, "CUDA:", torch.cuda.is_available())
if torch.cuda.is_available(): print("GPU:", torch.cuda.get_device_name(0))
PY
  command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=index,name,memory.free,memory.total,utilization.gpu --format=csv,noheader || :
  df -h "$HOME"
  exit 0
fi
VARIANT="${ACTION#*-}"
VARIANT="${VARIANT^^}"
ARGS=(--variant "$VARIANT" --workspace "$CP2_WORKSPACE" --output "$CP2_OUT"
      --seed "${CP2_SEED:-42}" --require-gpu)
case "$ACTION" in
  smoke-*) ARGS+=(--epochs 1 --max-train 32 --max-validation 16) ;;
  pilot-*) ARGS+=(--epochs 1 --pilot-full-data) ;;
  full-*)
    [[ "${CP2_CONFIRM_MM_FULL:-}" == YES ]] || { echo "ERROR: Set CP2_CONFIRM_MM_FULL=YES only after CV review, tests, smoke, pilot and GPU approval"; exit 3; }
    ARGS+=(--full --epochs 5) ;;
  resume-*)
    [[ "${CP2_CONFIRM_MM_FULL:-}" == YES ]] || { echo "ERROR: review existing checkpoint and set CP2_CONFIRM_MM_FULL=YES"; exit 3; }
    ARGS+=(--full --epochs 5 --resume) ;;
esac
python -m cp2.mm_runner "${ARGS[@]}"
