#!/usr/bin/env bash
# CV01 pooled-vision B1/B2 operator script. Run from repository root.
# Nothing runs on GPU until an operator explicitly calls smoke/pilot/full.
set -euo pipefail
ACTION="${1:-}"
case "$ACTION" in
  test|preflight|smoke-b1|smoke-b2|pilot-b1|pilot-b2|full-b1|full-b2|resume-b1|resume-b2) ;;
  *) echo "Usage: bash scripts/cp2_mm_run.sh {test|preflight|smoke-b1|smoke-b2|pilot-b1|pilot-b2|full-b1|full-b2|resume-b1|resume-b2}"; exit 2 ;;
esac
if [[ ! -f src/cp2/mm_runner.py || ! -f src/master_workflow.py ]]; then
  echo "ERROR: run from root of CP2 implementation branch"; exit 2
fi
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1
export CP2_WORKSPACE="${CP2_WORKSPACE:-$HOME/kelompok-7-private/cp2_data}"
export CP2_OUT="${CP2_OUT:-$HOME/kelompok-7-private/cp2_runs}"
if [[ "$ACTION" == "test" ]]; then
  python -m pytest -q tests/test_cp2_nlp.py tests/test_cp2_mm.py
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
mkdir -p "$CP2_WORKSPACE" "$CP2_OUT"
case "$(realpath "$CP2_OUT")" in
  "$(realpath .)"/*) echo "ERROR: private output directory must be outside the repository"; exit 2 ;;
esac
case "$ACTION" in
  *b1) VARIANT=B1 ;;
  *b2) VARIANT=B2 ;;
esac
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
