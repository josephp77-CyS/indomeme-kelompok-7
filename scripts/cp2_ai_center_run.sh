#!/usr/bin/env bash
# Execute in a JupyterLab Terminal on the authorized AI Data Center account.
# Never copies credentials or runs background tasks without operator action.
set -euo pipefail
case "${1:-}" in
  preflight|test|init-data|audit|calibrate-attention|calibrate-b0|calibrate-vision|smoke-b0|pilot-b0|full-b0) ;;
  *) echo "Usage: bash scripts/cp2_ai_center_run.sh {preflight|test|init-data|audit|calibrate-attention|calibrate-b0|calibrate-vision|smoke-b0|pilot-b0|full-b0}"; exit 2;;
esac
ACTION="$1"
if [[ ! -f src/cp2/nlp_runner.py || ! -f src/master_workflow.py ]]; then
  echo 'ERROR: run from repo root with both CP1 and CP2 source'; exit 2
fi
export PYTHONPATH="${PWD}/src"
export PYTHONNOUSERSITE=1
export CP2_WORKSPACE="${CP2_WORKSPACE:-$HOME/kelompok-7-private/cp2_data}"
export CP2_OUT="${CP2_OUT:-$HOME/kelompok-7-private/cp2_runs}"
case "$ACTION" in
  preflight)
    echo "Git branch: $(git branch --show-current 2>/dev/null || echo unknown)"
    echo "Git SHA: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
    python --version
    command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=index,name,memory.free,memory.total,utilization.gpu --format=csv,noheader || echo 'No nvidia-smi detected'
    df -h "$HOME"; free -h || true
    python - <<'PY'
import torch
print('Torch:', torch.__version__, 'CUDA:', torch.version.cuda, 'available:', torch.cuda.is_available())
PY
    ;;
  test) python -m pytest -q tests; python verify_submission.py;;
  init-data) python -m cp2.nlp_runner init-data --workspace "$CP2_WORKSPACE";;
  audit) python -m cp2.nlp_runner audit --workspace "$CP2_WORKSPACE" --output "$CP2_OUT/audits";;
  calibrate-attention) python -m cp2.nlp_runner calibrate --component attention --batch 1 --output "$CP2_OUT/calibration" --require-gpu;;
  calibrate-b0) python -m cp2.nlp_runner calibrate --component b0 --batch 1 --output "$CP2_OUT/calibration" --require-gpu;;
  calibrate-vision) python -m cp2.nlp_runner calibrate --component vision --batch 1 --output "$CP2_OUT/calibration" --require-gpu;;
  smoke-b0) python -m cp2.nlp_runner train-b0 --workspace "$CP2_WORKSPACE" --output "$CP2_OUT" --seed 42 --epochs 1 --max-train 32 --max-validation 16 --require-gpu;;
  pilot-b0) python -m cp2.nlp_runner train-b0 --workspace "$CP2_WORKSPACE" --output "$CP2_OUT" --seed 42 --pilot-full-data --epochs 1 --require-gpu;;
  full-b0)
    [[ "${CP2_CONFIRM_FULL:-}" == 'YES' ]] || { echo 'Set CP2_CONFIRM_FULL=YES only after approval and completed smoke/pilot'; exit 3; }
    python -m cp2.nlp_runner train-b0 --workspace "$CP2_WORKSPACE" --output "$CP2_OUT" --seed 42 --full --epochs 5 --require-gpu
    ;;
esac
