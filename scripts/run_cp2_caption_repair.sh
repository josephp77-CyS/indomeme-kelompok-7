#!/usr/bin/env bash
set -euo pipefail
umask 077

workspace="${1:?workspace path required}"
study="${2:?source study path required}"
job="${3:?caption repair path required}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1

exec 9>"$job/.repair-pipeline.lock"
flock -n 9
trap 'printf "%s\n" "$?" >"$job/pipeline_exit"' EXIT

stage() {
  printf "%s %s\n" "$(date -Iseconds)" "$1" | tee "$job/pipeline_stage"
}

stage awaiting_caption_generation
until [[ -f "$job/generation_exit" ]]; do
  if ! kill -0 "$(cat "$job/generation_pid")" 2>/dev/null; then
    printf "Caption generation process stopped without an exit receipt\n" >&2
    exit 1
  fi
  sleep 15
done
[[ "$(cat "$job/generation_exit")" == 0 ]]

predict=(python -u -m cp2.repair_predictions --workspace "$workspace" --study "$study" --job "$job")
run_router() {
  for split in validation test; do
    stage "router_predictions_$split"
    "${predict[@]}" --split "$split" || return
  done
  stage fitting_and_replaying_routers
  if [[ ! -f "$job/router_results.json" ]]; then
    python -u -m cp2.repair_router --study "$study" --job "$job" || return
  fi
}
router_exit=0
run_router || router_exit=$?
printf "%s\n" "$router_exit" >"$job/router_exit"

for seed in 42 123 2026; do
  run="$job/runs/S3_full_seed$seed"
  [[ -f "$run/result.json" ]] && continue
  stage "full_training_S3_seed$seed"
  resume=()
  [[ ! -f "$run/last.pt" ]] || resume=(--resume)
  python -u -m cp2.mm_runner --workspace "$workspace" --output "$job/runs" --variant S3 \
    --repair-captions "$job/fit_captions_PRIVATE.json" --epochs 5 --full --require-gpu --seed "$seed" "${resume[@]}"
done

for split in validation test; do
  stage "classifier_predictions_$split"
  "${predict[@]}" --split "$split" --new-model
done
stage summarizing_classifier_comparison
python -u -m cp2.repair_classifier_summary --study "$study" --job "$job"
[[ "$router_exit" == 0 ]]
stage complete
