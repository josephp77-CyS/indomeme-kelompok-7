#!/usr/bin/env bash
set -euo pipefail
umask 077

: "${CP2_STUDY_ROOT:?Set a private output folder with completed baseline benchmark}"
: "${CP2_WORKSPACE:?Set the pinned private INDOMEME workspace}"
export PYTHONPATH="$PWD/src" PYTHONNOUSERSITE=1
study="$CP2_STUDY_ROOT"
test "$(cat "$study/baseline_exit")" = 0
seeds=$(python -c 'import json; print(" ".join(map(str,json.load(open("configs/cp2_source_reliability.json"))["seeds"])))')

for model in S0 S1 S2; do
    printf 'smoke-%s\n' "$model" > "$study/training_stage"
    if [ ! -s "$study/runs/${model}_smoke_seed42/result.json" ]; then
        python -u -m cp2.mm_runner --variant "$model" --workspace "$CP2_WORKSPACE" \
            --output "$study/runs" --seed 42 --max-train 32 --max-validation 16 --require-gpu
    fi
done

for seed in $seeds; do
    models=(S1 S2 S0)
    if [ "$seed" != 42 ]; then models+=(B2 M2); fi
    checkpoints=()
    for model in "${models[@]}"; do
        printf 'train-%s-seed-%s\n' "$model" "$seed" > "$study/training_stage"
        run="$study/runs/${model}_full_seed$seed"
        if [ ! -s "$run/result.json" ]; then
            resume=()
            if [ -s "$run/last.pt" ]; then resume=(--resume); fi
            TZ=Asia/Jakarta date -Is
            python -u -m cp2.mm_runner --variant "$model" --workspace "$CP2_WORKSPACE" \
                --output "$study/runs" --seed "$seed" --epochs 5 --full --require-gpu "${resume[@]}"
        fi
        checkpoints+=("$run/best.pt")
    done
    printf 'evaluate-seed-%s\n' "$seed" > "$study/training_stage"
    python -u -m cp2.noise_eval --workspace "$CP2_WORKSPACE" --output "$study/benchmark" \
        --require-gpu "${checkpoints[@]}"
    CUDA_VISIBLE_DEVICES='' python -m cp2.reliability_summary --benchmark "$study/benchmark" \
        --output "$study/summary_seed$seed.json" --seeds "$seed"
done

printf 'aggregate-all-seeds\n' > "$study/training_stage"
# The protocol's seed list supplies separate integer arguments.
CUDA_VISIBLE_DEVICES='' python -m cp2.reliability_summary --benchmark "$study/benchmark" \
    --output "$study/summary_all_seeds.json" --seeds $seeds
printf 'completed\n' > "$study/training_stage"
