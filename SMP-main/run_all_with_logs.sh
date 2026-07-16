#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")"

readonly GPU_ID="${GPU_ID:-0}"
readonly SHEETS=(
    "IN1K-NoShuffle1993"
    "IN1K-Shuffle1993"
    "IN1K-Shuffle2025"
    "IN1K-Shuffle42"
)

for sheet in "${SHEETS[@]}"; do
    output_dir="../logs/${sheet}"
    output_file="${output_dir}/SMP-${sheet//1993/-1993}"
    output_file="${output_file//2025/-2025}"
    output_file="${output_file//42/-42}-A800.out"
    mkdir -p "$output_dir"
    echo "[$(date --iso-8601=seconds)] Starting $sheet on physical GPU $GPU_ID"
    CUDA_VISIBLE_DEVICES="$GPU_ID" ./train_smp.sh "$sheet" > "$output_file" 2>&1
done
