#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")"

readonly SHEETS=(
    "IN1K-NoShuffle1993"
    "IN1K-Shuffle1993"
    "IN1K-Shuffle2025"
    "IN1K-Shuffle42"
)
readonly CONFIGS=(
    "smp_cub_B100_Inc10.json"
    "smp_cifar_B60_Inc5.json"
    "smp_inr_B100_Inc10.json"
    "smp_mini_B60_Inc5.json"
)

run_sheet() {
    local sheet="$1"
    local config

    for config in "${CONFIGS[@]}"; do
        echo "[$(date --iso-8601=seconds)] ${sheet}/${config}"
        python -u main.py --config="./configs/${sheet}/${config}"
    done
}

selection="${1:-all}"
if [[ "$selection" == "all" ]]; then
    for sheet in "${SHEETS[@]}"; do
        run_sheet "$sheet"
    done
else
    valid=false
    for sheet in "${SHEETS[@]}"; do
        if [[ "$selection" == "$sheet" ]]; then
            valid=true
            run_sheet "$sheet"
            break
        fi
    done
    if [[ "$valid" == false ]]; then
        echo "Unknown sheet: $selection" >&2
        echo "Expected one of: all ${SHEETS[*]}" >&2
        exit 2
    fi
fi
