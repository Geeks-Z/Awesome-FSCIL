#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")"
mkdir -p ../logs

nohup env GPU_ID=0 bash ./run_all_with_logs.sh \
    > ../logs/SMP-submit-gpu0.out 2>&1 < /dev/null &

pid=$!
echo "$pid" > ../logs/SMP-submit-gpu0.pid
echo "SMP submitted on GPU 0 (PID: $pid)"
echo "Driver log: ../logs/SMP-submit-gpu0.out"
