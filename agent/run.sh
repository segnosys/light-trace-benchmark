#!/bin/bash
# Code-agent QPS sweep entry point.
#
#   ./run.sh [server_url] [model_name]
#
# Drives runner.py -> agent_throughput.py against the generic code-agent profile.

set -euo pipefail

SERVER="${1:-http://localhost:8000}"
MODEL="${2:-dsv3}"
STAMP="$(date +%Y%m%d_%H%M%S)"
NAME="code-agent-${STAMP}"

cd "$(dirname "$0")"

printf '%s\n' "------------------------------------------"
printf ' code-agent QPS sweep\n'
printf '   server : %s\n' "$SERVER"
printf '   model  : %s\n' "$MODEL"
printf '   run    : %s\n' "$NAME"
printf '%s\n' "------------------------------------------"

python3 runner.py \
  --name              "$NAME" \
  --workload-config   workloads/code_agent_16k.yaml \
  --start-qps         0.2 \
  --end-qps           1.0 \
  --step              0.2 \
  --sustain-duration  300 \
  --max-inflight      16 \
  --server            "$SERVER" \
  --model             "$MODEL" \
  --results-dir       qps_sweep_results_code_agent

printf 'finished %s\n' "$(date)"
