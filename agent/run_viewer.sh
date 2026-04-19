#!/bin/bash
# Launch the viewer (Dash/Plotly) against the code-agent sweep results.
#
#   ./run_viewer.sh [data_dir] [port]
#
# data_dir defaults to the directory written by run.sh.

set -euo pipefail

DATA_DIR="${1:-./qps_sweep_results_code_agent}"
PORT="${2:-8050}"

cd "$(dirname "$0")"

if [ ! -d "$DATA_DIR" ]; then
    printf 'data dir %s not found — creating empty dir\n' "$DATA_DIR"
    mkdir -p "$DATA_DIR"
fi

printf 'viewer  data=%s  port=%s\n' "$DATA_DIR" "$PORT"

exec python3 viewer.py --data-dir "$DATA_DIR" --port "$PORT"
