#!/bin/bash
# End-to-end test: drive the 128K-context sglang server with the code-agent
# workload profile.  Runs a single QPS point by default so you can see a
# clean ramp / sustain / drain breakdown; edit START_QPS / END_QPS to sweep.
#
#   ./run_128k.sh [server_url] [model_name]
#
# Pre-requisite:
#   ./run_sglang_128k.sh &   # start the server in another terminal
#   # wait for "The server is fired up" in its logs

set -euo pipefail

SERVER="${1:-http://localhost:8001}"
MODEL="${2:-qwen3-30b-a3b-nvfp4}"
NAME="code-agent-128k-$(date +%Y%m%d_%H%M%S)"

cd "$(dirname "$0")"

echo "=========================================="
echo " code-agent @ 128K  vs  $SERVER"
echo "   model : $MODEL"
echo "   name  : $NAME"
echo "=========================================="

HF_HUB_CACHE=/scratch/huggingface \
HF_HOME=/scratch/huggingface \
TRANSFORMERS_OFFLINE=1 \
python3 runner.py \
  --name              "$NAME" \
  --workload-config   workloads/code_agent_128k.yaml \
  --start-qps         0.2 \
  --end-qps           0.2 \
  --step              0.1 \
  --sustain-duration  300 \
  --max-inflight      8 \
  --server            "$SERVER" \
  --model             "$MODEL" \
  --results-dir       qps_sweep_results_code_agent

cat <<EOF

Done.

==== Where is TPM? ====
The summary section 'Phase Throughput Breakdown' (see the sim output above)
is the source of truth for tokens-per-minute numbers:

  - input TPM     total prompt tokens sent per minute (INCLUDES cache hits)
  - cached TPM    portion served from prefix cache (no compute)
  - uncached TPM  = input - cached = actual prefill work the server did
  - gen TPM       completion tokens per minute

The three rows are:
  ramp     only the linear ramp-up phase
  sustain  steady state — this is the number you quote for capacity
  drain    requests sent late in sustain that finished afterward

For a single capacity number, read the 'sustain' row.
EOF
