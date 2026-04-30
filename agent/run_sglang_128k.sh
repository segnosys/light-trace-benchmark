#!/bin/bash
# Launch sglang serving the NVFP4 Qwen3-30B MoE with 128K context.
#
#   ./run_sglang_128k.sh [gpu_ids] [host_port]
#
# gpu_ids: comma-separated CUDA device list; the count sets tensor parallel
#          size.  Example: "3,4" -> tp=2 across GPU 3 and GPU 4.
# Defaults: "3,4" (tp=2), host port 8001 (container port 30000).
# Serves on http://<host>:8001/v1/*  with model id qwen3-30b-a3b-nvfp4.

set -euo pipefail

GPU_IDS="${1:-3,4}"
HOST_PORT="${2:-8001}"
NAME="sglang-qwen3-nvfp4-128k"
TP_SIZE=$(awk -F, '{print NF}' <<<"$GPU_IDS")

MODEL_DIR="/scratch/huggingface/hub/models--nvidia--Qwen3-30B-A3B-NVFP4/snapshots/2538ded2a4edb247b4d2b4a8ba24e44bd4c017c3"

# If a previous instance is hanging around, stop it first.
docker rm -f "$NAME" >/dev/null 2>&1 || true

echo "=========================================="
echo " sglang (Qwen3-30B-A3B NVFP4 @ 128K)"
echo "   GPUs      : $GPU_IDS  (tp=$TP_SIZE)"
echo "   host port : $HOST_PORT -> container 30000"
echo "   model id  : qwen3-30b-a3b-nvfp4"
echo "=========================================="

docker run -d --name "$NAME" \
  --gpus "\"device=$GPU_IDS\"" \
  --shm-size 32g --ipc host \
  -p "$HOST_PORT:30000" \
  -v /scratch/huggingface:/hf \
  -e HF_HOME=/hf \
  -e HF_HUB_OFFLINE=1 \
  -e SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
  lmsysorg/sglang:latest \
  python3 -m sglang.launch_server \
    --model-path "/hf/hub/models--nvidia--Qwen3-30B-A3B-NVFP4/snapshots/2538ded2a4edb247b4d2b4a8ba24e44bd4c017c3" \
    --host 0.0.0.0 --port 30000 \
    --tp "$TP_SIZE" \
    --context-length 131072 \
    --served-model-name qwen3-30b-a3b-nvfp4 \
    --mem-fraction-static 0.9 \
    --max-running-requests 16 \
    --enable-cache-report

cat <<EOF

Container started: $NAME
To follow startup:     docker logs -f $NAME
Ready signal:          "The server is fired up and ready to roll!"
Health check:          curl http://localhost:$HOST_PORT/v1/models
Stop the server:       docker stop $NAME
EOF
