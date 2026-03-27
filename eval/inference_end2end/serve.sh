#!/bin/bash
# Launch Skyra inference server with vLLM
# Usage: bash serve.sh
# Environment variables:
#   MODEL_PATH  - path to model (default: /path/to/Skyra-SFT)
#   PORT        - server port (default: 8000)
#   TP_SIZE     - tensor parallel size (default: 4)
#   CUDA_VISIBLE_DEVICES - GPUs to use (default: 0,1,2,3)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate vllm

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

MODEL_PATH="${MODEL_PATH:-/path/to/Skyra-SFT}"
PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-4}"

echo "Starting Skyra Inference Server..."
echo "  Model: $MODEL_PATH"
echo "  Port:  $PORT"
echo "  TP:    $TP_SIZE"
echo "  GPUs:  $CUDA_VISIBLE_DEVICES"

python server.py \
    --model_path "$MODEL_PATH" \
    --port "$PORT" \
    --tensor_parallel_size "$TP_SIZE" \
    "$@"
