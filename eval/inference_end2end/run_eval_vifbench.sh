#!/bin/bash
# Evaluate Skyra-SFT and Skyra-RL on ViF-Bench
# Usage: bash run_eval_vifbench.sh [sft|rl|all]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate vllm

BENCH_ROOT="/path/to/ViF-Bench"
SAVE_DIR="./results"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
TP_SIZE="${TP_SIZE:-4}"

MODE="${1:-all}"

run_eval() {
    local model_path="$1"
    local model_name="$2"

    echo "============================================"
    echo "  Evaluating: $model_name"
    echo "  Model:      $model_path"
    echo "  GPUs:       $CUDA_VISIBLE_DEVICES (TP=$TP_SIZE)"
    echo "============================================"

    python eval_vifbench.py \
        --bench_root "$BENCH_ROOT" \
        --local \
        --model_path "$model_path" \
        --tp_size "$TP_SIZE" \
        --model_name "$model_name" \
        --save_dir "$SAVE_DIR"

    echo ""
    echo "Done: $model_name"
    echo ""
}

case "$MODE" in
    sft)
        run_eval "/path/to/Skyra-SFT" "Skyra-SFT"
        ;;
    rl)
        run_eval "/path/to/Skyra-RL" "Skyra-RL"
        ;;
    all)
        run_eval "/path/to/Skyra-SFT" "Skyra-SFT"
        run_eval "/path/to/Skyra-RL" "Skyra-RL"
        ;;
    *)
        echo "Usage: bash run_eval_vifbench.sh [sft|rl|all]"
        exit 1
        ;;
esac

echo "All evaluations complete. Results in $SAVE_DIR/"
