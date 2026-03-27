# Skyra End-to-End Inference & Evaluation

## Quick Start

### 1. Start the inference server

```bash
# Default: 4 GPUs, port 8000
bash serve.sh

# Custom configuration
CUDA_VISIBLE_DEVICES=0,1 TP_SIZE=2 PORT=8001 MODEL_PATH=/path/to/model bash serve.sh
```

### 2. Single/batch video inference

```bash
# Single video via server
python infer.py --server_url http://localhost:8000 --input /path/to/video.mp4

# Directory of videos
python infer.py --server_url http://localhost:8000 --input /path/to/videos/ --output results.json

# Local mode (no server needed)
python infer.py --local --model_path /path/to/model --input /path/to/video.mp4
```

### 3. ViF-Bench evaluation

```bash
# Via server
python eval_vifbench.py \
    --bench_root /path/to/ViF-Bench \
    --server_url http://localhost:8000 \
    --save_dir results/

# Local mode
python eval_vifbench.py \
    --bench_root /path/to/ViF-Bench \
    --local --model_path /path/to/model \
    --save_dir results/
```

Evaluation supports **checkpoint resuming** — re-run the same command to continue from where it left off.

## File Structure

| File | Description |
|------|-------------|
| `serve.sh` | Launch vLLM server |
| `server.py` | FastAPI server (`/v1/analyze`, `/v1/analyze_batch`, `/v1/analyze_frames`) |
| `model_engine.py` | vLLM model loading & inference |
| `video_processor.py` | Video frame extraction (16 frames, short side 256px) |
| `infer.py` | CLI inference (single/batch, local/server) |
| `eval_vifbench.py` | ViF-Bench evaluation with paired metrics |
