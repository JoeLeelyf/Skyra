
<h1 align="center">
  <font color=#0088cc>Skyra</font>: AI-Generated Video Detection via Grounded Artifact Reasoning
</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2512.15693" style="margin-right: 10px;">
    <img src="https://img.shields.io/badge/arXiv-2512.15693-b31b1b.svg?logo=arXiv">
  </a>
  <a href="https://huggingface.co/collections/JoeLeelyf/skyra" style="margin-right: 10px;">
    <img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Skyra-ffd21e">
  </a>
  <a href="https://joeleelyf.github.io/Skyra/" style="margin-right: 10px;">
    <img src="https://img.shields.io/badge/Project-Skyra-black?logo=github">
  </a>
</p>

> **Abstract:** The misuse of AI-driven video generation technologies has raised serious social concerns, highlighting the urgent need for reliable AI-generated video detectors. However, most existing methods are limited to binary classification and lack the necessary explanations for human interpretation. In this paper, we present **Skyra**, a specialized multimodal large language model (MLLM) that identifies human-perceivable visual artifacts in AI-generated videos and leverages them as grounded evidence for both detection and explanation. To support this objective, we construct **ViF-CoT-4K** for Supervised Fine-Tuning (SFT), which represents the first large-scale AI-generated video artifact dataset with fine-grained human annotations. We then develop a two-stage training strategy that systematically enhances our model's spatio-temporal artifact perception, explanation capability, and detection accuracy. Extensive experiments demonstrate that Skyra surpasses existing methods across multiple benchmarks, providing valuable insights for advancing explainable AI-generated video detection.

## 🌟 Introduction

### 🎯 Core Capabilities
Unlike traditional binary detectors or general MLLMs, Skyra focuses on **Grounded Artifact Reasoning**:
- **Artifact Perception**: Identifies subtle visual anomalies (e.g., *Physics Violation*, *Texture Jittering*).
- **Spatio-Temporal Grounding**: Pinpoints exact timestamps and bounding boxes where artifacts occur.
- **Explanatory Reasoning**: Provides detailed Chain-of-Thought (CoT) explanations for why a video is Real or Fake.

### 🧩 Hierarchical Artifact Taxonomy
We define a comprehensive taxonomy to categorize AI generation errors, dividing them into **Low-level Forgery** (e.g., texture/color anomalies) and **Violation of Laws** (e.g., physical inconsistencies).

<p align="center">
  <img src="static/images/taxonomy.png" alt="Taxonomy of Artifacts" width="60%">
</p>

## 📊 Dataset: ViF-CoT-4K

**ViF-CoT-4K** is constructed to address the lack of detailed artifact annotations in existing datasets.

- **Scale**: ~4,000 videos, including high-quality samples from **Sora-2, Wan2.1, Kling**, and more.
- **Annotation**: Fine-grained labels including artifact type, textual explanation, timestamps, and bounding boxes.
- **Real-Fake Pairs**: Generated videos are semantically aligned with real counterparts to prevent shortcut learning.
<p align="center">
  <img src="static/images/statistics.png" alt="Dataset Statistics" width="90%">
</p>

## 🚀 Methodology

Skyra employs a **Two-Stage Training Strategy** to achieve interpretable detection:

1.  **Cold-Start Initialization (SFT)**: Fine-tuning Qwen2.5-VL on ViF-CoT-4K to endow the model with basic detection and explanation capabilities.
2.  **Reinforcement Learning (RL)**: Utilizing Group Relative Policy Optimization (GRPO) with an **Asymmetric Reward** design. This encourages the model to actively explore artifacts while strictly supervising classification accuracy.

## 📈 Experimental Results

Skyra achieves state-of-the-art performance, significantly outperforming binary detectors (e.g., DeMamba, NSG-VD) and general MLLMs (e.g., GPT-4o, Gemini).

<p align="center">
<img src="static/images/performance.png" alt="Radar Chart Performance" width="45%">
</p>

**ViF-Bench**: Skyra achieves **91.02% Accuracy**, surpassing the second-best method by a large margin.

## 🛠️ Usage

### Requirements
- **SFT Stage**: follow [LlaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for environment setup.
- **RL Stage**: follow [verl](https://github.com/volcengine/verl) for environment setup.
- **Inference**: follow [Qwen-2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) for quick start and [vLLM](https://github.com/vllm-project/vllm) for deployment.

### Data Preparation
- Training data: Download and prepare the **ViF-CoT-4K** dataset from [here](https://huggingface.co/datasets/JoeLeelyf/ViF-CoT-4K).

- Evaluation data: Download evaluation datasets (e.g., **ViF-Bench**) from [here](https://huggingface.co/datasets/JoeLeelyf/ViF-Bench). And modify the path to your local directory in `test_index.json`.
The `test_index.json` file should contain the following format:
```json
{
    "Real": [
        "path_to_parsed_frames_dir/Real/gdymHI9S6gM-0",
        ...
    ],
    "LTX-Video-13B-T": [
        "path_to_parsed_frames_dir/Fake/LTX-Video-13B-T/gdymHI9S6gM-0",
        ...
    ],
    ...
```

### Supervised Fine-Tuning (SFT)
We use LLaMA-Factory for SFT. You can start training after setup the dataset config following the instructions in the LLaMA-Factory repository.

```bash
cd train/LLaMA-Factory
bash train.sh
```

### Reinforcement Learning (RL)
We use verl for RL training with GRPO, with adapted reward design provided in `train/verl/verl/utils/reward_score/ladm.py`.

### Evaluation

Evaluate scripts are provided in the `eval/` directory. You can run the evaluation script as follows:

- inference: Run inference to get model predictions and explanations, save the results in a JSON file.
```bash
cd eval
bash scripts/Skyra/inference.sh
# or
python inference.py \
    --index_json /path_to/test_index.json \
    --model_path /path_to/Skyra-SFT \
    --model_name Skyra-SFT \
    --save_dir results/Skyra
```

- evaluation: Evaluate the model predictions against ground truth and compute metrics.
```bash
cd eval
bash scripts/Skyra/eval.sh
# or
python eval.py \
    --json_file_path results/Skyra/Skyra-SFT_predictions.json
```

### End-to-End Inference

We provide a self-contained end-to-end inference pipeline in `eval/inference_end2end/`, which integrates video preprocessing, vLLM model loading, and supports both single-video inference and ViF-Bench batch evaluation.

#### File Structure

| File | Description |
|------|-------------|
| `serve.sh` | Launch the vLLM inference server |
| `server.py` | FastAPI server with endpoints: `/v1/analyze`, `/v1/analyze_batch`, `/v1/analyze_frames` |
| `model_engine.py` | vLLM model loading, prompt construction, and inference |
| `video_processor.py` | Video frame extraction (16 uniformly-sampled frames, short side resized to 256px) |
| `infer.py` | CLI inference tool (single video / batch, local / server mode) |
| `eval_vifbench.py` | ViF-Bench evaluation with paired metrics (Balanced ACC, Recall, F1) |
| `run_eval_vifbench.sh` | One-click script to evaluate Skyra-SFT and Skyra-RL on ViF-Bench |

#### 1. Start the Inference Server

```bash
cd eval/inference_end2end

# Default: 4 GPUs, port 8000
MODEL_PATH=/path/to/Skyra-SFT bash serve.sh

# Custom configuration
CUDA_VISIBLE_DEVICES=0,1 TP_SIZE=2 PORT=8001 MODEL_PATH=/path/to/Skyra-SFT bash serve.sh
```

#### 2. Single / Batch Video Inference

```bash
# Single video via server
python infer.py --server_url http://localhost:8000 --input /path/to/video.mp4

# Directory of videos
python infer.py --server_url http://localhost:8000 --input /path/to/videos/ --output results.json

# Local mode (no server needed, loads vLLM directly)
python infer.py --local --model_path /path/to/Skyra-SFT --input /path/to/video.mp4
```

#### 3. ViF-Bench Evaluation

```bash
# Quick start: evaluate both Skyra-SFT and Skyra-RL
bash run_eval_vifbench.sh all    # or "sft" / "rl" for individual models

# Via server
python eval_vifbench.py \
    --bench_root /path/to/ViF-Bench \
    --server_url http://localhost:8000 \
    --save_dir results/

# Local mode
python eval_vifbench.py \
    --bench_root /path/to/ViF-Bench \
    --local --model_path /path/to/Skyra-SFT \
    --save_dir results/
```

The evaluation script supports **checkpoint resuming** — re-run the same command to continue from where it left off. Results include per-model paired metrics (Balanced ACC, Recall, F1) saved in both JSON and CSV formats.

## ⚖️ License

The **ViF-CoT-4K** dataset and **Skyra** model weights are released under the **CC BY 4.0** license. Users must adhere to the terms of source datasets (Kinetics-400, Panda-70M, HD-VILA-100M).

## 📍 Citation

If you find Skyra or ViF-CoT-4K useful, please cite our paper:

```bibtex
@article{li2025skyra,
  title={Skyra: AI-Generated Video Detection via Grounded Artifact Reasoning},
  author={Li, Yifei and Zheng, Wenzhao and Zhang, Yanran and Sun, Runze and Zheng, Yu and Chen, Lei and Zhou, Jie and Lu, Jiwen},
  journal={arXiv preprint arXiv:2512.15693},
  year={2025}
}
```
