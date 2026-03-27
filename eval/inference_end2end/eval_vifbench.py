"""
ViF-Bench evaluation script.
Runs inference on all videos in ViF-Bench and computes paired metrics.
"""

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import requests
from tqdm import tqdm

try:
    from sklearn.metrics import balanced_accuracy_score, recall_score, f1_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("[Warning] scikit-learn not found. Install with: pip install scikit-learn")


# ===================== Inference =====================

def infer_via_server(frame_paths: list[str], timestamps: list[float], server_url: str) -> str:
    """Send pre-extracted frames to the server for inference."""
    url = f"{server_url.rstrip('/')}/v1/analyze_frames"
    resp = requests.post(url, json={"frame_paths": frame_paths, "timestamps": timestamps}, timeout=300)
    resp.raise_for_status()
    return resp.json()["response"]


def infer_local(frame_paths: list[str], user_prompt: str, engine) -> str:
    """Run inference locally."""
    return engine.run_inference(frame_paths, user_prompt)


# ===================== Data Loading =====================

def load_tasks(index_json: str) -> list[dict]:
    """Load tasks from ViF-Bench test_index.json."""
    parsed_frames_dir = os.path.dirname(os.path.abspath(index_json))

    with open(index_json, "r") as f:
        index_data = json.load(f)

    # Skip full-videos entries
    tasks = []
    for aigc_model, frame_dirs in index_data.items():
        gt = "Real" if aigc_model.lower() == "real" else "Fake"
        for frame_dir_path in frame_dirs:
            if "full-videos" in frame_dir_path:
                continue

            # Extract the video name (last component)
            video_name = Path(frame_dir_path).name

            # Resolve actual path: Real videos under Real/, Fake under Fake/<model>/
            if aigc_model.lower() == "real":
                actual_path = os.path.join(parsed_frames_dir, "Real", video_name)
                video_id = f"real/{video_name}"
            else:
                actual_path = os.path.join(parsed_frames_dir, "Fake", aigc_model, video_name)
                video_id = f"{aigc_model}/{video_name}"

            if not os.path.isdir(actual_path):
                continue

            tasks.append({
                "frame_dir_path": actual_path,
                "video_id": video_id,
                "aigc_model_name": aigc_model,
                "gt": gt,
            })
    return tasks


def build_user_prompt(frame_dir_path: str):
    """Build user prompt from pre-extracted frames directory."""
    timestamps_file = os.path.join(frame_dir_path, "timestamps.txt")
    if not os.path.exists(timestamps_file):
        return None, None, None

    with open(timestamps_file, "r") as f:
        timestamps = [float(line.strip()) for line in f]

    prompt_lines = ["Here are the video frames and their corresponding timestamps:"]
    frame_paths = []

    for i, ts in enumerate(timestamps):
        frame_path = os.path.join(frame_dir_path, f"{i + 1}.png")
        if not os.path.exists(frame_path):
            return None, None, None
        prompt_lines.append(f"[T={ts:.2f}s] <image>")
        frame_paths.append(os.path.abspath(frame_path))

    prompt_lines.append(
        "\nPlease analyze the video frames, determine if the video is "
        "real or fake, and provide your reasoning."
    )
    return "\n".join(prompt_lines), frame_paths, timestamps


def parse_answer(response: str) -> str:
    """Extract answer from model response."""
    match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if match:
        answer = match.group(1).strip()
        if "fake" in answer.lower():
            return "Fake"
        if "real" in answer.lower():
            return "Real"
        return answer
    return "Error"


# ===================== Metrics =====================

DESIRED_MODEL_ORDER = [
    "Wan2.1-T2V-1.3B", "CogVideoX1.5-5B-T", "Wan2.2-TI2V-5B-T",
    "Wan2.2-TI2V-5B-I", "HunyuanVideo", "HunyuanVideo-I2V",
    "Wan2.1-VACE-1.3B-T", "Wan2.2-T2V-14B", "Wan2.2-I2V-14B",
    "SkyReels-V2", "SkyReels-V2-I2V-14B-540P", "LTX-Video-13B-T",
    "LTX-Video-13B-I", "gen4-turbo", "hailuo", "pika-v2",
    "pixverse-v4-5", "kling-v1", "sora-2",
]


def compute_paired_metrics(results: list[dict]) -> dict:
    """Compute paired Balanced ACC, Recall, F1 per AIGC model."""
    if not SKLEARN_AVAILABLE:
        print("[Error] scikit-learn is required.")
        return {}

    # Group by base_id
    video_pairs = defaultdict(dict)
    for item in results:
        gt = item.get("gt")
        answer = item.get("answer")
        video_id = item.get("video_id")
        aigc_model = item.get("aigc_model_name")
        if not all([gt, answer, video_id, aigc_model]):
            continue

        parts = video_id.split("/", 1)
        if len(parts) != 2:
            continue
        base_id = parts[1]

        if aigc_model.lower() == "real":
            video_pairs[base_id]["real_pred"] = answer.lower()
        else:
            video_pairs[base_id][aigc_model] = {"pred": answer.lower()}

    # Build paired results per model
    paired_results = defaultdict(list)
    for base_id, preds in video_pairs.items():
        if "real_pred" not in preds:
            continue
        real_pred = preds["real_pred"]
        for model_name, fake_data in preds.items():
            if model_name == "real_pred":
                continue
            paired_results[model_name].append({
                "real_pred": real_pred,
                "fake_pred": fake_data["pred"],
            })

    # Calculate metrics
    all_metrics = {}
    numeric_for_avg = defaultdict(list)

    # Order models
    present = set(paired_results.keys())
    ordered = [m for m in DESIRED_MODEL_ORDER if m in present]
    ordered += sorted(present - set(ordered))

    for model_name in ordered:
        pairs = paired_results[model_name]
        if not pairs:
            all_metrics[model_name] = {"num_pairs": 0, "acc": "N/A", "recall": "N/A", "f1": "N/A"}
            continue

        y_true, y_pred = [], []
        for p in pairs:
            y_true.extend(["real", "fake"])
            y_pred.extend([p["real_pred"], p["fake_pred"]])

        encode = lambda x: 0 if x == "real" else 1
        yt = np.array([encode(x) for x in y_true])
        yp = np.array([encode(x) for x in y_pred])

        acc = balanced_accuracy_score(yt, yp)
        recall = recall_score(yt, yp, pos_label=1, zero_division=0)
        f1 = f1_score(yt, yp, pos_label=1, zero_division=0)

        numeric_for_avg["acc"].append(acc)
        numeric_for_avg["recall"].append(recall)
        numeric_for_avg["f1"].append(f1)

        all_metrics[model_name] = {
            "num_pairs": len(pairs),
            "acc": f"{acc * 100:.2f}",
            "recall": f"{recall * 100:.2f}",
            "f1": f"{f1 * 100:.2f}",
        }

        print(f"  {model_name:<30} | {len(pairs)} pairs | ACC={acc*100:.2f}% Recall={recall*100:.2f}% F1={f1*100:.2f}%")

    # Average
    avg = {}
    if numeric_for_avg:
        avg = {
            "acc": f"{np.mean(numeric_for_avg['acc']) * 100:.2f}",
            "recall": f"{np.mean(numeric_for_avg['recall']) * 100:.2f}",
            "f1": f"{np.mean(numeric_for_avg['f1']) * 100:.2f}",
        }
        print(f"\n  {'Average':<30} | ACC={avg['acc']}% Recall={avg['recall']}% F1={avg['f1']}%")

    return {"per_model": all_metrics, "ordered_models": ordered, "average": avg}


def save_csv(metrics: dict, csv_path: str):
    """Save metrics as transposed CSV."""
    ordered = metrics["ordered_models"]
    per_model = metrics["per_model"]
    avg = metrics["average"]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Metric"] + ordered + ["Average"])
        w.writerow(["Num_Pairs"] + [per_model[m]["num_pairs"] for m in ordered] + ["N/A"])
        w.writerow(["Avg_ACC"] + [per_model[m]["acc"] for m in ordered] + [avg.get("acc", "N/A")])
        w.writerow(["Recall"] + [per_model[m]["recall"] for m in ordered] + [avg.get("recall", "N/A")])
        w.writerow(["F1"] + [per_model[m]["f1"] for m in ordered] + [avg.get("f1", "N/A")])


# ===================== Main =====================

def main():
    parser = argparse.ArgumentParser(description="ViF-Bench Evaluation")
    parser.add_argument("--bench_root", type=str, required=True,
                        help="Path to ViF-Bench (contains parsed_frames/test_index.json)")
    parser.add_argument("--save_dir", type=str, default="./results",
                        help="Directory to save results")
    parser.add_argument("--model_name", type=str, default="Skyra",
                        help="Model name for output files")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--server_url", type=str, help="Server URL")
    mode.add_argument("--local", action="store_true", help="Run locally")

    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--tp_size", type=int, default=None)
    args = parser.parse_args()

    if args.local and not args.model_path:
        parser.error("--model_path is required when using --local")

    os.makedirs(args.save_dir, exist_ok=True)
    output_json = os.path.join(args.save_dir, f"{args.model_name}.json")

    # Load index
    index_json = os.path.join(args.bench_root, "parsed_frames", "test_index.json")
    if not os.path.exists(index_json):
        print(f"[Error] Index not found: {index_json}")
        sys.exit(1)

    tasks = load_tasks(index_json)
    print(f"Total tasks: {len(tasks)}")

    # Resume support
    results = []
    processed_ids = set()
    if os.path.exists(output_json):
        with open(output_json, "r", encoding="utf-8") as f:
            results = json.load(f)
        processed_ids = {r["video_id"] for r in results}
        print(f"Resuming: {len(processed_ids)} already processed.")

    remaining = [t for t in tasks if t["video_id"] not in processed_ids]
    print(f"Remaining: {len(remaining)}")

    if not remaining:
        print("All tasks already processed.")
    else:
        # Load local engine if needed
        engine = None
        if args.local:
            from model_engine import ModelEngine
            engine = ModelEngine(model_path=args.model_path, tensor_parallel_size=args.tp_size)

        for i, task in enumerate(tqdm(remaining, desc="ViF-Bench Eval")):
            user_prompt, frame_paths, timestamps = build_user_prompt(task["frame_dir_path"])
            if not user_prompt:
                print(f"  [Skip] {task['video_id']}: missing frames/timestamps")
                continue

            try:
                if args.local:
                    response = infer_local(frame_paths, user_prompt, engine)
                else:
                    response = infer_via_server(frame_paths, timestamps, args.server_url)
            except Exception as e:
                print(f"  [Error] {task['video_id']}: {e}")
                response = ""

            answer = parse_answer(response)
            results.append({
                "video_id": task["video_id"],
                "aigc_model_name": task["aigc_model_name"],
                "mllm_model_name": args.model_name,
                "gt": task["gt"],
                "response": response,
                "answer": answer,
            })

            # Periodic save
            if (i + 1) % 10 == 0:
                with open(output_json, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)

        # Final save
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Results saved to {output_json}")

    # Compute metrics
    print("\n=== Paired Metrics ===")
    metrics = compute_paired_metrics(results)
    if metrics:
        csv_path = os.path.join(args.save_dir, f"{args.model_name}_metrics.csv")
        save_csv(metrics, csv_path)
        print(f"Metrics CSV saved to {csv_path}")

        metrics_json_path = os.path.join(args.save_dir, f"{args.model_name}_metrics.json")
        with open(metrics_json_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"Metrics JSON saved to {metrics_json_path}")


if __name__ == "__main__":
    main()
