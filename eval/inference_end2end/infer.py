"""
End-to-end inference script.
Supports local mode (direct vLLM) and server mode (HTTP API).
"""

import argparse
import json
import os
import sys

import requests


def infer_local(video_path: str, model_path: str, tp_size: int | None = None) -> dict:
    """Run inference locally using vLLM."""
    from model_engine import ModelEngine
    from video_processor import extract_frames_from_video, cleanup_frames

    engine = ModelEngine(model_path=model_path, tensor_parallel_size=tp_size)
    frames_dir = None
    try:
        frames_dir, frame_paths, timestamps = extract_frames_from_video(video_path)
        user_prompt = engine.build_user_prompt(frame_paths, timestamps)
        response = engine.run_inference(frame_paths, user_prompt)
        answer = engine.parse_answer(response)
        return {"video_path": video_path, "response": response, "answer": answer}
    finally:
        if frames_dir:
            cleanup_frames(frames_dir)


def infer_server(video_path: str, server_url: str) -> dict:
    """Run inference via HTTP server."""
    url = f"{server_url.rstrip('/')}/v1/analyze"
    resp = requests.post(url, json={"video_path": video_path}, timeout=300)
    resp.raise_for_status()
    return resp.json()


def collect_video_paths(input_path: str) -> list[str]:
    """Collect video paths from a file, directory, or single path."""
    if os.path.isfile(input_path):
        ext = os.path.splitext(input_path)[1].lower()
        if ext == ".txt":
            with open(input_path) as f:
                return [line.strip() for line in f if line.strip()]
        elif ext == ".json":
            with open(input_path) as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
            raise ValueError("JSON file must contain a list of video paths")
        else:
            return [input_path]
    elif os.path.isdir(input_path):
        video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}
        paths = []
        for fname in sorted(os.listdir(input_path)):
            if os.path.splitext(fname)[1].lower() in video_exts:
                paths.append(os.path.join(input_path, fname))
        return paths
    else:
        raise FileNotFoundError(f"Input not found: {input_path}")


def main():
    parser = argparse.ArgumentParser(description="Skyra End-to-End Inference")
    parser.add_argument("--input", type=str, required=True,
                        help="Video path, directory, or txt/json file with paths")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file (default: stdout)")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--server_url", type=str, help="Server URL (e.g. http://localhost:8000)")
    mode.add_argument("--local", action="store_true", help="Run locally with vLLM")

    parser.add_argument("--model_path", type=str, default=None,
                        help="Model path (required for --local)")
    parser.add_argument("--tp_size", type=int, default=None,
                        help="Tensor parallel size (for --local)")
    args = parser.parse_args()

    if args.local and not args.model_path:
        parser.error("--model_path is required when using --local")

    video_paths = collect_video_paths(args.input)
    print(f"Found {len(video_paths)} video(s) to process.")

    results = []
    for i, vp in enumerate(video_paths):
        print(f"[{i+1}/{len(video_paths)}] {vp}")
        try:
            if args.local:
                # For local mode, load engine once for first video, reuse for rest
                if i == 0:
                    from model_engine import ModelEngine
                    from video_processor import extract_frames_from_video, cleanup_frames
                    _engine = ModelEngine(model_path=args.model_path, tensor_parallel_size=args.tp_size)

                frames_dir = None
                try:
                    frames_dir, frame_paths, timestamps = extract_frames_from_video(vp)
                    user_prompt = _engine.build_user_prompt(frame_paths, timestamps)
                    response = _engine.run_inference(frame_paths, user_prompt)
                    answer = _engine.parse_answer(response)
                    result = {"video_path": vp, "response": response, "answer": answer}
                finally:
                    if frames_dir:
                        cleanup_frames(frames_dir)
            else:
                result = infer_server(vp, args.server_url)
            results.append(result)
            print(f"  -> {result['answer']}")
        except Exception as e:
            print(f"  -> Error: {e}", file=sys.stderr)
            results.append({"video_path": vp, "answer": "Error", "error": str(e)})

    output_json = json.dumps(results, ensure_ascii=False, indent=2)
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output_json)
        print(f"\nResults saved to {args.output}")
    else:
        print("\n" + output_json)


if __name__ == "__main__":
    main()
