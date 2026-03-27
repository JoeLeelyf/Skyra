"""
Minimal FastAPI inference server for end-to-end evaluation.
Endpoints: POST /v1/analyze, POST /v1/analyze_batch, POST /v1/analyze_frames
"""

import argparse
import os

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import uvicorn

from video_processor import extract_frames_from_video, cleanup_frames
from model_engine import ModelEngine

app = FastAPI(title="Skyra Inference Server")

engine: ModelEngine | None = None


def _run_analysis(video_path: str, num_frames: int = 16) -> dict:
    """Core analysis: preprocess video -> build prompt -> run inference."""
    frames_dir = None
    try:
        frames_dir, frame_paths, timestamps = extract_frames_from_video(
            video_path, num_frames=num_frames
        )
        user_prompt = engine.build_user_prompt(frame_paths, timestamps)
        response = engine.run_inference(frame_paths, user_prompt)
        answer = engine.parse_answer(response)
        return {
            "video_path": video_path,
            "response": response,
            "answer": answer,
        }
    finally:
        if frames_dir:
            cleanup_frames(frames_dir)


@app.post("/v1/analyze")
async def analyze(request: Request):
    """
    Analyze a single video. JSON body: {"video_path": "...", "num_frames": 16}
    """
    body = await request.json()
    video_path = body.get("video_path")
    num_frames = body.get("num_frames", 16)

    if not video_path or not os.path.isfile(video_path):
        raise HTTPException(status_code=400, detail=f"Invalid video_path: {video_path}")

    result = _run_analysis(video_path, num_frames=num_frames)
    return JSONResponse(result)


@app.post("/v1/analyze_batch")
async def analyze_batch(request: Request):
    """
    Analyze multiple videos. JSON body: {"video_paths": [...], "num_frames": 16}
    """
    body = await request.json()
    video_paths = body.get("video_paths", [])
    num_frames = body.get("num_frames", 16)

    if not video_paths:
        raise HTTPException(status_code=400, detail="No video_paths provided")

    results = []
    for vp in video_paths:
        if not os.path.isfile(vp):
            results.append({"video_path": vp, "response": "", "answer": "Error", "error": f"File not found: {vp}"})
            continue
        try:
            results.append(_run_analysis(vp, num_frames=num_frames))
        except Exception as e:
            results.append({"video_path": vp, "response": "", "answer": "Error", "error": str(e)})

    return JSONResponse({"results": results})


@app.post("/v1/analyze_frames")
async def analyze_frames(request: Request):
    """
    Analyze pre-extracted frames. JSON body: {"frame_paths": [...], "timestamps": [...]}
    """
    body = await request.json()
    frame_paths = body.get("frame_paths", [])
    timestamps = body.get("timestamps", [])

    if not frame_paths:
        raise HTTPException(status_code=400, detail="No frame_paths provided")

    user_prompt = engine.build_user_prompt(frame_paths, timestamps)
    response = engine.run_inference(frame_paths, user_prompt)
    answer = engine.parse_answer(response)

    return JSONResponse({"response": response, "answer": answer})


def main():
    parser = argparse.ArgumentParser(description="Skyra Inference Server")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--tensor_parallel_size", type=int, default=None)
    args = parser.parse_args()

    global engine
    engine = ModelEngine(
        model_path=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
    )

    print(f"\n[Skyra Server] Running at http://{args.host}:{args.port}")
    print(f"  POST /v1/analyze        - single video")
    print(f"  POST /v1/analyze_batch  - batch videos")
    print(f"  POST /v1/analyze_frames - pre-extracted frames")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
