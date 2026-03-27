"""
Video preprocessing aligned with LADMBench pipeline.
Extracts 16 uniformly-sampled frames, resizes short side to 256px, saves as PNG.
"""

import os
import tempfile
import shutil

import cv2


NUM_FRAMES = 16
SHORT_SIDE = 256


def extract_frames_from_video(
    video_path: str,
    num_frames: int = NUM_FRAMES,
    short_side: int = SHORT_SIDE,
    output_dir: str | None = None,
) -> tuple[str, list[str], list[float]]:
    """
    Extract frames from a video file, matching LADMBench preprocessing.

    Returns:
        (frames_dir, frame_paths, timestamps)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0 or fps <= 0:
        cap.release()
        raise ValueError(f"Invalid video: frames={total_frames}, fps={fps}")

    if total_frames <= num_frames:
        indices = list(range(total_frames))
    else:
        indices = [
            int(round(i * (total_frames - 1) / (num_frames - 1)))
            for i in range(num_frames)
        ]

    timestamps = [idx / fps for idx in indices]

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="skyra_frames_")
    else:
        os.makedirs(output_dir, exist_ok=True)

    frame_paths = []
    for frame_num, idx in enumerate(indices, start=1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            cap.release()
            raise ValueError(f"Failed to read frame at index {idx}")

        h, w = frame.shape[:2]
        if h < w:
            new_h = short_side
            new_w = int(w * (short_side / h))
        else:
            new_w = short_side
            new_h = int(h * (short_side / w))
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        frame_path = os.path.join(output_dir, f"{frame_num}.png")
        cv2.imwrite(frame_path, frame)
        frame_paths.append(os.path.abspath(frame_path))

    cap.release()

    ts_path = os.path.join(output_dir, "timestamps.txt")
    with open(ts_path, "w") as f:
        for ts in timestamps:
            f.write(f"{ts:.2f}\n")

    return output_dir, frame_paths, timestamps


def cleanup_frames(frames_dir: str):
    """Remove temporary frame directory."""
    if frames_dir and os.path.isdir(frames_dir):
        shutil.rmtree(frames_dir, ignore_errors=True)
