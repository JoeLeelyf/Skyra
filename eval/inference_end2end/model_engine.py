"""
vLLM model engine for end-to-end inference.
Loads the model once and provides inference matching the benchmark pipeline.
"""

import os
import re

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import torch
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info


SYSTEM_PROMPT = """\nYou are an expert AI video analyst. Your primary task is to review a sequence of video frames and provide a step-by-step analysis of their authenticity.\n\nYou MUST output your entire analysis using the following structure:\n1.  A `<think>...</think>` block containing your detailed reasoning.\n2.  An `<answer>...</answer>` block containing the final, one-word verdict: 'Fake' or 'Real'.\n\nInside the `<think>` block, you MUST:\n1.  Start by briefly describing the overall content of the video frames.\n2.  Follow a detailed, step-by-step \"discovery\" or \"verification\" process.\n3.  When you identify an artifact (or clear a region), you MUST use a valid L3 Category Name from the \"Artifact Category Definitions\" provided below.\n4.  You MUST embed your finding using the following exact tag structure:\n    <type>L3 Category Name</type> in <t>[startTime, endTime]</t> at <bbox>[x1, y1, x2, y2]</bbox>\n5.  If multiple artifacts are present, you must find and tag all of them in temporal order.\n6.  Your entire reasoning process must be self-contained\n\n---\n## Artifact Category Definitions (Valid L3 Categories for the <type> tag)\n\n### 1. Low-Level Forgery\n* **1.1 Texture Anomaly**:\n    * <type>Structure Anomaly</type>\n    * <type>Texture Jittering</type>\n    * <type>Unnatural Blur</type>\n* **1.2 Color and Lighting Anomaly**:\n    * <type>Color Over-saturation</type>\n    * <type>Lighting Inconsistency</type>\n* **1.3 Move Forgery**:\n    * <type>Camera Motion Inconsistency</type>\n\n### 2. Violation of Laws\n* **2.1 Object Inconsistency**:\n    * <type>Abnormal Object Disappearance</type>\n    * <type>Abnormal Object Appearance</type>\n    * <type>Person Identity Inconsistency</type>\n    * <type>General Object Identity Inconsistency</type>\n    * <type>Shape Distortion</type>\n* **2.2 Interaction Inconsistency**:\n    * <type>Abnormal Rigid-Body Crossing</type>\n    * <type>Abnormal Multi-Object Merging</type>\n    * <type>Abnormal Object Splitting</type>\n    * <type>General Interaction Anomaly</type>\n* **2.3 Unnatural Movement**:\n    * <type>Unnatural Human Movement</type>\n    * <type>Unnatural Animal Movement</type>\n    * <type>Unnatural General Object Movement</type>\n* **2.4 Violation of Causality Law**:\n    * <type>Violation of Physical Law</type>\n    * <type>Violation of General Causality Law</type>\n* **2.5 Violation of Commonsense**:\n    * <type>Abnormal Human Body Structure</type>\n    * <type>Abnormal General Object Structure</type>\n    * <type>Text Distortion</type>\n"""


class ModelEngine:
    """vLLM inference engine matching Qwen2_5_VL_vLLM.py exactly."""

    def __init__(self, model_path: str, tensor_parallel_size: int | None = None):
        self.model_path = model_path
        tp = tensor_parallel_size or torch.cuda.device_count()

        print(f"[ModelEngine] Loading vLLM model from {model_path} with tp={tp}...")
        self.llm = LLM(
            model=model_path,
            limit_mm_per_prompt={"image": 32, "video": 0},
            tensor_parallel_size=tp,
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.sampling_params = SamplingParams(
            temperature=0,
            top_p=1.0,
            max_tokens=8192,
        )
        print(f"[ModelEngine] Model loaded on {tp} GPUs.")

    def build_user_prompt(
        self, frame_paths: list[str], timestamps: list[float]
    ) -> str:
        """Build user prompt matching LADMBench._build_user_prompt exactly."""
        lines = ["Here are the video frames and their corresponding timestamps:"]
        for ts, _ in zip(timestamps, frame_paths):
            lines.append(f"[T={ts:.2f}s] <image>")
        lines.append(
            "\nPlease analyze the video frames, determine if the video is "
            "real or fake, and provide your reasoning."
        )
        return "\n".join(lines)

    def run_inference(self, frame_paths: list[str], user_prompt: str) -> str:
        """Run inference matching Qwen2_5_VL_vLLM.run_inference exactly."""
        user_content = []
        frame_path_iter = iter(frame_paths)
        text_parts = re.split(r"<image>", user_prompt)

        for i, text_part in enumerate(text_parts):
            if text_part.strip():
                user_content.append({"type": "text", "text": text_part})
            if i < len(text_parts) - 1:
                try:
                    user_content.append({
                        "type": "image",
                        "image": next(frame_path_iter),
                        "min_pixels": 224 * 224,
                        "max_pixels": 1280 * 28 * 28,
                    })
                except StopIteration:
                    break

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True
        )
        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs

        llm_inputs = {
            "prompt": prompt,
            "multi_modal_data": mm_data,
            "mm_processor_kwargs": video_kwargs,
        }
        outputs = self.llm.generate([llm_inputs], sampling_params=self.sampling_params)
        return outputs[0].outputs[0].text

    def parse_answer(self, response: str) -> str:
        """Extract answer from response."""
        match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
        if match:
            answer = match.group(1).strip()
            if "fake" in answer.lower():
                return "Fake"
            if "real" in answer.lower():
                return "Real"
            return answer
        return "Error"
