import torch
import re
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
# Ensure you have installed qwen_vl_utils, which is required for vLLM + Qwen-VL
# (pip install qwen-vl-utils)
from qwen_vl_utils import process_vision_info
from utils.ViFBench import ViFBench  # Import the base class from your framework

class vLLMModel(ViFBench):
    """
    vLLM (Qwen-VL) implementation for the ViFBench framework.

    It inherits from ViFBench and implements model loading (load_model) and inference (run_inference).
    It preserves the interleaved text-image input format rather than using a single 'video' blob.
    """

    def load_model(self):
        """
        (vLLM requirement) Load the vLLM engine, Processor, and SamplingParams.
        """

        print("start init")
        # 1. Load the vLLM engine
        # Allow many image inputs (e.g., 32) and disable video (0)
        self.llm = LLM(
            model=self.model_path,
            limit_mm_per_prompt={"image": 32, "video": 0},
            # Automatically use all available GPUs
            tensor_parallel_size=torch.cuda.device_count()
        )
        print("end init")

        # 2. Load the Processor
        self.processor = AutoProcessor.from_pretrained(self.model_path)

        # 3. Define SamplingParams
        # Use temperature=0 (greedy decoding) to match do_sample=False in your Qwen2_5_VL.py
        # Increase max_tokens to 1024 (or higher) to ensure there is enough room for the <think> block
        self.sampling_params = SamplingParams(
            temperature=0,
            top_p=1.0,
            max_tokens=4096*2
        )
        print(f"vLLM model {self.model_path} loaded with {torch.cuda.device_count()} GPUs.")


    def run_inference(self, frame_paths: list, user_prompt: str) -> str:
        """
        (vLLM requirement) Run vLLM inference.

        This method converts the user_prompt (with <image> tags) and frame_paths provided by ViFBench
        into the 'messages' format required by vLLM, then calls llm.generate.
        """

        # 1. (Reuse the logic from Qwen2_5_VL.py) Reconstruct the user_content list
        user_content = []
        frame_path_iter = iter(frame_paths)

        # Split text by <image> placeholders
        text_parts = re.split(r'<image>', user_prompt)

        for i, text_part in enumerate(text_parts):
            # Add the text part
            if text_part.strip():
                user_content.append({
                    "type": "text",
                    "text": text_part
                })

            # Add an image after the text part (except the last part)
            if i < len(text_parts) - 1:
                try:
                    # Following your vLLM example, pass the file path
                    # and add the pixel constraints used in the vLLM example
                    user_content.append({
                        "type": "image",
                        "image": next(frame_path_iter),
                        "min_pixels": 224 * 224,
                        "max_pixels": 1280 * 28 * 28,
                    })
                except StopIteration:
                    print(f"  [WARNING] More <image> tags in the prompt than images in the frame_paths list.")
                    break


        # 2. (vLLM format) Build messages
        messages = [
            {
                "role": "system",
                "content": self.SYSTEM_PROMPT  # Use the System Prompt defined in the base class
            },
            {
                "role": "user",
                "content": user_content,
            }
        ]

        # 3. (vLLM format) Apply Chat Template
        try:
            prompt = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            # 4. (vLLM format) Process vision information
            # We only care about image_inputs because we disabled video
            image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

            mm_data = {}
            if image_inputs is not None:
                mm_data["image"] = image_inputs

            # 5. (vLLM format) Build llm_inputs
            llm_inputs = {
                "prompt": prompt,
                "multi_modal_data": mm_data,
                "mm_processor_kwargs": video_kwargs,  # Pass as in the vLLM example
            }

            # 6. (vLLM format) Run inference
            outputs = self.llm.generate([llm_inputs], sampling_params=self.sampling_params)
            generated_text = outputs[0].outputs[0].text

            return generated_text

        except Exception as e:
            print(f"  [CRITICAL ERROR] vLLM inference failed: {e}")
            print(f"  Failed Prompt (text only): {prompt[:200]}...")
            return "Error: vLLM inference failed"