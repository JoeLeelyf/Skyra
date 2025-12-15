import torch
import re
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
# Ensure you have installed qwen_vl_utils (pip install qwen-vl-utils)
from qwen_vl_utils import process_vision_info
from utils.ViFBench import ViFBench  # Import the base class from your framework

# ---------------- 1. BusterX-specific prompts (keep as required) ----------------
SYSTEM = ("A conversation between User and Assistant. The user asks a question, "
          "and the Assistant solves it. The assistant first thinks about the reasoning process "
          "in the mind and then provides the user with the answer. The reasoning process and answer "
          "are enclosed within <think> </think> and <answer> </answer> tags, respectively, "
          "i.e., <think> reasoning process here </think><answer> answer here </answer>")

USER = ("Please analyze whether there are any inconsistencies or obvious signs of forgery in the video, and finally come to a conclusion: Is this video real or fake? Please think about this question as if you were a human pondering deeply. Engage in an internal dialogue using expressions such as ’let me think’, ’wait’, ’Hmm’, ’oh, I see’, ’let’s break it down’, etc, or other natural language thought expressions. It’s encouraged to include self-reflection or verification in the reasoning process. Then, just answer this MCQ with a single letter: Q: Is this video real or fake? Options: A) real B) fake")


class BusterXModel(ViFBench):
    """
    BusterX (vLLM) implementation for the ViFBench framework.

    This implementation forces the use of BusterX-specific SYSTEM and USER prompts,
    and passes the frame list provided by ViFBench to vLLM using the {"type": "video"} format.
    """

    def load_model(self):
        """
        (BusterX-vLLM requirement) Load the vLLM engine, Processor, and SamplingParams.
        """

        print("start init")
        # 1. Load the vLLM engine
        # (Key change) Disable image and enable video.
        # ViFBench may pass many frames, so set a higher limit (e.g., 64).
        self.llm = LLM(
            model=self.model_path,
            limit_mm_per_prompt={"image": 0, "video": 64},
            tensor_parallel_size=torch.cuda.device_count()
        )
        print("end init")

        # 2. Load the Processor
        self.processor = AutoProcessor.from_pretrained(self.model_path)

        # 3. Define SamplingParams
        # Match do_sample=False in BusterX_original.py
        self.sampling_params = SamplingParams(
            temperature=0,
            top_p=1.0,
            max_tokens=8192*32
        )
        print(f"vLLM (BusterX) model {self.model_path} loaded with {torch.cuda.device_count()} GPUs.")

    def _build_messages(self, frame_paths: list) -> list:
        """
        (BusterX requirement)
        Build messages using the fixed SYSTEM and USER prompts and a "video" type input.
        """
        return [
            {"role": "system", "content": SYSTEM},
            {"role": "user",   "content": [
                # (Key) We use the frame_paths list provided by ViFBench
                {"type": "video", "video": frame_paths},
                {"type": "text",  "text": USER}
            ]}
        ]

    def run_inference(self, frame_paths: list, user_prompt: str) -> str:
        """
        (BusterX-vLLM requirement) Run vLLM inference.

        This method *ignores* the incoming user_prompt and uses BusterX's fixed prompts.
        It uses the frame_paths list provided by ViFBench.
        """

        # 1. (BusterX requirement) Build messages (ignore ViFBench's user_prompt)
        #    We use frame_paths provided by the base class instead of sampling ourselves.
        messages = self._build_messages(frame_paths)

        try:
            # 2. (vLLM format) Apply Chat Template
            prompt = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            # 3. (vLLM format) Process vision information
            # Since our message type is "video", this will populate video_inputs
            image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

            # 4. (Key change) Prepare multimodal data (mm_data)
            #    Populate the "video" field instead of "image"
            mm_data = {}
            if video_inputs is not None:
                mm_data["video"] = video_inputs
            else:
                # Image input should also be empty, but just in case
                if image_inputs is not None:
                     print("  [WARNING] BusterX vLLM: Unexpected 'image' input detected, but configured for 'video'.")
                     mm_data["image"] = image_inputs

            # 5. (vLLM format) Build llm_inputs
            llm_inputs = {
                "prompt": prompt,
                "multi_modal_data": mm_data,
                "mm_processor_kwargs": video_kwargs,
            }

            # 6. (vLLM format) Run inference
            outputs = self.llm.generate([llm_inputs], sampling_params=self.sampling_params)
            generated_text = outputs[0].outputs[0].text

            return generated_text

        except Exception as e:
            print(f"  [CRITICAL ERROR] BusterX vLLM inference failed: {e}")
            # Try printing the failed prompt (without vision information)
            safe_prompt = prompt if 'prompt' in locals() else "Prompt generation failed"
            print(f"  Failed Prompt (text only): {safe_prompt[:200]}...")
            return "Error: vLLM inference failed"