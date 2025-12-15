import torch
import re
from transformers import AutoModelForCausalLM, AutoProcessor
from utils.ViFBench import ViFBench

class VideoLLaMA3Model(ViFBench):
    """
    Concrete implementation of the VideoLLaMA3-7B model.
    Inherits from ViFBench and implements model loading and inference.

    (Based on example_videollama3.py)
    """

    def load_model(self):
        """
        (VideoLLaMA3.py requirement) Load the VideoLLaMA3 model and Processor.
        """
        print(f"Loading model {self.model_name} from {self.model_path}...")

        # (Based on example_videollama3.py)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )

        # Get the device the model was loaded onto
        self.device = self.model.device
        self.model.eval()
        print(f"Model loaded to device: {self.device}")

    def run_inference(self, frame_paths: list, user_prompt: str) -> str:
        """
        (VideoLLaMA3.py requirement) Run inference for VideoLLaMA3.

        This method takes a user_prompt string containing <image> placeholders and a frame_paths list,
        and reconstructs them into the interleaved text-image 'content' list format required by VideoLLaMA3.

        Args:
            frame_paths (list): List of absolute paths (e.g., ['/path/to/1.png', ...])
            user_prompt (str): User prompt text containing <image> placeholders

        Returns:
            str: The model's raw text response
        """

        # 1. Reconstruct the user_content list from user_prompt and frame_paths
        # (This logic is the same as in Qwen2_5_VL.py and APIModel.py)
        user_content = []
        frame_path_iter = iter(frame_paths)

        # Split text by <image> placeholders
        text_parts = re.split(r'<image>', user_prompt)

        for i, text_part in enumerate(text_parts):
            # Add the text part (if non-empty)
            if text_part.strip():
                user_content.append({
                    "type": "text",
                    "text": text_part
                })

            # Add an image after each text part except the last
            if i < len(text_parts) - 1:
                try:
                    frame_path = next(frame_path_iter)

                    # (Key) VideoLLaMA3 format (based on example_videollama3.py)
                    user_content.append({
                        "type": "image",
                        "image": {"image_path": frame_path}
                    })
                except StopIteration:
                    print(f"  [VideoLLaMA3 WARNING] More <image> tags in the prompt than images in the frame_paths list.")
                    break

        # 2. Build the VideoLLaMA3 conversation format
        conversation = [
            {
                "role": "system",
                "content": self.SYSTEM_PROMPT  # (Key) Use the ViFBench System Prompt
            },
            {
                "role": "user",
                "content": user_content,  # (Key) Insert the reconstructed interleaved list
            }
        ]

        # 3. Run inference (based on the infer function in example_videollama3.py)

        with torch.inference_mode():
            # 3.1. Processor
            inputs = self.processor(
                conversation=conversation,
                add_system_prompt=False,  # We already manually added ViFBench's system prompt
                add_generation_prompt=True,
                return_tensors="pt"
            )

            # 3.2. Move to device
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

            # 3.3. Cast pixel_values (as shown in the example)
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

            # 3.4. Generate
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=1024,  # ViFBench requirement
                do_sample=False       # Ensure consistent results
            )

            response = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            return response