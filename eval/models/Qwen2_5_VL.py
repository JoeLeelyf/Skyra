import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
# from qwen_vl_utils import process_vision_info  # (REMOVED) No longer needed
from utils.ViFBench import ViFBench
import re  # Import re for more complex text processing

class Qwen2Model(ViFBench):
    """
    Concrete implementation of the Qwen2.5-VL model.
    Inherits from ViFBench and implements model loading and inference.
    """

    def load_model(self):
        """
        (Qwen2.5VL.py requirement) Load the Qwen2.5-VL model and Processor.
        """
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )

        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.device = self.model.device
        self.model.eval()
        print(f"Model loaded to device: {self.device}")

    def run_inference(self, frame_paths: list, user_prompt: str) -> str:
        """
        (Qwen2.5VL.py requirement - MODIFIED) Run inference for Qwen2.5-VL.

        This method takes a user_prompt string containing <image> placeholders and a frame_paths list,
        and reconstructs them into the required interleaved list-of-dicts format.

        Args:
            frame_paths (list): List in the format 'file:///path/to/1.png'
            user_prompt (str): User prompt text containing <image> placeholders

        Returns:
            str: The model's raw text response
        """

        # (NEW) 1. Reconstruct the user_content list from user_prompt and frame_paths
        user_content = []
        frame_path_iter = iter(frame_paths)

        # Split text by <image> placeholders using regular expressions
        # This preserves the text around the <image> placeholders.
        # Example: "text1 <image> text2 <image> text3"
        # -> ["text1 ", " text2 ", " text3"]
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
                    user_content.append({
                        "type": "image",
                        "image": next(frame_path_iter)
                    })
                except StopIteration:
                    print(f"  [WARNING] More <image> tags in the prompt than images in the frame_paths list.")
                    break


        # (NEW) 2. Construct the required messages format
        messages = [
            {
                "role": "system",
                "content": self.SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": user_content,  # (Key change)
            }
        ]

        # 3. Prepare the chat template
        # apply_chat_template will process this messages list,
        # extract images, and convert text into a format with <image> placeholders.
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # 4. Prepare the overall model inputs
        # processor() still needs to receive text and images separately
        inputs = self.processor(
            text=[text],
            images=frame_paths,  # (Key) images passed as a separate argument
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        # 5. Run inference
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=False
            )

        # 6. Decode
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return output_text[0] if output_text else "Error: Empty response from model"