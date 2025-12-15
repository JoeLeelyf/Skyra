import os
import base64
import io
import time
import re  # (NEW) Import regular expressions
from PIL import Image
from openai import OpenAI
from utils.ViFBench import ViFBench  # Ensure ViFBench.py is in the same directory or on the Python path

class APIModel(ViFBench):
    """
    ViFBench implementation for API-based models (e.g., GPT-4o, Claude, etc.).
    (UPDATED)
    This version supports the interleaved text-and-image format passed by ViFBench.
    It takes a user_prompt string containing <image> placeholders and a frame_paths list,
    and reconstructs them into the "content" list format required by the API.
    """

    def __init__(self, index_json: str, model_name: str, dataset_set_name: str, save_dir: str = "./results", api_key: str = None, base_url: str = None):
        """
        Initialize the API model evaluator.

        (UPDATED) Removed the 'max_frames' parameter because it is incompatible with the
        interleaved prompt format (the number of <image> tokens in the prompt must match
        the length of the frame_paths list).
        """

        # 1. Configure API key and URL
        # (Reference GPT.py) Prefer environment variables, then arguments, then a hard-coded default
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.apiyi.com/v1"  # Use the default from your GPT.py

        print(f"Using API Base URL: {self.base_url}")

        if not self.api_key:
            raise ValueError("API key not provided. Set OPENAI_API_KEY environment variable or pass via api_key argument.")

        print(f"APIModel Config: model_name='{model_name}', base_url='{self.base_url}'")

        # 2. Call parent __init__
        # ViFBench requires 'model_path'; we repurpose it as 'base_url' for logging
        super().__init__(
            index_json=index_json,
            model_path=self.base_url,  # Reuse the model_path field
            model_name=model_name,
            dataset_set_name=dataset_set_name,
            save_dir=save_dir
        )

    def load_model(self):
        """
        (APIModel requirement) Initialize the API client.
        """
        print(f"Initializing OpenAI client (Base URL: {self.base_url})...")
        try:
            self.client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key
            )
            # (APIModel requirement) Add retries and timeout due to unstable networks
            self.client = self.client.with_options(max_retries=3, timeout=120.0)
            print("API client initialization complete.")
        except Exception as e:
            print(f"Error: Failed to initialize OpenAI client: {e}")
            raise

    def _encode_image_pil(self, image_path: str, detail: str = "low"):
        """
        (APIModel requirement) Encode a frame file provided by ViFBench into Base64.
        Uses PIL to handle format conversion (e.g., PNG -> JPEG) and resizing (for "low" detail).
        """
        clean_path = image_path.replace("file://", "")
        if not os.path.exists(clean_path):
            print(f"  [APIModel ERROR] Frame not found: {clean_path}")
            return None, False

        try:
            with Image.open(clean_path) as img:
                # Ensure RGB (JPEG does not support RGBA)
                if img.mode == 'RGBA':
                    img = img.convert('RGB')

                if detail == "low":
                    # Resize to match OpenAI's "low" detail definition (512px square).
                    # To preserve aspect ratio, scale by making the shorter side 512px.
                    w, h = img.size
                    if w < h:
                        new_w = 512
                        new_h = int(h * (512 / w))
                    else:
                        new_h = 512
                        new_w = int(w * (512 / h))
                    img = img.resize((new_w, new_h), Image.LANCZOS)

                # Save to an in-memory JPEG
                buffered = io.BytesIO()
                img.save(buffered, format="JPEG", quality=85)
                base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

                # Return in data URI format
                return f"data:image/jpeg;base64,{base64_image}", True
        except Exception as e:
            print(f"  [APIModel ERROR] Failed to encode image {clean_path}: {e}")
            return None, False

    def run_inference(self, frame_paths: list, user_prompt: str) -> str:
        """
        (APIModel requirement - MODIFIED) Perform an inference call to an OpenAI-compatible API.

        This method takes a user_prompt string containing <image> placeholders and a frame_paths list,
        and reconstructs them into the interleaved text-and-image "content" list format required by the API.

        Args:
            frame_paths (list): List in the format 'file:///path/to/1.png'
            user_prompt (str): User prompt text containing <image> placeholders

        Returns:
            str: The model's raw text response
        """

        # 1. Build the message list
        system_message = {"role": "system", "content": self.SYSTEM_PROMPT}

        user_content = []

        # (NEW) 2. Reconstruct user_content from user_prompt and frame_paths
        frame_path_iter = iter(frame_paths)

        # Split text by <image> placeholders using regular expressions
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
                except StopIteration:
                    print(f"  [APIModel WARNING] More <image> tags in the prompt than images in the frame_paths list.")
                    break

                # (APIModel requirement) Encode to Base64
                base64_url, success = self._encode_image_pil(frame_path, detail="low")

                if not success:
                    return f"Error: Failed to encode image {frame_path}"

                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": base64_url,
                        "detail": "low"  # Use "low" for video frames
                    }
                })

        user_message = {"role": "user", "content": user_content}
        messages = [system_message, user_message]

        # 3. Call the API (with retries)
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,  # e.g., "gpt-4o", "anthropic/claude-3.5-sonnet"
                messages=messages,
                temperature=0.1,  # Use a low temperature to ensure reproducibility
                max_tokens=1024   # Ensure there is enough space for the <think> block
            )
            return response.choices[0].message.content

        except Exception as e:
            print(f"  [APIModel CRITICAL ERROR] API call failed: {e}")
            # (APIModel requirement) Add one simple delayed retry in addition to base class retries
            time.sleep(10)
            try:
                print("  [APIModel] Retrying...")
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=1024
                )
                return response.choices[0].message.content
            except Exception as e2:
                print(f"  [APIModel CRITICAL ERROR] Retry failed: {e2}")
                return f"API Call Error (Retry Failed): {e2}"