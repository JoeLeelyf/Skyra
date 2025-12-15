import math
import torch
import numpy as np
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, AutoConfig

# Import the base class from the ViFBench framework
from utils.ViFBench import ViFBench

#
# -----------------------------------------------------------------
# (1) Helper functions copied from the official InternVL3 script
# -----------------------------------------------------------------
#

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    """
    (From the official script)
    Load a single image file, apply dynamic preprocessing (tiling), and return the stacked tensor.
    """
    try:
        image = Image.open(image_file).convert('RGB')
        transform = build_transform(input_size=input_size)
        images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values
    except Exception as e:
        print(f"  [ERROR] Failed to load or process image: {image_file}, Error: {e}")
        return None

def split_model(model_path, model_name):
    """
    (From the official script)
    Compute the device_map for multi-GPU setups.
    """
    device_map = {}
    world_size = torch.cuda.device_count()
    if world_size == 1:
         return "auto"  # Use auto for single GPU

    try:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        num_layers = config.llm_config.num_hidden_layers
    except Exception as e:
        print(f"  [WARNING] Unable to auto-compute device_map (Error: {e}). Falling back to 'auto'.")
        return "auto"

    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)

    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            if layer_cnt >= num_layers:  # Prevent out-of-bounds indexing
                break
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1

    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map['language_model.lm_head'] = 0
    # Ensure the last layer is on GPU 0 (or the primary GPU)
    if (num_layers - 1) in device_map:
        device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    print(f"Automatically generated device_map for {world_size} GPUs.")
    return device_map


#
# -----------------------------------------------------------------
# (2) InternVL3Model class adapted to the ViFBench framework
# -----------------------------------------------------------------
#

class InternVL3Model(ViFBench):
    """
    Concrete implementation of the InternVL3 model.
    Inherits from ViFBench and implements model loading and inference.
    """

    def load_model(self):
        """
        (ViFBench requirement) Load the InternVL3 model and tokenizer.
        """

        # 1. Compute device_map
        # (Note: 'InternVL3-8B' is hard-coded in the official script; we use self.model_name instead)
        device_map = split_model(self.model_path, self.model_name)

        # 2. Load the model
        # (Note: load_in_8bit=False is from the official example; adjust based on your hardware)
        self.model = AutoModel.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            load_in_8bit=False,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map=device_map
        ).eval()

        # 3. Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            use_fast=False
        )

        # 4. Define generation_config used for evaluation
        # (Note: official examples use do_sample=True; for evaluation use do_sample=False for reproducibility)
        self.generation_config = dict(
            max_new_tokens=1024,
            do_sample=False
        )

        # 5. (Optional) Define image processing parameters
        self.input_size = 448
        self.max_num_tiles = 12  # Corresponds to max_num=12 in the official script

        print(f"InternVL3 model '{self.model_name}' loaded successfully.")

    def run_inference(self, frame_paths: list, user_prompt: str) -> str:
        """
        (ViFBench requirement) Run inference for InternVL3.

        This method perfectly matches InternVL3's "multi-image multi-round conversation, separate images"
        mode (multiple images, multi-round conversation, independent images).

        Args:
            frame_paths (list): List in the format 'file:///path/to/1.png'
            user_prompt (str): User prompt text containing <image> placeholders

        Returns:
            str: The model's raw text response
        """

        pixel_values_list = []
        num_patches_list = []

        # 1. Iterate over all frame paths provided by ViFBench
        for frame_path in frame_paths:
            # Use the official load_image function (includes dynamic tiling)
            pixel_values_frame = load_image(
                frame_path,
                input_size=self.input_size,
                max_num=self.max_num_tiles
            )

            if pixel_values_frame is None:
                continue  # Skip images that failed to load

            pixel_values_list.append(pixel_values_frame)
            # Record how many tiles (patches) this frame was split into
            num_patches_list.append(pixel_values_frame.size(0))

        # 2. Check whether there are valid images
        if not pixel_values_list:
            print("  [WARNING] Failed to load any valid frames.")
            return "Error: No valid frames loaded."

        # 3. Concatenate all tiles from all frames
        pixel_values = torch.cat(pixel_values_list, dim=0)

        # 4. Move to device
        # (Official example uses .cuda(); this is standard in multi-GPU device_map mode)
        pixel_values = pixel_values.to(torch.bfloat16).cuda()

        # 5. Prepare the final prompt
        # The ViFBench base class provides SYSTEM_PROMPT; we need to combine it with user_prompt
        final_question = f"{self.SYSTEM_PROMPT}\n\n{user_prompt}"

        # 6. Run inference
        # We use model.chat() and pass num_patches_list.
        # This tells the model how to align the pixel_values tensor with the <image> tags in the prompt.
        try:
            with torch.inference_mode():
                response = self.model.chat(
                    self.tokenizer,
                    pixel_values,
                    final_question,
                    self.generation_config,
                    num_patches_list=num_patches_list,
                    history=None,       # Stateless single-round evaluation each time
                    return_history=False
                )
            return response

        except Exception as e:
            print(f"  [CRITICAL ERROR] InternVL3 inference failed: {e}")
            import traceback
            traceback.print_exc()
            return f"Error: Inference failed ({e})"