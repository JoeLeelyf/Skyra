import json
import os
import re
from abc import ABC, abstractmethod
from tqdm import tqdm
from pathlib import Path

SYSTEM_PROMPT = """\nYou are an expert AI video analyst. Your primary task is to review a sequence of video frames and provide a step-by-step analysis of their authenticity.\n\nYou MUST output your entire analysis using the following structure:\n1.  A `<think>...</think>` block containing your detailed reasoning.\n2.  An `<answer>...</answer>` block containing the final, one-word verdict: 'Fake' or 'Real'.\n\nInside the `<think>` block, you MUST:\n1.  Start by briefly describing the overall content of the video frames.\n2.  Follow a detailed, step-by-step \"discovery\" or \"verification\" process.\n3.  When you identify an artifact (or clear a region), you MUST use a valid L3 Category Name from the \"Artifact Category Definitions\" provided below.\n4.  You MUST embed your finding using the following exact tag structure:\n    <type>L3 Category Name</type> in <t>[startTime, endTime]</t> at <bbox>[x1, y1, x2, y2]</bbox>\n5.  If multiple artifacts are present, you must find and tag all of them in temporal order.\n6.  Your entire reasoning process must be self-contained\n\n---\n## Artifact Category Definitions (Valid L3 Categories for the <type> tag)\n\n### 1. Low-Level Forgery\n* **1.1 Texture Anomaly**:\n    * <type>Structure Anomaly</type>\n    * <type>Texture Jittering</type>\n    * <type>Unnatural Blur</type>\n* **1.2 Color and Lighting Anomaly**:\n    * <type>Color Over-saturation</type>\n    * <type>Lighting Inconsistency</type>\n* **1.3 Move Forgery**:\n    * <type>Camera Motion Inconsistency</type>\n\n### 2. Violation of Laws\n* **2.1 Object Inconsistency**:\n    * <type>Abnormal Object Disappearance</type>\n    * <type>Abnormal Object Appearance</type>\n    * <type>Person Identity Inconsistency</type>\n    * <type>General Object Identity Inconsistency</type>\n    * <type>Shape Distortion</type>\n* **2.2 Interaction Inconsistency**:\n    * <type>Abnormal Rigid-Body Crossing</type>\n    * <type>Abnormal Multi-Object Merging</type>\n    * <type>Abnormal Object Splitting</type>\n    * <type>General Interaction Anomaly</type>\n* **2.3 Unnatural Movement**:\n    * <type>Unnatural Human Movement</type>\n    * <type>Unnatural Animal Movement</type>\n    * <type>Unnatural General Object Movement</type>\n* **2.4 Violation of Causality Law**:\n    * <type>Violation of Physical Law</type>\n    * <type>Violation of General Causality Law</type>\n* **2.5 Violation of Commonsense**:\n    * <type>Abnormal Human Body Structure</type>\n    * <type>Abnormal General Object Structure</type>\n    * <type>Text Distortion</type>\n"""


class ViFBench(ABC):
    
    def __init__(self, index_json: str, model_path: str, model_name: str, save_dir: str = "./results"):
        self.index_json_path = index_json
        self.model_path = model_path
        self.model_name = model_name
        self.save_dir = save_dir
        self.SYSTEM_PROMPT = SYSTEM_PROMPT

        self.output_file = os.path.join(self.save_dir, f"{self.model_name}.json")

        os.makedirs(self.save_dir, exist_ok=True)
        
        self.all_tasks = []
        self.results = []
        self.processed_video_ids = set()

        print(f"Loading model {self.model_name} from {self.model_path}...")
        self.load_model()
        print("Model loaded.")

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def run_inference(self, frame_paths: list, user_prompt: str) -> str:
        pass

    def _load_data(self):
        print(f"Loading index file: {self.index_json_path}")
        with open(self.index_json_path, 'r') as f:
            index_data_ = json.load(f)
        # skip "full-videos"
        index_data = {}
        for k,v in index_data_.items():
            index_data[k] = [dir for dir in v if "full-videos" not in dir]
        
        if os.path.exists(self.output_file):
            print(f"Found existing results file for resuming: {self.output_file}")
            try:
                with open(self.output_file, 'r', encoding='utf-8') as f:
                    self.results = json.load(f)
                    self.processed_video_ids = {item['video_id'] for item in self.results}
                    print(f"Loaded {len(self.results)} results, {len(self.processed_video_ids)} video_ids processed.")
            except json.JSONDecodeError:
                self.results = []
                self.processed_video_ids = set()

        for aigc_model, frame_dirs in index_data.items():
            gt = "Real" if aigc_model == "real" or aigc_model == "Real" else "Fake"
            
            for frame_dir_path in frame_dirs:

                try:
                    parts = Path(frame_dir_path).parts
                    base_index = parts.index('test_normalized')
                    relevant_parts = parts[base_index+1:]
                    video_id = "/".join(relevant_parts)
                    
                except (ValueError, IndexError):
                    p = Path(frame_dir_path)
                    video_id = f"{p.parent.name}/{p.name}"

                if video_id in self.processed_video_ids:
                    continue

                task = {
                    "frame_dir_path": frame_dir_path,
                    "video_id": video_id,
                    "aigc_model_name": aigc_model,
                    "gt": gt
                }
                self.all_tasks.append(task)
                
        print(f"All task: {len(index_data.items())} categories, {sum(len(v) for v in index_data.values())} videos.")
        print(f"Processed: {len(self.processed_video_ids)} videos.")
        print(f"Remaining tasks: {len(self.all_tasks)} videos.")

    def _build_user_prompt(self, frame_dir_path: str):
        timestamps_file = os.path.join(frame_dir_path, "timestamps.txt")
        if not os.path.exists(timestamps_file):
            print(f"  [Warning] Missing {timestamps_file}, skipping this video.")
            return None, None

        with open(timestamps_file, 'r') as f:
            timestamps = [float(line.strip()) for line in f]

        prompt_lines = ["Here are the video frames and their corresponding timestamps:"]
        frame_paths = []

        for i, ts in enumerate(timestamps):
            frame_num = i + 1
            
            # 1. Construct frame path
            frame_path = os.path.join(frame_dir_path, f"{frame_num}.png")
            if not os.path.exists(frame_path):
                 print(f"  [Warning] Frame not found: {frame_path} (recorded in {timestamps_file}), skipping this video.")
                 return None, None
            
            # 2. (New format) Add text line with <image> placeholder
            prompt_lines.append(f"[T={ts:.2f}s] <image>")
            
            # 3. (新格式) 添加帧的绝对路径
            frame_paths.append(os.path.abspath(frame_path))
        
        prompt_lines.append("\nPlease analyze the video frames, determine if the video is real or fake, and provide your reasoning.")
        
        return "\n".join(prompt_lines), frame_paths

    def _parse_response(self, response: str) -> str:
        match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
        if match:
            answer = match.group(1).strip()
            if 'fake' in answer.lower():
                return 'Fake'
            if 'real' in answer.lower():
                return 'Real'
            return answer
        else:
            print(f"[Warning] {response[:200]}...")
            return "Error"

    def _save_results(self, final=False):
        mode = "final" if final else "temp"
        print(f"\n[{mode} Save] Saving {len(self.results)} results to {self.output_file}...")
        try:
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"  [Error] Failed to save results: {e}")
            # Attempt to save backup
            try:
                backup_file = self.output_file + ".bak"
                print(f"  [Attempt] Saving backup to {backup_file}")
                with open(backup_file, 'w', encoding='utf-8') as f:
                    json.dump(self.results, f, ensure_ascii=False, indent=4)
            except Exception as e_bak:
                print(f"  [Error] Backup save failed: {e_bak}")


    def run(self):

        self._load_data()
        
        if not self.all_tasks:
            print("No tasks to process. Exiting.")
            return

        pbar = tqdm(self.all_tasks, desc=f"LADMBench Evaluation ({self.model_name} on {self.dataset_set_name})")
        
        for i, task in enumerate(pbar):
            frame_dir = task['frame_dir_path']
            pbar.set_postfix_str(f"Processing: {task['video_id']}")

            user_prompt, frame_paths = self._build_user_prompt(frame_dir)
            if not user_prompt or not frame_paths:
                continue

            response_text = self.run_inference(frame_paths, user_prompt)

            answer = self._parse_response(response_text)

            result_item = {
                "video_id": task['video_id'],
                "aigc_model_name": task['aigc_model_name'],
                "mllm_model_name": self.model_name,
                "gt": task['gt'],
                "response": response_text,
                "answer": answer
            }

            self.results.append(result_item)

            if (i + 1) % 10 == 0:
                self._save_results(final=False)

        self._save_results(final=True)
        print("评测完成。")