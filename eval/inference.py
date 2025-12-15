import argparse
import multiprocessing

def main():
    parser = argparse.ArgumentParser(description="运行 LADMBench 评测")
    
    parser.add_argument(
        "--index_json", 
        type=str, 
        required=True,
        help="Path to test index"
    )
    
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True,
        help="Path to the model checkpoint or API model identifier."
    )

    # For API models, model_path will be used as base_url
    parser.add_argument(
        "--api_key", 
        type=str, 
        default=None,
        help="API Key for accessing the model (if applicable)。"
    )
    
    parser.add_argument(
        "--base_url", 
        type=str, 
        default=None,
        help="Base URL for the API (default: read from $OPENAI_BASE_URL or use 'https://api.apiyi.com/v1')."
    )
    
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="Qwen2.5-VL-7B",
        help="Name of the model to evaluate (e.g., 'Qwen2.5-VL-7B', 'gpt-4o', 'gemini-pro')."
    )
    
    parser.add_argument(
        "--save_dir", 
        type=str, 
        default="./results",
        help="Directory to save evaluation results."
    )
    
    args = parser.parse_args()

    if "vLLM" in args.model_name:
        from models.Qwen2_5_VL_vLLM import vLLMModel
        evaluator = vLLMModel(
            index_json=args.index_json,
            model_path=args.model_path,
            model_name=args.model_name,
            save_dir=args.save_dir
        )
    elif "BusterX" in args.model_name:
        from models.BusterX_vLLM import BusterXModel
        evaluator = BusterXModel(
            index_json=args.index_json,
            model_path=args.model_path,
            model_name=args.model_name,
            save_dir=args.save_dir
        )
    elif "VideoLLaMA3" in args.model_name:
        from models.VideoLLaMA3 import VideoLLaMA3Model
        evaluator = VideoLLaMA3Model(
            index_json=args.index_json,
            model_path=args.model_path,
            model_name=args.model_name,
            save_dir=args.save_dir
        )
    elif "Qwen2.5-VL" in args.model_name or "Skyra" in args.model_name:
        from models.Qwen2_5_VL import Qwen2Model
        evaluator = Qwen2Model(
            index_json=args.index_json,
            model_path=args.model_path,
            model_name=args.model_name,
            save_dir=args.save_dir
        )
    elif "InternVL3" in args.model_name:
        from models.InternVL3 import InternVL3Model 
        evaluator = InternVL3Model(
            index_json=args.index_json,
            model_path=args.model_path,
            model_name=args.model_name,
            save_dir=args.save_dir
        )
    elif "gpt" in args.model_name or "gemini" in args.model_name:
        from models.APIModel import APIModel
        evaluator = APIModel(
            index_json=args.index_json,
            model_name=args.model_name,
            save_dir=args.save_dir,
            api_key=args.api_key,
            base_url=args.base_url
        )
        
    evaluator.run()


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    main()
