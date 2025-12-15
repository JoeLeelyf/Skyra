import json
import csv
import os
from collections import defaultdict
import numpy as np
import argparse

try:
    from sklearn.metrics import (
        balanced_accuracy_score,
        recall_score, 
        f1_score
        # 移除了 roc_auc_score
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("[Warning] 'scikit-learn' not found. This script 'pip install scikit-learn'\n")


# ===================== Helper Functions =====================

def _parse_video_id(video_id):
    """Splits a video_id like 'kling-v1/XYZ-123' into ('kling-v1', 'XYZ-123')"""
    try:
        parts = video_id.split('/', 1)
        if len(parts) == 2:
            return parts[0], parts[1]
        else:
            return None, None
    except:
        return None, None


def _load_and_prepare_data(json_file_path):
    """
    Loads data and creates paired results.
    For each base_id, it finds the 'real_pred' and the 'fake_pred' 
    for each AIGC model.
    """
    video_pairs = defaultdict(dict)

    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Pass 1: Collect all predictions, indexed by base_id
    for item in data:

        gt = item.get('gt')
        answer = item.get('answer')
        video_id = item.get('video_id')
        aigc_model = item.get('aigc_model_name')

        if gt is None or answer is None or video_id is None or aigc_model is None:
            continue

        _model_name_from_id, base_id = _parse_video_id(video_id)
        if not base_id:
            continue
        
        if aigc_model.lower() == 'real':
            # Store the model's prediction for the REAL video
            video_pairs[base_id]['real_pred'] = answer.lower()
        else:
            # Store the model's prediction for the FAKE video
            video_pairs[base_id][aigc_model] = {'pred': answer.lower()}

    # Pass 2: Build the final paired list, grouped by AIGC model
    paired_results = defaultdict(list)
    for base_id, preds in video_pairs.items():
        if 'real_pred' not in preds:
            # Skip if no 'Real' counterpart was found for this base_id
            continue

        real_prediction = preds['real_pred']

        for model_name, fake_data in preds.items():
            if model_name == 'real_pred':
                continue
            
            # We have a valid pair: (real_prediction, fake_prediction)
            fake_prediction = fake_data['pred']
            paired_results[model_name].append({
                'real_pred': real_prediction,
                'fake_pred': fake_prediction
            })
            
    return paired_results


# ===================== Metric Calculation =====================

DESIRED_MODEL_ORDER = [
    "Wan2.1-T2V-1.3B",
    "CogVideoX1.5-5B-T",
    "Wan2.2-TI2V-5B-T",
    "Wan2.2-TI2V-5B-I",
    "HunyuanVideo",
    "HunyuanVideo-I2V",
    "Wan2.1-VACE-1.3B-T",
    "Wan2.2-T2V-14B",
    "Wan2.2-I2V-14B",
    "SkyReels-V2",
    "SkyReels-V2-I2V-14B-540P",
    "LTX-Video-13B-T",
    "LTX-Video-13B-I",
    "gen4-turbo",
    "hailuo",
    "pika-v2",
    "pixverse-v4-5",
    "kling-v1",
    "sora-2"
]

def _calculate_paired_metrics(paired_results): # 移除了 csv_writer 参数
    """
    Calculates ACC, F1, Recall for each AIGC model.
    Prints to CLI as it goes.
    Returns all results for later CSV writing.
    """
    if not SKLEARN_AVAILABLE:
        print("[Error] 'scikit-learn' is required for this evaluation mode.")
        print("        Please run: pip install scikit-learn")
        return None, None, None

    print("\n--- [Report: Paired Real/Fake Metrics] ---")
    
    all_model_results = {}
    numeric_metrics_for_avg = defaultdict(list)
    
    # --- Build the list of models to print in the desired order ---
    all_present_models = set(paired_results.keys())
    ordered_models_to_print = []

    # 1. Add models from the desired order first
    for model_name in DESIRED_MODEL_ORDER:
        if model_name in all_present_models:
            ordered_models_to_print.append(model_name)
            all_present_models.remove(model_name) # Mark as added
    
    # 2. Add any remaining models (e.g., new models not in the list)
    if all_present_models:
        print(f"\n[Info] Found additional models not in custom order list (adding to end):")
        print(f"       {', '.join(sorted(all_present_models))}")
        ordered_models_to_print.extend(sorted(all_present_models))
    # --- End of ordering logic ---

    for model_name in ordered_models_to_print:
        model_results = {}
        pairs_list = paired_results[model_name]
        
        if not pairs_list:
            print(f"\n  > Model: {model_name:<20} | No valid pairs found.")
            model_results = {
                'num_pairs': 0, 
                'acc': 'N/A', 
                'recall': 'N/A', 
                'f1': 'N/A'
            }
            all_model_results[model_name] = model_results
            continue

        y_true_paired = []
        y_pred_paired = []
        
        for pair in pairs_list:
            y_true_paired.extend(['real', 'fake'])
            y_pred_paired.extend([pair['real_pred'], pair['fake_pred']])

        def encode_label(x):
            return 0 if x == 'real' else 1
        
        y_true_bin = np.array([encode_label(x) for x in y_true_paired])
        y_pred_bin = np.array([encode_label(x) for x in y_pred_paired])
        
        num_pairs = len(pairs_list)
        model_results['num_pairs'] = num_pairs
        
        # Calculate metrics
        try:
            acc = balanced_accuracy_score(y_true_bin, y_pred_bin)
            recall = recall_score(y_true_bin, y_pred_bin, pos_label=1, zero_division=0)
            f1 = f1_score(y_true_bin, y_pred_bin, pos_label=1, zero_division=0)

            numeric_metrics_for_avg['acc'].append(acc)
            numeric_metrics_for_avg['recall'].append(recall)
            numeric_metrics_for_avg['f1'].append(f1)

            model_results['acc'] = f"{acc * 100:.2f}"
            model_results['recall'] = f"{recall * 100:.2f}"
            model_results['f1'] = f"{f1 * 100:.2f}"

        except ValueError as e:
            print(f"[Warning] Could not calculate all metrics for {model_name}.")
            print(f"          Error: {e}. (This can happen if all preds are the same)")
            model_results.update({
                'acc': 'Error', 
                'recall': 'Error', 
                'f1': 'Error'
            })
            all_model_results[model_name] = model_results
            continue

        all_model_results[model_name] = model_results

        print(f"\n  > Model: {model_name:<20} | ({num_pairs} pairs)")
        print(f"    - Avg. ACC: {model_results['acc']}%")
        print(f"    - Recall:   {model_results['recall']}%")
        print(f"    - F1:       {model_results['f1']}%")

    # Calculate and print the average across all models
    avg_metrics_formatted = {'acc': 'N/A', 'recall': 'N/A', 'f1': 'N/A'}
    
    if numeric_metrics_for_avg:
        avg_acc = np.mean(numeric_metrics_for_avg['acc'])
        avg_recall = np.mean(numeric_metrics_for_avg['recall'])
        avg_f1 = np.mean(numeric_metrics_for_avg['f1'])
        
        avg_metrics_formatted = {
            'acc': f"{avg_acc * 100:.2f}",
            'recall': f"{avg_recall * 100:.2f}",
            'f1': f"{avg_f1 * 100:.2f}"
        }
        
        print("\n------------------------------------------------")
        print("  > Average Across All Fake Models")
        print(f"    - Avg. ACC: {avg_metrics_formatted['acc']}%")
        print(f"    - Recall:   {avg_metrics_formatted['recall']}%")
        print(f"    - F1:       {avg_metrics_formatted['f1']}%")

    return all_model_results, ordered_models_to_print, avg_metrics_formatted


# ===================== Main Evaluation =====================

def evaluate_model(json_file_path): 
    print(f"=== [STARTING PAIRED EVALUATION] ===")
    print(f"File: {json_file_path}")
    print("=" * 30)

    paired_results = _load_and_prepare_data(json_file_path)
    if not paired_results:
        print("Evaluation halted: No valid real/fake pairs were found in the data.")
        return

    all_metrics, ordered_models, avg_metrics = _calculate_paired_metrics(paired_results)
    
    if all_metrics is None:
        print("Evaluation failed, skipping CSV write.")
        return

    base_name, _ = os.path.splitext(json_file_path)
    csv_output_path = f"{base_name}_paired_metrics_transposed.csv" # 改了文件名
    print(f"\nSaving transposed results to: {csv_output_path}")

    try:
        with open(csv_output_path, 'w', newline='', encoding='utf-8') as csv_file:
            csv_writer = csv.writer(csv_file)

            header = ["Metric"] + ordered_models + ["Average"]
            csv_writer.writerow(header)
            
            row_num_pairs = ["Num_Pairs"] + \
                            [all_metrics[m]['num_pairs'] for m in ordered_models] + \
                            ["N/A"]
            csv_writer.writerow(row_num_pairs)
            
            row_acc = ["Avg_ACC"] + \
                      [all_metrics[m]['acc'] for m in ordered_models] + \
                      [avg_metrics['acc']]
            csv_writer.writerow(row_acc)

            row_recall = ["Recall"] + \
                         [all_metrics[m]['recall'] for m in ordered_models] + \
                         [avg_metrics['recall']]
            csv_writer.writerow(row_recall)

            row_f1 = ["F1"] + \
                     [all_metrics[m]['f1'] for m in ordered_models] + \
                     [avg_metrics['f1']]
            csv_writer.writerow(row_f1)


    except IOError as e:
        print(f"\n[ERROR] Could not write to CSV file: {e}")

    print(f"\n=== [EVALUATION COMPLETE] ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model predictions on real vs. fake video classification.")

    parser.add_argument(
        "--json_file_path", 
        type=str, 
        required=True,
        help="Path to the model predictions JSON file."
    )

    args = parser.parse_args()
    evaluate_model(args.json_file_path)