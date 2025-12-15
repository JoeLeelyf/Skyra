# Copyright 2024 Bytedance Ltd. and/or its affiliates
# (Modifications by user for the AIGC detection task)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import math
from mathruler.grader import extract_boxed_content, grade_answer  # Keep the original import in case the grpo framework depends on it

# --- Define evidence regular expressions for the AIGC task ---

# Match "Fake" evidence: <type>...</type> in <t>...</t> at <bbox>...</bbox>
# P_FAKE_EVIDENCE = re.compile(
#     r"<type>.*?</type>\s*in\s*<t>.*?</t>\s*at\s*<bbox>.*?</bbox>",
#     re.DOTALL
# )

# Match "Real" evidence: <t>...</t> at <bbox>...</bbox>
P_REAL_EVIDENCE = re.compile(
    r"<t>.*?</t>\s*at\s*<bbox>.*?</bbox>",
    re.DOTALL
)

P_FAKE_EVIDENCE = P_REAL_EVIDENCE

def format_reward_aigc(predict_str: str) -> float:
    """
    Check the output format for the AIGC task.
    - A "Fake" answer must contain at least one <type>... in <t>... at <bbox>...</bbox> piece of evidence.
    - A "Real" answer must contain at least one <t>... at <bbox>...</bbox> piece of evidence and must not contain "Fake" evidence.
    """
    think_match = re.search(r"<think>(.*?)</think>", predict_str, re.DOTALL)
    answer_match = re.search(r"<answer>(.*?)</answer>", predict_str, re.DOTALL)

    # 1. Must contain both <think> and <answer>
    if not think_match or not answer_match:
        return 0.0

    think_content = think_match.group(1)
    answer_content = answer_match.group(1).strip()

    # 2. Check evidence format in the <think> block based on <answer> content
    if answer_content == "Fake":
        # If Fake, must find at least one Fake evidence
        num_fake = len(P_FAKE_EVIDENCE.findall(think_content))
        return 1.0 if num_fake > 0 else 0.0

    elif answer_content == "Real":
        # If Real, must find at least one Real evidence
        # And must not contain Fake evidence (ensure by removing Fake evidence first, then searching for Real evidence)
        string_without_fake = P_FAKE_EVIDENCE.sub("", think_content)
        num_real = len(P_REAL_EVIDENCE.findall(string_without_fake))
        return 1.0 if num_real > 0 else 0.0

    else:
        # <answer> is neither "Fake" nor "Real" -> format error
        return 0.0

def acc_reward_aigc(predict_str: str, ground_truth: str) -> float:
    """
    Compute accuracy and evidence reward for the AIGC task.
    1. Gating: if <answer> (Fake/Real) is wrong, reward is 0.
    2. Evidence reward: if <answer> is correct, then depending on ground_truth type,
       count the number of valid evidence items in the <think> block, and use log(1+N) as the reward.
    """
    ground_truth_answer = ground_truth.strip()

    # 1. Extract the predicted answer and apply gating
    answer_match = re.search(r"<answer>(.*?)</answer>", predict_str, re.DOTALL)

    if not answer_match:
        return 0.0  # No <answer> tag found -> wrong answer

    predicted_answer = answer_match.group(1).strip()

    if predicted_answer != ground_truth_answer:
        return 0.0  # Wrong answer (Fake/Real) -> reward 0

    # 2. Answer is correct; compute evidence reward
    think_match = re.search(r"<think>(.*?)</think>", predict_str, re.DOTALL)
    if not think_match:
        return 0.0  # Correct answer but no <think> block -> no evidence

    think_content = think_match.group(1)
    num_evidence = 0

    if ground_truth_answer == "Fake":
        # Count "Fake" evidence items
        num_evidence = len(P_FAKE_EVIDENCE.findall(think_content))

    elif ground_truth_answer == "Real":
        # Count "Real" evidence items
        # Remove all "Fake" evidence first to prevent the model from using "Fake" evidence to masquerade as "Real" evidence
        string_without_fake = P_FAKE_EVIDENCE.sub("", think_content)
        num_evidence = len(P_REAL_EVIDENCE.findall(string_without_fake))

    # 3. Use log1p(N) = log(1+N) as reward
    # 0 evidence -> 0.0
    # 1 evidence -> ~0.693
    # 2 evidence -> ~1.098
    return min(math.log1p(num_evidence), math.log1p(3))  # Cap at log1p(3)


def compute_score_aigc(predict_str: str, ground_truth: str) -> float:
    """
    Compute the total score for the AIGC task. (V3 - No negative rewards)

    Core logic:
    1. Any error (wrong answer, format error) yields a reward of 0.0.
    2. Only when the answer is correct AND the format is correct, grant a positive reward of
       (0.5 * log1p(N) + 0.5 * 1.0).
    """
    ground_truth_answer = ground_truth.strip()

    # --- 1. Extract the predicted answer ---
    answer_match = re.search(r"<answer>(.*?)</answer>", predict_str, re.DOTALL)

    if not answer_match:
        return 0.0  # Format error: no <answer> tag

    predicted_answer = answer_match.group(1).strip()

    # --- 2. Answer gating (core fix) ---
    # Check any type of wrong answer
    # if predicted_answer != ground_truth_answer:
    #     # Whether "GT=Real, Pred=Fake" or "GT=Fake, Pred=Real"
    #     # As long as the answer is wrong, reward is 0.0
    #     # This fixes the V1 bug where "GT=Real, Pred=Fake" received +0.5
    #     return 0.0
    if ground_truth_answer == "Real" and predicted_answer == "Fake":
        return -0.2
    if ground_truth_answer == "Fake" and predicted_answer == "Real":
        return 0.0

    # --- 3. Answer is correct; compute format and evidence rewards ---
    # If execution reaches here, it means predicted_answer == ground_truth_answer

    # 1. Set weights
    format_score_val = 0.2
    acc_score_val = 1.0 - format_score_val

    # 2. Compute the two sub-rewards
    # f_reward: check whether the format is correct (e.g., whether a Fake answer actually includes <type> evidence)
    f_reward = format_reward_aigc(predict_str)

    # a_reward: compute log1p(N) evidence reward
    # (acc_reward_aigc already contains answer gating, but this is the correct-answer path so it will compute log1p)
    a_reward = acc_reward_aigc(predict_str, ground_truth)

    # 3. Return weighted sum
    # If the answer is correct (a_reward>0) but the format is wrong (f_reward=0), total score is still 0.0
    # Only when the answer is correct AND the format is correct can the reward be > 0
    # return (acc_score_val * a_reward) + (format_score_val * f_reward)
    return (acc_score_val * a_reward)

if __name__ == '__main__':
    # Scenario 1: Correct "Fake" prediction (from your example)
    pred_fake_correct = """<think>The video shows a person's hands holding a glass jar...
    This is a clear example of <type>Camera Motion Inconsistency</type> in <t>[3.89, 4.53]</t> at <bbox>[0.0, 0.0, 0.0, 0.0]</bbox>,
    where the camera's movement is erratic and jarring...</think>
    <answer>Fake</answer>"""
    gt_fake = "Fake"
    # Expected (V3): format=1.0, acc=log1p(1)≈0.693
    # Total: 0.5 * 0.693 + 0.5 * 1.0 = 0.8465
    print(f"--- Scenario 1: Correct Fake ---")
    print(f"Format: {format_reward_aigc(pred_fake_correct)}")
    print(f"Acc: {acc_reward_aigc(pred_fake_correct, gt_fake)}")
    print(f"Total: {compute_score_aigc(pred_fake_correct, gt_fake)}\n")

    # Scenario 2: Wrong "Fake" prediction (ground truth is Real)
    gt_real = "Real"
    # Expected (V3): triggers "predicted != ground_truth" gating
    # Total: 0.0 (this fixes the +0.5 bug in V1)
    print(f"--- Scenario 2: Wrong Fake answer (GT=Real) ---")
    print(f"Format: {format_reward_aigc(pred_fake_correct)}")
    print(f"Acc: {acc_reward_aigc(pred_fake_correct, gt_real)}")
    print(f"Total: {compute_score_aigc(pred_fake_correct, gt_real)}\n")

    # Scenario 3: Correct "Real" prediction
    pred_real_correct = """<think>The motion seems natural.
    Here is a normal segment: <t>[1.50, 2.50]</t> at <bbox>[0.0, 0.0, 0.0, 0.0]</bbox>.
    Here is another one: <t>[3.0, 4.0]</t> at <bbox>[0.1, 0.1, 0.5, 0.5]</bbox>.
    </think>
    <answer>Real</answer>"""
    gt_real = "Real"
    # Expected (V3): format=1.0, acc=log1p(2)≈1.098
    # Total: 0.5 * 1.098 + 0.5 * 1.0 = 1.049
    print(f"--- Scenario 3: Correct Real (2 pieces of evidence) ---")
    print(f"Format: {format_reward_aigc(pred_real_correct)}")
    print(f"Acc: {acc_reward_aigc(pred_real_correct, gt_real)}")
    print(f"Total: {compute_score_aigc(pred_real_correct, gt_real)}\n")

    # Scenario 4: Wrong-format "Real" prediction (provided Fake evidence)
    pred_real_wrong_format = """<think>The motion seems natural.
    But I found <type>Camera Motion Inconsistency</type> in <t>[3.89, 4.53]</t> at <bbox>[0.0, 0.0, 0.0, 0.0]</bbox>.
    </think>
    <answer>Real</answer>"""
    gt_real = "Real"
    # Expected (V3): answer is correct, but format is wrong
    # f_reward = 0.0 (because answer is Real, but it should not provide Fake evidence)
    # a_reward = 0.0 (because no <t>...</t> evidence was found)
    # Total: 0.5 * 0.0 + 0.5 * 0.0 = 0.0
    print(f"--- Scenario 4: Wrong-format Real (provided Fake evidence) ---")
    print(f"Format: {format_reward_aigc(pred_real_wrong_format)}")
    print(f"Acc: {acc_reward_aigc(pred_real_wrong_format, gt_real)}")
    print(f"Total: {compute_score_aigc(pred_real_wrong_format, gt_real)}\n")

    # Scenario 5: Wrong-format "Fake" prediction (no evidence provided)
    pred_fake_no_evidence = """<think>I think this is fake, but I am not sure why.</think>
    <answer>Fake</answer>"""
    gt_fake = "Fake"
    # Expected (V3): answer is correct, but format is wrong
    # f_reward = 0.0 (because answer is Fake, but no <type> evidence was provided)
    # a_reward = 0.0 (because log1p(0)=0)
    # Total: 0.5 * 0.0 + 0.5 * 0.0 = 0.0
    print(f"--- Scenario 5: Wrong-format Fake (no evidence) ---")
    print(f"Format: {format_reward_aigc(pred_fake_no_evidence)}")
    print(f"Acc: {acc_reward_aigc(pred_fake_no_evidence, gt_fake)}")
    print(f"Total: {compute_score_aigc(pred_fake_no_evidence, gt_fake)}\n")