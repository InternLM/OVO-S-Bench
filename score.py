#!/usr/bin/env python3
"""
Scoring script for OVO-S evaluation results.

Usage:
    python score.py --result results/gpt-4o/level_1.json
    python score.py --result results/gpt-4o/level_1.json --output scores/gpt-4o_level_1_scores.json
"""

import os
import sys
import json
import re
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Tuple


def parse_args():
    parser = argparse.ArgumentParser(description="Score OVO-S evaluation results")

    parser.add_argument(
        "--result",
        type=str,
        required=True,
        help="Path to result JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for scores (default: {result_dir}/scores_{result_name}.json)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed results"
    )

    return parser.parse_args()


def extract_answer(response: str) -> str:
    """
    Extract the answer letter from model response.

    Order of attempts (later attempts only fire if earlier ones fail):
      1. Strip ``<think>...</think>`` block — only the post-think tail is parsed.
      2. Tail-of-response ``Answer: X`` (also catches rescue patches that
         append "Answer: X" to truncated thinking traces).
      3. Cosmos-style ``<answer>X</answer>`` tags.
      4. Bare single letter at the very end (e.g. ``...therefore B``).
      5. GLM-style ``<|begin_of_box|>X<|end_of_box|>``.
      6. Single letter at the start of the (stripped) response.
      7. ``answer/choice/option(s): X`` *anywhere* — note: the keyword must be
         followed by at least one separator (``[:\\s]+``); plain ``options`` as
         part of free-flowing prose used to wrongly capture the trailing ``s``
         and resolve to ``S``.
      8. Single letter in parens / brackets.

    Returns the extracted letter or an empty string. The valid letter range is
    restricted to ``A-E`` (the bench uses at most 5 options).

    Args:
        response: Model response string

    Returns:
        Extracted answer letter (A-E) or empty string
    """
    if not response:
        return ""

    response = response.strip()

    # Strip thinking blocks (Qwen3.5 / DeepSeek-R1 / InternVL-thinking / GLM
    # all use the same `</think>` delimiter at the close of reasoning).
    if "</think>" in response:
        response = response.split("</think>")[-1].strip()

    tail = response[-300:]

    # (2) Tail-of-response "Answer: X" — catches both natural ``Answer: X``
    # tails and rescued thinking outputs that have ``\n\nAnswer: X`` appended.
    m = re.search(r'(?:answer|final\s+answer|final)[:\s]+([A-E])\b', tail, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # (3) Cosmos-Reason1 official prompt asks for <answer>...</answer>.
    m = re.search(r'<answer>\s*([A-E])\b', tail, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # (4) Bare single letter at the very end (e.g. "...therefore B" / "...B.")
    m = re.search(r'\b([A-E])\b\s*\.?\s*$', tail, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # (5) GLM-style boxed answer
    m = re.search(r'<\|begin_of_box\|>\s*([A-E])', tail, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # (6) Single letter at the start of the (stripped) response
    m = re.match(r'^([A-E])\b', response.upper())
    if m:
        return m.group(1)

    # (7) "answer/choice/option(s): X" anywhere in the response. Require a real
    # separator (``[:\s]+``) so ``options`` as part of "Let's look at the
    # options: 3m, 5m, ..." no longer captures the trailing ``s`` as the letter.
    m = re.search(r'(?:answer|choice|option)s?[:\s]+([A-E])\b', response, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # (8) Single letter in parens/brackets, e.g. "(B)" or "[C]"
    m = re.search(r'[\(\[]\s*([A-E])\s*[\)\]]', response, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    return ""


def check_answer(response: str, ground_truth) -> bool:
    """
    Check if the response matches the ground truth answer.

    Args:
        response: Model response string
        ground_truth: Correct answer (string or list of strings)

    Returns:
        True if response matches ground truth
    """
    extracted = extract_answer(response)
    if not extracted:
        return False

    # Handle both string and list ground truth
    if isinstance(ground_truth, list):
        gt_normalized = [str(gt).strip().upper() for gt in ground_truth]
    else:
        gt_normalized = [str(ground_truth).strip().upper()]

    return extracted in gt_normalized


def calculate_scores(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate scores from evaluation results.

    Args:
        results: List of result dictionaries

    Returns:
        Dictionary with scores by task and overall
    """
    # Group by task subcategory
    by_task = defaultdict(list)
    by_main_category = defaultdict(list)

    for result in results:
        if "error" in result and result.get("response") is None:
            continue

        task_sub = result.get("task_subcategory", "unknown")
        main_cat = task_sub.split(".")[0] if "." in task_sub else "unknown"

        response = result.get("response", "")
        ground_truth = result.get("ground_truth", [])

        is_correct = check_answer(response, ground_truth)

        by_task[task_sub].append({
            "id": result.get("query_id", result.get("id")),
            "correct": is_correct,
            "response": response,
            "ground_truth": ground_truth,
            "extracted": extract_answer(response)
        })

        by_main_category[main_cat].append(is_correct)

    # Calculate per-task scores
    task_scores = {}
    for task, items in sorted(by_task.items()):
        correct = sum(1 for item in items if item["correct"])
        total = len(items)
        accuracy = correct / total if total > 0 else 0

        task_scores[task] = {
            "correct": correct,
            "total": total,
            "accuracy": accuracy,
            "details": items
        }

    # Calculate per-main-category scores
    category_scores = {}
    for cat, correct_list in sorted(by_main_category.items()):
        correct = sum(correct_list)
        total = len(correct_list)
        accuracy = correct / total if total > 0 else 0

        category_scores[cat] = {
            "correct": correct,
            "total": total,
            "accuracy": accuracy
        }

    # Calculate overall score
    all_correct = sum(s["correct"] for s in task_scores.values())
    all_total = sum(s["total"] for s in task_scores.values())
    overall_accuracy = all_correct / all_total if all_total > 0 else 0

    return {
        "overall": {
            "correct": all_correct,
            "total": all_total,
            "accuracy": overall_accuracy
        },
        "by_main_category": category_scores,
        "by_task": task_scores
    }


def print_scores(scores: Dict[str, Any], verbose: bool = False):
    """Print scores in a formatted way."""
    print("\n" + "=" * 60)
    print("OVO-S Evaluation Scores")
    print("=" * 60)

    # Overall
    overall = scores["overall"]
    print(f"\nOverall: {overall['correct']}/{overall['total']} = {overall['accuracy']:.2%}")

    # By main category
    print("\nBy Main Category:")
    print("-" * 40)
    for cat, data in sorted(scores["by_main_category"].items()):
        print(f"  {cat}: {data['correct']}/{data['total']} = {data['accuracy']:.2%}")

    # By task
    print("\nBy Task Subcategory:")
    print("-" * 40)
    for task, data in sorted(scores["by_task"].items()):
        print(f"  {task}: {data['correct']}/{data['total']} = {data['accuracy']:.2%}")

    if verbose:
        print("\nDetailed Results:")
        print("-" * 40)
        for task, data in sorted(scores["by_task"].items()):
            print(f"\n{task}:")
            for item in data["details"]:
                status = "✓" if item["correct"] else "✗"
                print(f"  {status} {item['id']}: {item['extracted']} (GT: {item['ground_truth']})")


def main():
    args = parse_args()

    # Load results
    result_path = Path(args.result)
    if not result_path.exists():
        print(f"Error: Result file not found: {result_path}")
        sys.exit(1)

    print(f"Loading results from: {result_path}")
    with open(result_path, "r") as f:
        data = json.load(f)

    results = data.get("results", [])
    print(f"Loaded {len(results)} results")

    # Calculate scores
    scores = calculate_scores(results)

    # Print scores
    print_scores(scores, args.verbose)

    # Save scores
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = result_path.parent / f"scores_{result_path.stem}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare output (remove details for cleaner output)
    output_scores = {
        "source_file": str(result_path),
        "model": data.get("model", "unknown"),
        "overall": scores["overall"],
        "by_main_category": scores["by_main_category"],
        "by_task": {
            task: {k: v for k, v in data.items() if k != "details"}
            for task, data in scores["by_task"].items()
        }
    }

    with open(output_path, "w") as f:
        json.dump(output_scores, f, indent=2, ensure_ascii=False)

    print(f"\nScores saved to: {output_path}")


if __name__ == "__main__":
    main()
