#!/usr/bin/env python3
"""
Aggregate custom VQA evaluation outputs for the 10-category physics suite.

This script consumes the `vqa_detailed_results.json` written by `evaluate_vqa.py`
and produces a compact summary with:
- overall video-level accuracy
- overall question-level accuracy
- per-category averages for custom physics categories
- per-question-slot accuracy across the whole benchmark

It is intended for custom prompt suites whose video ids use category codes such as
`FA_001`, `PR_004`, `BO_010`, etc.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List


CATEGORY_CODE_TO_NAME = {
    "FA": "falling",
    "PR": "projectile",
    "SW": "swinging",
    "CP": "compression",
    "SP": "spinning",
    "SL": "sliding",
    "BO": "bouncing",
    "FL": "fluid_flowing",
    "FT": "floating",
    "SH": "splashing",
}

CATEGORY_ORDER = ["FA", "PR", "SW", "CP", "SP", "SL", "BO", "FL", "FT", "SH"]


def _load_json(path: Path):
    return json.loads(path.read_text())


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _question_accuracy(question_result: dict) -> float:
    if "normalized_score" in question_result:
        return float(question_result["normalized_score"])
    if "accuracy" in question_result:
        return float(question_result["accuracy"])
    if "score" in question_result and "max_score" in question_result and "min_score" in question_result:
        min_score = float(question_result["min_score"])
        max_score = float(question_result["max_score"])
        denom = max_score - min_score
        return (float(question_result["score"]) - min_score) / denom if denom > 0 else 0.0
    if "is_correct" in question_result:
        return 1.0 if question_result["is_correct"] else 0.0
    return 0.0


def _video_accuracy(video_result: dict) -> float:
    if "accuracy" in video_result:
        return float(video_result["accuracy"])
    question_scores = [_question_accuracy(q) for q in video_result.get("results", [])]
    return _mean(question_scores)


def _category_code_from_video_result(video_result: dict) -> str:
    video_id = str(video_result.get("video_id", ""))
    match = re.match(r"^([A-Za-z]{2})_\d+", video_id)
    if match:
        return match.group(1).upper()

    category_name = str(video_result.get("category", "")).strip()
    for code, name in CATEGORY_CODE_TO_NAME.items():
        if category_name == name:
            return code
    return "UNK"


def aggregate_results(detailed_results: List[dict]) -> dict:
    per_category: Dict[str, dict] = {}
    per_question_index: Dict[int, List[float]] = {}
    all_video_scores: List[float] = []
    all_question_scores: List[float] = []

    for video_result in detailed_results:
        code = _category_code_from_video_result(video_result)
        category_name = CATEGORY_CODE_TO_NAME.get(code, str(video_result.get("category", "unknown")))
        video_score = _video_accuracy(video_result)
        question_scores = []

        for idx, question_result in enumerate(video_result.get("results", []), start=1):
            score = _question_accuracy(question_result)
            question_scores.append(score)
            all_question_scores.append(score)
            per_question_index.setdefault(idx, []).append(score)

        entry = per_category.setdefault(
            code,
            {
                "category_code": code,
                "category_name": category_name,
                "video_scores": [],
                "question_scores": [],
                "video_ids": [],
            },
        )
        entry["video_scores"].append(video_score)
        entry["question_scores"].extend(question_scores)
        entry["video_ids"].append(str(video_result.get("video_id", "")))
        all_video_scores.append(video_score)

    by_category = []
    ordered_codes = [code for code in CATEGORY_ORDER if code in per_category]
    ordered_codes.extend(sorted(code for code in per_category if code not in CATEGORY_ORDER))
    for code in ordered_codes:
        entry = per_category[code]
        by_category.append(
            {
                "category_code": code,
                "category_name": entry["category_name"],
                "num_videos": len(entry["video_scores"]),
                "num_questions": len(entry["question_scores"]),
                "video_score": _mean(entry["video_scores"]),
                "question_score": _mean(entry["question_scores"]),
            }
        )

    by_question_index = []
    for idx in sorted(per_question_index):
        scores = per_question_index[idx]
        by_question_index.append(
            {
                "question_index": idx,
                "score": _mean(scores),
                "count": len(scores),
            }
        )

    return {
        "overall_video_score": _mean(all_video_scores),
        "overall_question_score": _mean(all_question_scores),
        "total_videos": len(all_video_scores),
        "total_questions": len(all_question_scores),
        "by_category": by_category,
        "by_question_index": by_question_index,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate custom VQA evaluation outputs")
    parser.add_argument(
        "--detailed-results",
        type=str,
        required=True,
        help="Path to vqa_detailed_results.json from evaluate_vqa.py",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="custom-model",
        help="Display name used in printed summaries / LaTeX row",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to write the aggregated JSON summary",
    )
    parser.add_argument(
        "--latex",
        action="store_true",
        help="Print a LaTeX table row with overall and per-category accuracies",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    detailed_results_path = Path(args.detailed_results).expanduser().resolve()
    detailed_results = _load_json(detailed_results_path)
    summary = aggregate_results(detailed_results)
    summary["model_name"] = args.model_name
    summary["source_file"] = str(detailed_results_path)

    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2))
        print(f"Wrote aggregated custom VQA summary to: {output_path}")

    if args.latex:
        category_lookup = {item["category_code"]: item["video_score"] for item in summary["by_category"]}
        row_values = [f"{summary['overall_video_score']:.4f}"]
        row_values.extend(f"{category_lookup.get(code, 0.0):.4f}" for code in CATEGORY_ORDER)
        print(f"{args.model_name} & " + " & ".join(row_values) + r" \\")
        return

    print("=" * 72)
    print("CUSTOM VQA AGGREGATION")
    print("=" * 72)
    print(f"Model: {args.model_name}")
    print(f"Detailed Results: {detailed_results_path}")
    print(f"Overall Video Score:    {summary['overall_video_score']:.4f}")
    print(f"Overall Question Score: {summary['overall_question_score']:.4f}")
    print(f"Total Videos:              {summary['total_videos']}")
    print(f"Total Questions:           {summary['total_questions']}")
    print("-" * 72)
    print("Per-Category Video Score")
    for item in summary["by_category"]:
        print(
            f"  {item['category_code']:<3} {item['category_name']:<15} "
            f"video={item['video_score']:.4f} "
            f"question={item['question_score']:.4f} "
            f"(videos={item['num_videos']}, questions={item['num_questions']})"
        )
    print("-" * 72)
    print("Per-Question-Slot Score")
    for item in summary["by_question_index"]:
        print(f"  q{item['question_index']}: {item['score']:.4f} (count={item['count']})")
    print("=" * 72)


if __name__ == "__main__":
    main()
