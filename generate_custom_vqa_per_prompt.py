#!/usr/bin/env python3
"""
Generate per-prompt custom VQA files directly from a plain-text prompt list.

This is useful for prompt suites such as `origin_grpo/newyear_physics_prompts_100.txt`,
where prompts are stored one per line and grouped by category in fixed-size blocks.
Each prompt receives prompt-specific scored questions, and one JSON file is
written per prompt using a category-coded `video_id` such as `FA_001`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from generate_custom_vqa import CATEGORY_CODE_MAP, SCORE_RUBRIC, _questions_for_prompt


CATEGORY_SEQUENCE: List[str] = [
    "falling",
    "projectile",
    "swinging",
    "compression",
    "spinning",
    "rolling",
    "bouncing",
    "fluid_flowing",
    "floating",
    "splashing",
]

CATEGORY_NAME_TO_CODE = {name: code for code, name in CATEGORY_CODE_MAP.items()}


def _read_prompts(path: Path) -> List[str]:
    prompts = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not prompts:
        raise ValueError(f"No prompts found in: {path}")
    return prompts


def _video_id_for_prompt(category: str, prompt_index_within_category: int) -> str:
    code = CATEGORY_NAME_TO_CODE[category]
    return f"{code}_{prompt_index_within_category:03d}"


def _make_qa(video_id: str, category: str, idx: int, question: str) -> dict:
    return {
        "uid": f"{video_id}_q{idx+1}",
        "question": question,
        "evaluation_mode": "score",
        "min_score": 0,
        "max_score": 5,
        "rubric": SCORE_RUBRIC,
        "task": category,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate per-prompt custom VQA files from a plain-text prompt file"
    )
    parser.add_argument(
        "--prompt-txt",
        type=str,
        required=True,
        help="Plain-text prompt file with one prompt per line",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to write per-prompt VQA JSON files",
    )
    parser.add_argument(
        "--category-block-size",
        type=int,
        default=10,
        help="Number of prompts per category block in the prompt file",
    )
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help="Also write prompt_manifest.json with video_id, prompt_en, and category",
    )
    parser.add_argument(
        "--question-set",
        type=str,
        default="custom_physics",
        choices=["custom_physics", "alignment_only"],
        help="Which prompt-specific question set to generate.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    prompt_txt = Path(args.prompt_txt).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    prompts = _read_prompts(prompt_txt)
    block_size = args.category_block_size
    manifest = []
    question_set_name = args.question_set

    for global_idx, prompt in enumerate(prompts):
        category_idx = global_idx // block_size
        if category_idx >= len(CATEGORY_SEQUENCE):
            raise ValueError(
                f"Prompt index {global_idx} exceeds configured category sequence. "
                f"Increase --category-block-size or extend CATEGORY_SEQUENCE."
            )

        category = CATEGORY_SEQUENCE[category_idx]
        prompt_idx_within_category = (global_idx % block_size) + 1
        video_id = _video_id_for_prompt(category, prompt_idx_within_category)
        questions, task_name = _questions_for_prompt(prompt, category, question_set_name)
        qa_data = [_make_qa(video_id, task_name, i, q) for i, q in enumerate(questions)]

        (output_dir / f"{video_id}.json").write_text(json.dumps(qa_data, indent=2))
        manifest.append(
            {
                "video_id": video_id,
                "prompt_en": prompt,
                "category": category,
                "source_index": global_idx + 1,
            }
        )

    if args.write_manifest:
        (output_dir / "prompt_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"✅ Wrote {len(prompts)} per-prompt VQA files to: {output_dir}")
    if question_set_name == "alignment_only":
        print(f"Each prompt received {len(questions)} prompt-specific alignment 0-5 scored questions.")
    else:
        print(f"Each prompt received {len(questions)} prompt-specific physics 0-5 scored questions.")


if __name__ == "__main__":
    main()
