#!/usr/bin/env python3
"""
Aggregate physical-ai-bench evaluate.py outputs into a compact quality-metrics row.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


KEY_MAP = {
    "SC": "subject_consistency",
    "BC": "background_consistency",
    "MS": "motion_smoothness",
    "AQ": "aesthetic_quality",
    "IQ": "imaging_quality",
    "OC": "overall_consistency",
    "IS": "i2v_subject",
    "IB": "i2v_background",
}


def _extract_scalar(value):
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, list) and len(value) > 0:
        first = value[0]
        if isinstance(first, (int, float)):
            return float(first)
        if isinstance(first, list) and len(first) > 0 and isinstance(first[0], (int, float)):
            return float(first[0])
    raise ValueError(f"Could not extract scalar metric from value of type {type(value)}: {value!r}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Aggregate evaluate.py quality metrics into a compact row")
    p.add_argument("--results-json", required=True, help="Path to results_*_eval_results.json")
    p.add_argument("--model-name", default="MODEL", help="Optional model name prefix for pretty printing")
    p.add_argument("--latex", action="store_true", help="Print LaTeX table row")
    return p


def main() -> None:
    args = build_parser().parse_args()
    results_path = Path(args.results_json).expanduser().resolve()
    data = json.loads(results_path.read_text())

    scores = {}
    for short, full in KEY_MAP.items():
        if full not in data:
            scores[short] = None
            continue
        scores[short] = _extract_scalar(data[full])

    valid_scores = [v for v in scores.values() if v is not None]
    avg = sum(valid_scores) / len(valid_scores) if valid_scores else None

    if args.latex:
        vals = []
        for short in ["SC", "BC", "MS", "AQ", "IQ", "OC", "IS", "IB"]:
            v = scores[short]
            vals.append("--" if v is None else f"{100.0 * v:.1f}")
        vals.append("--" if avg is None else f"{100.0 * avg:.1f}")
        print(f"{args.model_name} & " + " & ".join(vals) + r" \\")
        return

    print(f"Model: {args.model_name}")
    for short in ["SC", "BC", "MS", "AQ", "IQ", "OC", "IS", "IB"]:
        v = scores[short]
        print(f"{short}: {'N/A' if v is None else f'{v:.4f}'}")
    print(f"Avg: {'N/A' if avg is None else f'{avg:.4f}'}")


if __name__ == "__main__":
    main()
