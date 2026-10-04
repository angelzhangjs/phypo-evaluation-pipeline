import argparse
import hashlib
import logging
import os
import re
import time
from collections import Counter
from pathlib import Path

from PIL import Image

from pbench.utils import load_json, save_json
from pbench.vqa_evaluation import (
    OpenAIEvaluator,
    QuotaExceededError,
    QwenVLEvaluator,
    classify_openai_error,
    prepare_multimodal_input,
    retry_delay_seconds,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_QUESTION_TEMPLATE = (
    "Which video generates a more natural physical {category} behavior while also aligning "
    "closely with the text description?"
)

CATEGORY_CODE_MAP = {
    "FA": "falling",
    "PR": "projectile",
    "SW": "swinging",
    "CP": "compression",
    "SP": "spinning",
    "SL": "sliding",
    "BO": "bouncing",
    "FL": "fluid flowing",
    "FT": "floating",
    "SH": "splashing",
}


PAIRWISE_SYSTEM_PROMPT = (
    "You are a careful video evaluator. Compare Video A and Video B against the supplied "
    "text description. Judge both physical naturalness and text alignment. You must select "
    "the better video even when the difference is small; ties are not allowed. Respond with "
    "exactly one token: A or B."
)


def _get_prompt(item):
    if isinstance(item, str):
        return item
    if not isinstance(item, dict):
        return None
    for key in ("prompt", "caption", "text", "description"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def load_prompt_map(prompt_file):
    """Load either a list of metadata objects or a video_id-keyed JSON object."""
    data = load_json(prompt_file)
    prompt_map = {}

    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            video_id = item.get("video_id") or item.get("id")
            prompt = _get_prompt(item)
            if video_id and prompt:
                prompt_map[str(video_id)] = prompt
    elif isinstance(data, dict):
        for video_id, item in data.items():
            prompt = _get_prompt(item)
            if prompt:
                prompt_map[str(video_id)] = prompt
    else:
        raise ValueError("Prompt file must contain a JSON list or object")

    if not prompt_map:
        raise ValueError(
            "No prompts found. Expected video_id/id plus prompt/caption/text/description fields."
        )
    return prompt_map


def index_videos(video_dir, id_prefix=None):
    """Map video_id to seed/path entries for direct or nested MP4 files."""
    videos = {}
    for path in sorted(Path(video_dir).rglob("*.mp4")):
        if ".cache" in path.parts or not path.is_file():
            continue
        stem = path.stem
        if "__" in stem:
            video_id, seed = stem.rsplit("__", 1)
        else:
            video_id, seed = stem, "default"
        index_id = f"{id_prefix}::{video_id}" if id_prefix else video_id
        videos.setdefault(index_id, []).append({"seed": seed, "path": str(path)})
    return videos


def pair_seed_videos(video_id, baseline_entries, phypo_entries):
    baseline_by_seed = {item["seed"]: item["path"] for item in baseline_entries}
    phypo_by_seed = {item["seed"]: item["path"] for item in phypo_entries}
    common_seeds = sorted(set(baseline_by_seed) & set(phypo_by_seed))

    if common_seeds:
        return [
            {
                "video_id": video_id,
                "seed": seed,
                "baseline_path": baseline_by_seed[seed],
                "phypo_path": phypo_by_seed[seed],
            }
            for seed in common_seeds
        ]

    if len(baseline_entries) == 1 and len(phypo_entries) == 1:
        return [{
            "video_id": video_id,
            "seed": f'{baseline_entries[0]["seed"]}|{phypo_entries[0]["seed"]}',
            "baseline_path": baseline_entries[0]["path"],
            "phypo_path": phypo_entries[0]["path"],
        }]

    logger.warning("No matching seeds for %s; skipping", video_id)
    return []


def should_swap_position(video_id, seed, position_seed):
    digest = hashlib.sha256(f"{position_seed}:{video_id}:{seed}".encode()).digest()
    return bool(digest[0] & 1)


def category_for_video_id(video_id):
    code = video_id.split("_", 1)[0].upper()
    return CATEGORY_CODE_MAP.get(code, "general")


def question_for_video_id(video_id, question_override=None):
    if question_override:
        return question_override
    return DEFAULT_QUESTION_TEMPLATE.format(category=category_for_video_id(video_id))


def build_user_prompt(text_description, question):
    return (
        f"Text description:\n{text_description}\n\n"
        f"Question:\n{question}\n\n"
        "Compare the complete motion in both videos. Balance physical realism and alignment "
        "to the description. You must pick one video; a tie is not allowed. Answer exactly A or B."
    )


def parse_choice(response):
    text = (response or "").strip().upper()
    if text in {"A", "B"}:
        return text

    first_line = text.splitlines()[0].strip() if text else ""
    if first_line in {"A", "B"}:
        return first_line

    final_match = re.search(r"(?:FINAL|ANSWER|CHOICE)\s*[:=-]\s*(A|B)\b", text)
    if final_match:
        return final_match.group(1)

    # Ignore "Video A" / "Video B" labels in reasoning; keep a standalone A/B.
    tokens = []
    for match in re.finditer(r"\b(?:A|B)\b", text):
        prefix = text[max(0, match.start() - 8):match.start()]
        if re.search(r"VIDEO\s*$", prefix):
            continue
        tokens.append(match.group(0))
    return tokens[-1] if tokens else None


def result_has_parseable_choice(result):
    if result.get("winner") in {None, "invalid"}:
        return False
    pairs = result.get("pairs") or []
    if not pairs:
        return False
    return all(parse_choice(pair.get("raw_response")) in {"A", "B"} for pair in pairs)


def create_evaluator(args):
    if args.backend == "auto":
        model_name_lower = args.model_name.lower()
        if model_name_lower.startswith(("gpt-", "chatgpt-", "o1-", "o3-")):
            backend = "openai"
        elif model_name_lower.startswith("gemini-"):
            backend = "gemini"
        elif "internvl" in model_name_lower:
            backend = "internvl"
        elif "molmo" in model_name_lower:
            backend = "molmo"
        else:
            backend = "qwen"
    else:
        backend = args.backend

    if backend in {"openai", "gemini"}:
        api_key = None
        base_url = None
        if backend == "gemini":
            api_key = os.getenv("GEMINI_API_KEY")
            base_url = os.getenv(
                "GEMINI_BASE_URL",
                "https://generativelanguage.googleapis.com/v1beta/openai/",
            )
            if not api_key:
                raise ValueError("GEMINI_API_KEY must be set for Gemini models")
        evaluator = OpenAIEvaluator(
            model_name=args.model_name,
            max_frames_num=args.max_frames,
            max_retries=args.api_max_retries,
            reasoning_effort=args.reasoning_effort,
            api_key=api_key,
            base_url=base_url,
            retry_base_delay=args.api_retry_base_delay,
        )
        evaluator.max_completion_tokens = args.api_max_completion_tokens
    elif backend in {"internvl", "molmo"}:
        evaluator = QwenVLEvaluator(
            model_name=args.model_name,
            device=args.device,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_tokens=args.local_max_tokens,
            multimodal_limits={"image": 2 * args.max_frames},
            require_qwen_vl_utils=False,
            processor_kwargs={},
        )
    else:
        evaluator = QwenVLEvaluator(
            model_name=args.model_name,
            device=args.device,
            tensor_parallel_size=args.tensor_parallel_size,
            max_videos_per_prompt=2,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_tokens=args.local_max_tokens,
        )
    evaluator.load_model()
    return evaluator, backend


def judge_with_qwen(evaluator, video_a, video_b, text_prompt):
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": PAIRWISE_SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text_prompt},
                {"type": "text", "text": "VIDEO A:"},
                {
                    "type": "video",
                    "video": video_a,
                    "fps": 2.0,
                    "max_pixels": 768 * 28 * 28,
                },
                {"type": "text", "text": "VIDEO B:"},
                {
                    "type": "video",
                    "video": video_b,
                    "fps": 2.0,
                    "max_pixels": 768 * 28 * 28,
                },
            ],
        },
    ]
    model_input = prepare_multimodal_input(messages, evaluator.processor)
    output = evaluator.model.generate(
        [model_input], sampling_params=evaluator.sampling_params, use_tqdm=False
    )[0]
    return output.outputs[0].text.strip()


def judge_with_internvl(evaluator, video_a, video_b, text_prompt, max_frames):
    frames_a = evaluator.extract_video_frames(video_a, max_frames=max_frames)
    frames_b = evaluator.extract_video_frames(video_b, max_frames=max_frames)
    if not frames_a or not frames_b:
        raise RuntimeError(f"Could not decode both videos: {video_a}, {video_b}")

    def prepare_frame(frame):
        frame = frame.copy()
        frame.thumbnail((448, 448), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (448, 448), color=(0, 0, 0))
        offset = ((448 - frame.width) // 2, (448 - frame.height) // 2)
        canvas.paste(frame, offset)
        return canvas

    frames_a = [prepare_frame(frame) for frame in frames_a]
    frames_b = [prepare_frame(frame) for frame in frames_b]
    content = [
        {"type": "text", "text": text_prompt},
        {"type": "text", "text": "VIDEO A frames in chronological order:"},
    ]
    content.extend({"type": "image"} for _ in frames_a)
    content.append({"type": "text", "text": "VIDEO B frames in chronological order:"})
    content.extend({"type": "image"} for _ in frames_b)
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": PAIRWISE_SYSTEM_PROMPT}],
        },
        {"role": "user", "content": content},
    ]
    prompt = evaluator.processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_input = {
        "prompt": prompt,
        "multi_modal_data": {"image": frames_a + frames_b},
    }
    output = evaluator.model.generate(
        [model_input], sampling_params=evaluator.sampling_params, use_tqdm=False
    )[0]
    return output.outputs[0].text.strip()


def judge_with_molmo(evaluator, video_a, video_b, text_prompt, max_frames):
    frames_a = evaluator.extract_video_frames(video_a, max_frames=max_frames)
    frames_b = evaluator.extract_video_frames(video_b, max_frames=max_frames)
    if not frames_a or not frames_b:
        raise RuntimeError(f"Could not decode both videos: {video_a}, {video_b}")

    def prepare_frame(frame):
        frame = frame.copy()
        frame.thumbnail((448, 448), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (448, 448), color=(0, 0, 0))
        offset = ((448 - frame.width) // 2, (448 - frame.height) // 2)
        canvas.paste(frame, offset)
        return canvas

    frames_a = [prepare_frame(frame) for frame in frames_a]
    frames_b = [prepare_frame(frame) for frame in frames_b]
    prompt = (
        f"{PAIRWISE_SYSTEM_PROMPT}\n\n{text_prompt}\n\n"
        f"The first {len(frames_a)} images are chronological frames from Video A. "
        f"The next {len(frames_b)} images are chronological frames from Video B. "
        "Start your reply with the single letter A or B on the first line. "
        "Do not write analysis before that letter."
    )
    model_input = {
        "prompt": prompt,
        "multi_modal_data": {"image": frames_a + frames_b},
    }
    output = evaluator.model.generate(
        [model_input], sampling_params=evaluator.sampling_params, use_tqdm=False
    )[0]
    return output.outputs[0].text.strip()


def judge_with_openai(evaluator, video_a, video_b, text_prompt):
    frames_a = evaluator.encode_video(video_a, evaluator.max_frames_num)
    frames_b = evaluator.encode_video(video_b, evaluator.max_frames_num)
    if not frames_a or not frames_b:
        raise RuntimeError(f"Could not decode both videos: {video_a}, {video_b}")

    media_type = getattr(evaluator, "image_media_type", "image/jpeg")
    content = [{"type": "text", "text": text_prompt}, {"type": "text", "text": "VIDEO A frames:"}]
    content.extend(
        {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{frame}"}}
        for frame in frames_a
    )
    content.append({"type": "text", "text": "VIDEO B frames:"})
    content.extend(
        {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{frame}"}}
        for frame in frames_b
    )

    messages = [
        {"role": "system", "content": PAIRWISE_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]
    max_completion_tokens = getattr(evaluator, "max_completion_tokens", 1024)
    for attempt in range(evaluator.max_retries):
        try:
            params = {"model": evaluator.base_model_name, "messages": messages}
            base_model_name_lower = evaluator.base_model_name.lower()
            is_reasoning_model = any(
                name in base_model_name_lower for name in ("o1", "o3", "gpt-5")
            )
            if is_reasoning_model:
                # GPT-5 / o-series hide chain-of-thought inside completion tokens.
                # 128 was too small, so the visible A/B answer was often empty.
                params["max_completion_tokens"] = (
                    max_completion_tokens if attempt == 0 else max(max_completion_tokens, 2048)
                )
                effort = evaluator.reasoning_effort
                if attempt > 0:
                    effort = "low"
                if effort is not None:
                    params["reasoning_effort"] = effort
            else:
                params.update(max_tokens=max_completion_tokens, temperature=0.0)
            response = evaluator.client.chat.completions.create(**params)
            choice = response.choices[0]
            text = (choice.message.content or "").strip()
            if text:
                return text
            finish_reason = getattr(choice, "finish_reason", None)
            if attempt == evaluator.max_retries - 1:
                logger.warning(
                    "API judge returned empty content after %s attempts (finish_reason=%s)",
                    evaluator.max_retries,
                    finish_reason,
                )
                return text
            delay = getattr(evaluator, "timeout", 10)
            logger.warning(
                "API judge returned empty content (finish_reason=%s); retrying %s/%s in %.1fs",
                finish_reason,
                attempt + 1,
                evaluator.max_retries,
                delay,
            )
            time.sleep(delay)
            continue
        except QuotaExceededError:
            raise
        except Exception as exc:
            error_kind = classify_openai_error(exc)
            if error_kind == "quota":
                raise QuotaExceededError(
                    "OpenAI returned 429 insufficient_quota. Add API credits at "
                    "https://platform.openai.com/settings/organization/billing "
                    "or point OPENAI_BASE_URL at a provider that has quota."
                ) from exc
            if error_kind == "not_found":
                raise
            if attempt == evaluator.max_retries - 1:
                raise
            delay = (
                retry_delay_seconds(exc, attempt, evaluator.retry_base_delay)
                if error_kind == "rate_limit"
                else evaluator.timeout
            )
            logger.warning(
                "API judge attempt %s/%s failed (%s); retrying in %.1fs",
                attempt + 1,
                evaluator.max_retries,
                error_kind,
                delay,
            )
            time.sleep(delay)


def comparison_result_key(model_group, video_id):
    return f"{model_group}::{video_id}" if model_group else video_id


def save_json_atomic(data, path):
    tmp_path = f"{path}.tmp"
    save_json(data, tmp_path)
    os.replace(tmp_path, path)


def build_summary(
    args,
    evaluator_backend,
    detailed_results,
    skipped_model_groups,
    missing_baseline,
    missing_phypo,
):
    counts = Counter(result["winner"] for result in detailed_results)
    valid_total = counts["baseline"] + counts["phypo"] + counts["tie"]
    decisive_total = counts["baseline"] + counts["phypo"]
    category_results = {}
    for category in sorted({result["category"] for result in detailed_results}):
        category_counts = Counter(
            result["winner"] for result in detailed_results if result["category"] == category
        )
        category_valid = (
            category_counts["baseline"] + category_counts["phypo"] + category_counts["tie"]
        )
        category_decisive = category_counts["baseline"] + category_counts["phypo"]
        category_results[category] = {
            "baseline_wins": category_counts["baseline"],
            "phypo_wins": category_counts["phypo"],
            "ties": category_counts["tie"],
            "invalid": category_counts["invalid"],
            "phypo_win_rate": (
                category_counts["phypo"] / category_valid if category_valid else 0.0
            ),
            "phypo_decisive_win_rate": (
                category_counts["phypo"] / category_decisive if category_decisive else 0.0
            ),
            "phypo_tie_adjusted_score": (
                (category_counts["phypo"] + 0.5 * category_counts["tie"]) / category_valid
                if category_valid else 0.0
            ),
        }

    model_results = {}
    for model_group in sorted(
        {result["model_group"] for result in detailed_results if result["model_group"]}
    ):
        model_counts = Counter(
            result["winner"]
            for result in detailed_results
            if result["model_group"] == model_group
        )
        model_valid = model_counts["baseline"] + model_counts["phypo"] + model_counts["tie"]
        model_decisive = model_counts["baseline"] + model_counts["phypo"]
        model_results[model_group] = {
            "evaluated_video_ids": sum(
                result["model_group"] == model_group for result in detailed_results
            ),
            "baseline_wins": model_counts["baseline"],
            "phypo_wins": model_counts["phypo"],
            "ties": model_counts["tie"],
            "invalid": model_counts["invalid"],
            "phypo_win_rate": model_counts["phypo"] / model_valid if model_valid else 0.0,
            "phypo_decisive_win_rate": (
                model_counts["phypo"] / model_decisive if model_decisive else 0.0
            ),
            "phypo_tie_adjusted_score": (
                (model_counts["phypo"] + 0.5 * model_counts["tie"]) / model_valid
                if model_valid else 0.0
            ),
        }

    return {
        "model_name": args.model_name,
        "question_template": args.question or DEFAULT_QUESTION_TEMPLATE,
        "evaluated_video_ids": len(detailed_results),
        "valid_video_ids": valid_total,
        "baseline_wins": counts["baseline"],
        "phypo_wins": counts["phypo"],
        "ties": counts["tie"],
        "invalid": counts["invalid"],
        "phypo_win_rate": counts["phypo"] / valid_total if valid_total else 0.0,
        "phypo_decisive_win_rate": counts["phypo"] / decisive_total if decisive_total else 0.0,
        "phypo_tie_adjusted_score": (
            (counts["phypo"] + 0.5 * counts["tie"]) / valid_total if valid_total else 0.0
        ),
        "category_results": category_results,
        "model_results": model_results,
        "skipped_model_groups": skipped_model_groups,
        "missing_baseline_ids": missing_baseline,
        "missing_phypo_ids": missing_phypo,
        "evaluation_params": {
            "backend": evaluator_backend,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "videos_root": args.videos_root,
            "baseline_dir": args.baseline_dir,
            "phypo_dir": args.phypo_dir,
            "prompt_file": args.prompt_file,
            "position_seed": args.position_seed,
        },
    }, counts, model_results


def print_summary(summary, counts, model_results, summary_path, details_path):
    print("\nPAIRWISE A/B SUMMARY")
    print("=" * 60)
    print(f"Baseline wins: {counts['baseline']}")
    print(f"PhyPO wins:    {counts['phypo']}")
    print(f"Ties:          {counts['tie']}")
    print(f"Invalid:       {counts['invalid']}")
    print(f"PhyPO win rate (wins / valid): {summary['phypo_win_rate']:.4f}")
    print(f"PhyPO decisive win rate:       {summary['phypo_decisive_win_rate']:.4f}")
    print(f"PhyPO tie-adjusted score:      {summary['phypo_tie_adjusted_score']:.4f}")
    for model_group, model_summary in model_results.items():
        print(
            f"{model_group}: PhyPO {model_summary['phypo_wins']} wins, "
            f"win rate {model_summary['phypo_win_rate']:.4f}, "
            f"ties {model_summary['ties']}"
        )
    print(f"Summary: {summary_path}")
    print(f"Details: {details_path}")


def aggregate_video_choice(pair_results):
    votes = Counter(result["winner"] for result in pair_results if result["winner"] != "invalid")
    phypo_votes = votes["phypo"]
    baseline_votes = votes["baseline"]
    if phypo_votes > baseline_votes:
        return "phypo"
    if baseline_votes > phypo_votes:
        return "baseline"
    if not votes:
        return "invalid"
    return next(
        result["winner"] for result in pair_results if result["winner"] != "invalid"
    )


def main():
    parser = argparse.ArgumentParser(description="Pairwise A/B evaluation of baseline and PhyPO videos")
    parser.add_argument("--videos_root",
                        help="Root containing <model>/baseline and <model>/phypo subfolders")
    parser.add_argument("--baseline_dir", help="Directory containing baseline MP4 files")
    parser.add_argument("--phypo_dir", help="Directory containing PhyPO MP4 files")
    parser.add_argument("--prompt_file", required=True, help="JSON containing video_id and text prompts")
    parser.add_argument("--model_name", default="~/models/Qwen2.5-VL-72B-Instruct")
    parser.add_argument(
        "--backend",
        choices=("auto", "openai", "gemini", "qwen", "internvl", "molmo"),
        default="auto",
        help="Model backend; auto detects OpenAI, Gemini, Qwen, InternVL, and Molmo",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--max_model_len", type=int, default=8192,
                        help="Maximum vLLM context length; 8192 is sufficient for two videos")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.65,
                        help="Fraction of each GPU available to vLLM")
    parser.add_argument("--output_dir", default="./ab_results")
    parser.add_argument("--question", default=None,
                        help="Optional fixed question; by default the category-specific question is used")
    parser.add_argument("--position_seed", type=int, default=42,
                        help="Seed used to deterministically randomize A/B positions")
    parser.add_argument("--max_frames", type=int, default=8,
                        help="Frames sampled per video for API, InternVL, and Molmo backends")
    parser.add_argument(
        "--reasoning_effort",
        default="low",
        help="Reasoning effort for GPT-5/o-series; low leaves tokens for the A/B answer",
    )
    parser.add_argument(
        "--api_max_completion_tokens",
        type=int,
        default=1024,
        help="Completion token budget for OpenAI/Gemini judges",
    )
    parser.add_argument(
        "--local_max_tokens",
        type=int,
        default=None,
        help="Completion token budget for local vLLM judges; Molmo defaults to 128, others to 16",
    )
    parser.add_argument("--enable_missing_videos", action="store_true")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from existing ab_detailed_results.json in --output_dir",
    )
    parser.add_argument(
        "--api_max_retries",
        type=int,
        default=12,
        help="Retries for OpenAI/Gemini 429 and transient errors",
    )
    parser.add_argument(
        "--api_retry_base_delay",
        type=float,
        default=15.0,
        help="Base seconds for exponential backoff on rate-limit 429s",
    )
    parser.add_argument(
        "--api_request_interval",
        type=float,
        default=2.0,
        help="Seconds to wait between OpenAI/Gemini requests",
    )
    args = parser.parse_args()
    if args.local_max_tokens is None:
        backend_hint = args.backend
        if backend_hint == "auto":
            model_name_lower = args.model_name.lower()
            if "molmo" in model_name_lower:
                backend_hint = "molmo"
        args.local_max_tokens = 128 if backend_hint == "molmo" else 16

    args.model_name = os.path.expanduser(args.model_name)
    if args.videos_root and (args.baseline_dir or args.phypo_dir):
        parser.error("Use either --videos_root or --baseline_dir with --phypo_dir, not both")
    if not args.videos_root and not (args.baseline_dir and args.phypo_dir):
        parser.error("Provide --videos_root, or both --baseline_dir and --phypo_dir")

    paths_to_validate = ["prompt_file"]
    paths_to_validate += ["videos_root"] if args.videos_root else ["baseline_dir", "phypo_dir"]
    for path_name in paths_to_validate:
        path = getattr(args, path_name)
        if not os.path.exists(path):
            parser.error(f"{path_name} does not exist: {path}")
    os.makedirs(args.output_dir, exist_ok=True)

    source_prompts = load_prompt_map(args.prompt_file)
    skipped_model_groups = []
    if args.videos_root:
        prompts = {}
        baseline_index = {}
        phypo_index = {}
        model_groups = []
        for model_dir in sorted(Path(args.videos_root).iterdir()):
            baseline_dir = model_dir / "baseline"
            phypo_dir = model_dir / "phypo"
            if not baseline_dir.is_dir() or not phypo_dir.is_dir():
                continue

            model_name = model_dir.name
            model_baseline = index_videos(baseline_dir, id_prefix=model_name)
            model_phypo = index_videos(phypo_dir, id_prefix=model_name)
            if not model_baseline or not model_phypo:
                skipped_model_groups.append(model_name)
                logger.warning(
                    "Skipping %s because baseline or phypo contains no MP4 files", model_name
                )
                continue

            model_groups.append(model_name)
            baseline_index.update(model_baseline)
            phypo_index.update(model_phypo)
            for video_id, prompt in source_prompts.items():
                prompts[f"{model_name}::{video_id}"] = prompt

        if not model_groups:
            raise ValueError(
                f"No non-empty <model>/baseline and <model>/phypo pairs found in {args.videos_root}"
            )
        logger.info("Discovered model groups: %s", ", ".join(model_groups))
    else:
        prompts = source_prompts
        baseline_index = index_videos(args.baseline_dir)
        phypo_index = index_videos(args.phypo_dir)

    expected_ids = set(prompts)
    missing_baseline = sorted(expected_ids - set(baseline_index))
    missing_phypo = sorted(expected_ids - set(phypo_index))
    if (missing_baseline or missing_phypo) and not args.enable_missing_videos:
        raise FileNotFoundError(
            f"Missing baseline IDs: {missing_baseline[:10]} "
            f"(total {len(missing_baseline)}); missing PhyPO IDs: {missing_phypo[:10]} "
            f"(total {len(missing_phypo)}). Use --enable_missing_videos to skip them."
        )

    shared_ids = sorted(expected_ids & set(baseline_index) & set(phypo_index))
    if not shared_ids:
        raise ValueError("No shared video_ids were found across prompts, baseline, and PhyPO")

    summary_path = os.path.join(args.output_dir, "ab_summary.json")
    details_path = os.path.join(args.output_dir, "ab_detailed_results.json")
    detailed_results = []
    completed_ids = set()
    if args.resume and os.path.exists(details_path):
        loaded = load_json(details_path)
        if isinstance(loaded, list):
            detailed_results = [
                result for result in loaded if result_has_parseable_choice(result)
            ]
            requeued = len(loaded) - len(detailed_results)
            completed_ids = {
                comparison_result_key(result.get("model_group"), result.get("video_id"))
                for result in detailed_results
                if result.get("video_id")
            }
            logger.info(
                "Resuming from %s completed video IDs (requeued %s invalid/unparseable)",
                len(completed_ids),
                requeued,
            )

    remaining_ids = [
        comparison_id
        for comparison_id in shared_ids
        if comparison_id not in completed_ids
    ]
    if not remaining_ids:
        logger.info("All %s video IDs already evaluated", len(shared_ids))
        summary, counts, model_results = build_summary(
            args, "resumed", detailed_results, skipped_model_groups, missing_baseline, missing_phypo
        )
        save_json_atomic(summary, summary_path)
        save_json_atomic(detailed_results, details_path)
        print_summary(summary, counts, model_results, summary_path, details_path)
        return

    evaluator, evaluator_backend = create_evaluator(args)

    def persist_progress():
        summary, _, _ = build_summary(
            args,
            evaluator_backend,
            detailed_results,
            skipped_model_groups,
            missing_baseline,
            missing_phypo,
        )
        save_json_atomic(detailed_results, details_path)
        save_json_atomic(summary, summary_path)
        return summary

    try:
        for comparison_id in remaining_ids:
            if "::" in comparison_id:
                model_group, video_id = comparison_id.split("::", 1)
            else:
                model_group, video_id = None, comparison_id
            category = category_for_video_id(video_id)
            question = question_for_video_id(video_id, args.question)
            pairs = pair_seed_videos(
                video_id, baseline_index[comparison_id], phypo_index[comparison_id]
            )
            pair_results = []
            for pair in pairs:
                swapped = should_swap_position(video_id, pair["seed"], args.position_seed)
                if swapped:
                    video_a, video_b = pair["phypo_path"], pair["baseline_path"]
                    label_for_position = {"A": "phypo", "B": "baseline"}
                else:
                    video_a, video_b = pair["baseline_path"], pair["phypo_path"]
                    label_for_position = {"A": "baseline", "B": "phypo"}

                text_prompt = build_user_prompt(prompts[comparison_id], question)
                if evaluator_backend in {"openai", "gemini"}:
                    response = judge_with_openai(evaluator, video_a, video_b, text_prompt)
                    if args.api_request_interval > 0:
                        time.sleep(args.api_request_interval)
                elif evaluator_backend == "internvl":
                    response = judge_with_internvl(
                        evaluator, video_a, video_b, text_prompt, args.max_frames
                    )
                elif evaluator_backend == "molmo":
                    response = judge_with_molmo(
                        evaluator, video_a, video_b, text_prompt, args.max_frames
                    )
                else:
                    response = judge_with_qwen(evaluator, video_a, video_b, text_prompt)
                choice = parse_choice(response)
                winner = label_for_position.get(choice, "invalid")
                pair_results.append({
                    **pair,
                    "video_a_label": label_for_position["A"],
                    "video_b_label": label_for_position["B"],
                    "raw_response": response,
                    "position_choice": choice,
                    "winner": winner,
                })
                logger.info(
                    "%s%s seed=%s winner=%s",
                    f"{model_group}/" if model_group else "",
                    video_id,
                    pair["seed"],
                    winner,
                )

            detailed_results.append({
                "model_group": model_group,
                "video_id": video_id,
                "category": category,
                "text_description": prompts[comparison_id],
                "question": question,
                "winner": aggregate_video_choice(pair_results),
                "pairs": pair_results,
            })
            persist_progress()
    except QuotaExceededError:
        persist_progress()
        raise
    except Exception:
        persist_progress()
        raise

    summary, counts, model_results = build_summary(
        args,
        evaluator_backend,
        detailed_results,
        skipped_model_groups,
        missing_baseline,
        missing_phypo,
    )
    save_json_atomic(summary, summary_path)
    save_json_atomic(detailed_results, details_path)
    print_summary(summary, counts, model_results, summary_path, details_path)


if __name__ == "__main__":
    main()
