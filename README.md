# PhyPO Evaluation Pipeline

[![Python Version](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Videos](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-basic__physics__videos-orange)](https://huggingface.co/datasets/angelic123/basic_physics_videos)
[![PAI-Bench](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-PAI--Bench--G-orange)](https://huggingface.co/datasets/shi-labs/physical-ai-bench-generation)

Evaluation code for comparing **baseline** text-to-video models against their
**PhyPO** (GRPO, physics-reward) fine-tuned versions. It has two parts:

1. **VLM judge (pairwise A/B)** — a vision-language model watches the baseline
   and PhyPO video for the same prompt and picks the one with more natural
   physics and better prompt alignment. Supported judges: ChatGPT / GPT-5.x and
   Gemini through their APIs, plus local Qwen2.5-VL, Qwen3-VL, InternVL3.5 and
   Molmo through vLLM.
2. **Video quality metrics** — VBench-style aesthetic, imaging, consistency and
   motion-smoothness scores.

## Table of Contents

- [Setup](#setup)
- [Get the Videos](#get-the-videos)
- [Add Your API Keys (ChatGPT and Gemini)](#add-your-api-keys-chatgpt-and-gemini)
- [Run the VLM Judge Evaluation](#run-the-vlm-judge-evaluation)
  - [ChatGPT judge](#chatgpt-judge)
  - [Gemini judge](#gemini-judge)
  - [Local open-source judges](#local-open-source-judges)
  - [All judges in one go](#all-judges-in-one-go)
  - [Outputs](#outputs)
  - [Options](#options)
- [Run Video Quality Evaluation](#run-video-quality-evaluation)
- [Troubleshooting](#troubleshooting)
- [Acknowledgments](#acknowledgments)

## Setup

```bash
git clone https://github.com/<github-user>/<repo-name>.git
cd <repo-name>

conda create -n pbench python=3.10 -y
conda activate pbench
pip install -e .
pip install --no-build-isolation "git+https://github.com/facebookresearch/detectron2.git"
```

`uv sync` also works if you prefer [uv](https://github.com/astral-sh/uv).

API judges (ChatGPT, Gemini) only need a network connection, not a GPU. Local
judges need GPUs: Qwen2.5-VL-72B and Molmo-72B are run with
`--tensor_parallel_size 4` on 4× 80 GB GPUs.

## Get the Videos

The baseline and PhyPO videos for six models (Cosmos 2B/14B, LTX-Video 2B/13B,
Wan 1.3B/14B) on the 500-prompt basic-physics set are on Hugging Face:

```bash
hf download angelic123/basic_physics_videos --repo-type dataset --local-dir videos/basic_physics
```

The layout is `videos/basic_physics/<model>/{baseline,phypo}/<video_id>.mp4`.
Each `phypo/` folder has a `prompts.json` that maps `video_id` to its prompt.
Video IDs start with a physics category code: `BO` bouncing, `CP` compression,
`FA` falling, `FL` fluid, `FT` floating, `PR` projectile, `SH` shattering,
`SL` sliding, `SP` spinning, `SW` swinging.

To evaluate your own videos, use the same layout: one folder per model, each
with `baseline/` and `phypo/` subfolders, and the same file name for the two
videos of a pair.

## Add Your API Keys (ChatGPT and Gemini)

You need an API key from each provider whose judge you want to run:

| Judge | Where to get a key | Environment variable |
|---|---|---|
| ChatGPT / GPT-5.x | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) (the account needs API billing credits; a ChatGPT Plus subscription is not enough) | `OPENAI_API_KEY` |
| Gemini 3.6 Flash / 3.1 Pro | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) | `GEMINI_API_KEY` |

Give the keys to the pipeline in **one** of these two ways.

**Option A — a `.env` file (recommended).** The code loads it automatically.

```bash
cp .env.example .env
# then edit .env:
#   OPENAI_API_KEY=sk-...
#   GEMINI_API_KEY=AIza...
```

**Option B — environment variables** for the current shell:

```bash
export OPENAI_API_KEY="sk-..."
export GEMINI_API_KEY="AIza..."
```

> **Never commit your keys.** `.env` is listed in `.gitignore`. Don't paste keys
> into scripts or the README.

Optional settings:

- `OPENAI_BASE_URL` — use an OpenAI-compatible proxy or gateway instead of
  `api.openai.com`. The model name you pass must exist on that endpoint.
- `GEMINI_BASE_URL` — defaults to Google's OpenAI-compatible endpoint,
  `https://generativelanguage.googleapis.com/v1beta/openai/`.

Check that a key works before starting a long run:

```bash
# OpenAI
python -c "from dotenv import load_dotenv; load_dotenv(); from openai import OpenAI; print([m.id for m in OpenAI().models.list()][:5])"

# Gemini
python -c "import os; from dotenv import load_dotenv; load_dotenv(); from openai import OpenAI; c = OpenAI(api_key=os.environ['GEMINI_API_KEY'], base_url='https://generativelanguage.googleapis.com/v1beta/openai/'); print([m.id for m in c.models.list()][:5])"
```

## Run the VLM Judge Evaluation

All judges use `evaluate_vqa.py`. For each prompt, the judge sees the baseline
and PhyPO videos as "Video A" and "Video B" in a randomized (but reproducible)
order and must pick one; ties are not allowed. The question names the physics
category, for example:

> Which video generates a more natural physical **falling** behavior while also
> aligning closely with the text description?

The backend is detected from `--model_name`: names starting with `gpt-` /
`o1-` / `o3-` use OpenAI, names starting with `gemini-` use Gemini, and anything
else is loaded locally. Override with `--backend {openai,gemini,qwen,internvl,molmo}`.

The examples below assume the videos were downloaded to `videos/basic_physics`.

### ChatGPT judge

```bash
python evaluate_vqa.py \
  --videos_root videos/basic_physics \
  --prompt_file videos/basic_physics/wan14b/phypo/prompts.json \
  --model_name gpt-5.6-luna \
  --output_dir ab_results/gpt \
  --enable_missing_videos
```

Use any vision-capable OpenAI model name your key has access to, such as
`gpt-4o`. For GPT-5 and o-series models, `--reasoning_effort low` (the default)
leaves enough of the completion-token budget for the answer.

### Gemini judge

```bash
python evaluate_vqa.py \
  --videos_root videos/basic_physics \
  --prompt_file videos/basic_physics/wan14b/phypo/prompts.json \
  --model_name gemini-3.6-flash \
  --output_dir ab_results/gemini_3_6_flash \
  --enable_missing_videos
```

For the larger model, use `--model_name gemini-3.1-pro-preview`.

### Local open-source judges

No API key is needed. Weights are downloaded from Hugging Face the first time.

```bash
python evaluate_vqa.py \
  --videos_root videos/basic_physics \
  --prompt_file videos/basic_physics/wan14b/phypo/prompts.json \
  --model_name Qwen/Qwen3-VL-32B-Instruct \
  --tensor_parallel_size 4 \
  --output_dir ab_results/qwen3_vl_32b \
  --enable_missing_videos
```

Other tested models: `Qwen/Qwen2.5-VL-72B-Instruct`,
`OpenGVLab/InternVL3_5-30B-A3B-HF`, `OpenGVLab/InternVL3_5-8B-HF` and
`allenai/Molmo-72B-0924`. For Molmo, add `--max_model_len 4096 --max_frames 3`.

### All judges in one go

`run_vqa_all_judges.sh` runs GPT, both Gemini models and the five local
judges one after another, saving each under `ab_results/all_judges/<judge>/`:

```bash
# keys from .env or exported in the shell
VIDEOS_ROOT=$PWD/videos/basic_physics bash run_vqa_all_judges.sh
```

The script activates a conda environment first. Point it at yours with
`CONDA_SH=/path/to/miniconda3/etc/profile.d/conda.sh CONDA_ENV=pbench`.

Useful overrides (all environment variables):

| Variable | Default | Meaning |
|---|---|---|
| `VIDEOS_ROOT` | (machine-specific path) | Root with `<model>/{baseline,phypo}` |
| `PROMPT_FILE` | `$VIDEOS_ROOT/wan14b/phypo/prompts.json` | Prompt map |
| `OUTPUT_ROOT` | `ab_results/all_judges` | Where results go |
| `GPT_MODEL` | `gpt-5.6-luna` | OpenAI judge model |
| `GEMINI_FLASH_MODEL` | `gemini-3.6-flash` | Gemini Flash judge model |
| `GEMINI_PRO_MODEL` | `gemini-3.1-pro-preview` | Gemini Pro judge model |
| `SKIP_GPT` / `SKIP_GEMINI` | `0` | Set to `1` to skip that API judge |
| `TENSOR_PARALLEL_SIZE` | `4` | GPUs per local judge |

The script stops with an error if `OPENAI_API_KEY` is not set, so set
`SKIP_GPT=1` if you only have a Gemini key. Gemini is skipped automatically when
`GEMINI_API_KEY` is missing. The Gemini results are saved in folders named
`gemini_2_5_pro/` and `gemini_2_5_flash/` regardless of which Gemini model
names you configure.

### Outputs

Each `--output_dir` gets:

- `ab_summary.json` — baseline wins, PhyPO wins, invalid answers, the overall
  **PhyPO win rate**, and breakdowns per physics category (`category_results`)
  and per video model (`model_results`).
- `ab_detailed_results.json` — for every pair: the A/B assignment, the judge's
  raw response and the winner.

Runs resume by default: re-running the same command skips pairs that already
have a valid answer in `ab_detailed_results.json`. Use `--no-resume` to start
over.

To evaluate a single model, pass the two folders directly instead of
`--videos_root`:

```bash
python evaluate_vqa.py \
  --baseline_dir videos/basic_physics/ltx13b/baseline \
  --phypo_dir videos/basic_physics/ltx13b/phypo \
  --prompt_file videos/basic_physics/ltx13b/phypo/prompts.json \
  --model_name gemini-3.6-flash \
  --output_dir ab_results/ltx13b_gemini
```

### Options

| Flag | Default | Meaning |
|---|---|---|
| `--max_frames` | `8` | Frames sampled per video for API, InternVL and Molmo judges |
| `--position_seed` | `42` | Seed for the A/B order |
| `--question` | category-specific | Fixed question to use instead |
| `--enable_missing_videos` | off | Skip pairs where a video is missing instead of failing |
| `--api_request_interval` | `2` | Seconds between API requests |
| `--api_max_retries` | `12` | Retries on 429 or transient API errors |
| `--api_retry_base_delay` | `15` | Base delay (s) for exponential backoff on 429 |
| `--api_max_completion_tokens` | `1024` | Completion budget for API judges |
| `--reasoning_effort` | `low` | Reasoning effort for GPT-5 / o-series |
| `--tensor_parallel_size` | `1` | GPUs for local vLLM judges |
| `--gpu_memory_utilization` | `0.65` | Fraction of each GPU vLLM may use |

## Run Video Quality Evaluation

VBench-style quality metrics, run on 8 GPUs:

```bash
python -m torch.distributed.run --standalone --nproc_per_node 8 evaluate.py \
  --mode custom_input \
  --prompt_file ${path_to_hf_dataset}/cosmos_predict2_bench_full_info.json \
  --custom_image_folder ${path_to_hf_dataset}/condition_image \
  --dimension aesthetic_quality background_consistency imaging_quality motion_smoothness overall_consistency subject_consistency i2v_background i2v_subject \
  --videos_path ${path_to_your_videos} \
  --output_path ./evaluation_results/
```

`run_quality_all.sh` runs this for every model folder. Results for the six
models are in `evaluation_results/`.

## Troubleshooting

- **`ValueError: GEMINI_API_KEY must be set`** or **`An API key must be
  configured`** — the key isn't visible to Python. Check `.env` is in the repo
  root or run `echo ${OPENAI_API_KEY:0:5}` in the same shell.
- **`429 insufficient_quota`** (OpenAI) — the API account has no credits. Add
  credits under *Settings → Billing* on platform.openai.com.
- **`429 Too Many Requests`** — rate limited. The script backs off and retries;
  raise `--api_request_interval` for lower-tier keys.
- **`404 model not found`** — the model name isn't available for your key or
  endpoint. List available models with the check command in
  [Add Your API Keys](#add-your-api-keys-chatgpt-and-gemini).
- **Many `invalid` answers** — the judge didn't answer "A" or "B". For
  reasoning models, raise `--api_max_completion_tokens` or keep
  `--reasoning_effort low`.

## Acknowledgments

Built on [PAI-Bench](https://github.com/SHI-Labs/physical-ai-bench) and
[VBench](https://github.com/Vchitect/VBench).
