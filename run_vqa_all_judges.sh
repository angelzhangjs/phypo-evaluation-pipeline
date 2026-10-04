#!/usr/bin/env bash
set -euo pipefail

CONDA_SH="${CONDA_SH:-/root/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-pbench}"
if [[ ! -f "${CONDA_SH}" ]]; then
  echo "ERROR: conda initialization script not found: ${CONDA_SH}" >&2
  exit 1
fi
# shellcheck source=/dev/null
source "${CONDA_SH}"
# cuda-nvcc's activate script expands unset NVCC_PREPEND_FLAGS / CFLAGS.
set +u
conda activate "${CONDA_ENV}"
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VIDEOS_ROOT="${VIDEOS_ROOT:-/root/autodl-tmp/angelz/videos/basic_physics}"
PROMPT_FILE="${PROMPT_FILE:-${VIDEOS_ROOT}/wan14b/phypo/prompts.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/ab_results/all_judges}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.65}"
MAX_FRAMES="${MAX_FRAMES:-8}"
API_REQUEST_INTERVAL="${API_REQUEST_INTERVAL:-2}"
API_MAX_RETRIES="${API_MAX_RETRIES:-12}"
API_RETRY_BASE_DELAY="${API_RETRY_BASE_DELAY:-15}"
API_MAX_COMPLETION_TOKENS="${API_MAX_COMPLETION_TOKENS:-1024}"
REASONING_EFFORT="${REASONING_EFFORT:-low}"

GPT_MODEL="${GPT_MODEL:-gpt-5.6-luna}"
SKIP_GPT="${SKIP_GPT:-0}"
GEMINI_PRO_MODEL="${GEMINI_PRO_MODEL:-gemini-3.1-pro-preview}"
GEMINI_FLASH_MODEL="${GEMINI_FLASH_MODEL:-gemini-3.6-flash}"
SKIP_GEMINI="${SKIP_GEMINI:-0}"
QWEN25_MODEL="${QWEN25_MODEL:-Qwen/Qwen2.5-VL-72B-Instruct}"
QWEN3_MODEL="${QWEN3_MODEL:-Qwen/Qwen3-VL-32B-Instruct}"
INTERNVL_MODEL="${INTERNVL_MODEL:-OpenGVLab/InternVL3_5-30B-A3B-HF}"
INTERNVL8_MODEL="${INTERNVL8_MODEL:-OpenGVLab/InternVL3_5-8B-HF}"
MOLMO_MODEL="${MOLMO_MODEL:-allenai/Molmo-72B-0924}"

if [[ ! -d "${VIDEOS_ROOT}" ]]; then
  echo "ERROR: videos root not found: ${VIDEOS_ROOT}" >&2
  exit 1
fi
if [[ ! -f "${PROMPT_FILE}" ]]; then
  echo "ERROR: prompt file not found: ${PROMPT_FILE}" >&2
  exit 1
fi
if [[ "${SKIP_GPT}" != "1" && -n "${GPT_MODEL}" && -z "${OPENAI_API_KEY:-}" ]]; then
  echo "ERROR: set OPENAI_API_KEY when GPT_MODEL is configured" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"
cd "${REPO_ROOT}"

run_judge() {
  local output_name="$1"
  local backend="$2"
  local model_name="$3"
  local model_max_len="${4:-${MAX_MODEL_LEN}}"
  local model_max_frames="${5:-${MAX_FRAMES}}"
  local local_max_tokens="${6:-}"

  echo "RUNNING JUDGE: ${model_name}"
  local extra_args=()
  if [[ -n "${local_max_tokens}" ]]; then
    extra_args+=(--local_max_tokens "${local_max_tokens}")
  fi
  python evaluate_vqa.py \
    --videos_root "${VIDEOS_ROOT}" \
    --prompt_file "${PROMPT_FILE}" \
    --model_name "${model_name}" \
    --backend "${backend}" \
    --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}" \
    --max_model_len "${model_max_len}" \
    --max_frames "${model_max_frames}" \
    --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
    --output_dir "${OUTPUT_ROOT}/${output_name}" \
    --enable_missing_videos \
    --resume \
    --api_request_interval "${API_REQUEST_INTERVAL}" \
    --api_max_retries "${API_MAX_RETRIES}" \
    --api_retry_base_delay "${API_RETRY_BASE_DELAY}" \
    --api_max_completion_tokens "${API_MAX_COMPLETION_TOKENS}" \
    --reasoning_effort "${REASONING_EFFORT}" \
    "${extra_args[@]}"
}

# The API judge is optional because model availability depends on the configured
# provider. Set GPT_MODEL to an identifier that the endpoint actually exposes.
if [[ "${SKIP_GPT}" == "1" ]]; then
  echo "SKIPPING GPT judge: SKIP_GPT=1"
elif [[ -n "${GPT_MODEL}" ]]; then
  if ! run_judge "gpt_api" "openai" "${GPT_MODEL}"; then
    echo "WARNING: GPT judge failed; continuing with Qwen judges" >&2
  fi
else
  echo "SKIPPING GPT judge: GPT_MODEL is not configured"
fi

# Gemini uses Google's OpenAI-compatible endpoint and a Google AI Studio key.
if [[ "${SKIP_GEMINI}" == "1" ]]; then
  echo "SKIPPING Gemini judges: SKIP_GEMINI=1"
elif [[ -z "${GEMINI_API_KEY:-}" ]]; then
  echo "WARNING: GEMINI_API_KEY is not configured; skipping Gemini judges" >&2
else
  if ! run_judge "gemini_2_5_pro" "gemini" "${GEMINI_PRO_MODEL}"; then
    echo "WARNING: Gemini 2.5 Pro judge failed; continuing" >&2
  fi
  if ! run_judge "gemini_2_5_flash" "gemini" "${GEMINI_FLASH_MODEL}"; then
    echo "WARNING: Gemini 2.5 Flash judge failed; continuing" >&2
  fi
fi

# Local/Hugging Face judges run in separate processes so GPU memory is released
# before the next model is loaded.
run_judge "qwen2_5_vl_72b" "qwen" "${QWEN25_MODEL}"
run_judge "qwen3_vl_32b" "qwen" "${QWEN3_MODEL}"
run_judge "internvl3_5_30b_a3b" "internvl" "${INTERNVL_MODEL}"
run_judge "internvl3_5_8b" "internvl" "${INTERNVL8_MODEL}"
run_judge "molmo_72b_0924" "molmo" "${MOLMO_MODEL}" 4096 3 128

echo "All judge results saved under: ${OUTPUT_ROOT}"
