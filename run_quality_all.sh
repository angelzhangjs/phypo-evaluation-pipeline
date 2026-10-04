#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASIC_ROOT="${BASIC_ROOT:-/root/autodl-tmp/angelz/videos/basic_physics}"
PAI_ROOT="${PAI_ROOT:-/root/autodl-tmp/angelz/videos/pai_bench}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/evaluation_results/quality_all}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
BASIC_PROMPT_FALLBACK="${BASIC_PROMPT_FALLBACK:-${BASIC_ROOT}/wan14b/phypo/prompts.json}"
export VBENCH_CACHE_DIR="${VBENCH_CACHE_DIR_OVERRIDE:-/root/autodl-tmp/angelz/.cache/vbench}"
MUSIQ_PATH="${VBENCH_CACHE_DIR}/pyiqa_model/musiq_spaq_ckpt-358bb6af.pth"
MUSIQ_URL="https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/musiq_spaq_ckpt-358bb6af.pth"

DIMENSIONS=(
  aesthetic_quality
  background_consistency
  imaging_quality
  motion_smoothness
  overall_consistency
  subject_consistency
)

for root in "${BASIC_ROOT}" "${PAI_ROOT}"; do
  if [[ ! -d "${root}" ]]; then
    echo "ERROR: videos root not found: ${root}" >&2
    exit 1
  fi
done

mkdir -p "${OUTPUT_ROOT}"

# Download and validate MUSIQ once before torchrun. Without this preflight,
# multiple ranks may race while creating the same checkpoint file.
mkdir -p "$(dirname "${MUSIQ_PATH}")"
if ! python - "${MUSIQ_PATH}" <<'PY'
import sys
import torch

try:
    torch.load(sys.argv[1], map_location="cpu", weights_only=True)
except Exception:
    raise SystemExit(1)
PY
then
  echo "MUSIQ checkpoint is missing or invalid; downloading a fresh copy."
  musiq_tmp="$(mktemp "$(dirname "${MUSIQ_PATH}")/.musiq.XXXXXX")"
  if ! wget --tries=20 --timeout=60 -O "${musiq_tmp}" "${MUSIQ_URL}"; then
    rm -f "${musiq_tmp}"
    echo "ERROR: failed to download MUSIQ checkpoint" >&2
    exit 1
  fi
  if ! python - "${musiq_tmp}" <<'PY'
import sys
import torch

try:
    torch.load(sys.argv[1], map_location="cpu", weights_only=True)
except Exception:
    raise SystemExit(1)
PY
  then
    rm -f "${musiq_tmp}"
    echo "ERROR: downloaded MUSIQ checkpoint is invalid" >&2
    exit 1
  fi
  mv -f "${musiq_tmp}" "${MUSIQ_PATH}"
fi

failures=0
evaluated=0
skipped=0

while IFS= read -r video_dir; do
  if [[ "${video_dir}" == "${BASIC_ROOT}"/* ]]; then
    suite="basic_physics"
    relative_dir="${video_dir#"${BASIC_ROOT}"/}"
    prompt_file="${video_dir}/prompts.json"
    if [[ ! -f "${prompt_file}" ]]; then
      prompt_file="$(find "$(dirname "${video_dir}")" -maxdepth 2 -type f -name prompts.json -print -quit)"
    fi
    if [[ -z "${prompt_file}" || ! -f "${prompt_file}" ]]; then
      prompt_file="${BASIC_PROMPT_FALLBACK}"
    fi
  else
    suite="pai_bench"
    relative_dir="${video_dir#"${PAI_ROOT}"/}"
    prompt_file="${video_dir}/prompts.json"
    if [[ ! -f "${prompt_file}" ]]; then
      prompt_file="$(find "$(dirname "${video_dir}")" -maxdepth 2 -type f -name prompts.json -print -quit)"
    fi
  fi

  if [[ -z "${prompt_file}" || ! -f "${prompt_file}" ]]; then
    echo "SKIP: no prompts.json found for ${video_dir}" >&2
    skipped=$((skipped + 1))
    continue
  fi

  output_dir="${OUTPUT_ROOT}/${suite}/${relative_dir}"
  mkdir -p "${output_dir}"
  echo "EVALUATE: ${video_dir}"
  echo "PROMPTS:  ${prompt_file}"
  if python -m torch.distributed.run --standalone --nproc_per_node "${NPROC_PER_NODE}" \
      "${REPO_ROOT}/evaluate.py" \
      --mode custom_input \
      --prompt_file "${prompt_file}" \
      --dimension "${DIMENSIONS[@]}" \
      --videos_path "${video_dir}" \
      --output_path "${output_dir}" \
      --enable_missing_videos; then
    evaluated=$((evaluated + 1))
  else
    echo "FAILED: ${video_dir}" >&2
    failures=$((failures + 1))
  fi
done < <(
  find "${BASIC_ROOT}" "${PAI_ROOT}" \
    -type f -name '*.mp4' -not -path '*/.cache/*' -printf '%h\n' |
    sort -u
)

echo "Quality evaluation complete: evaluated=${evaluated} skipped=${skipped} failed=${failures}"
[[ "${failures}" -eq 0 ]]
