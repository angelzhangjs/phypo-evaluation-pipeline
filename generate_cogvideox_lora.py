#!/usr/bin/env python3
"""
Generate videos for benchmark evaluation from a base CogVideoX model + saved continual LoRA checkpoint.

Outputs:
- MP4 files under <output_dir>/videos
- prompt manifest JSON under <output_dir>/prompt_manifest.json

The manifest can be used with:
  python evaluate.py --mode custom_input --prompt_file <manifest> --videos_path <output_dir>/videos ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from diffusers import CogVideoXDPMScheduler, CogVideoXPipeline
from diffusers.utils import export_to_video

from unified_grpo.lora_utils import apply_lora_to_transformer, load_lora_weights
from unified_grpo.utils import resolve_lora_blocks


def _read_prompts(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _load_metadata(checkpoint_dir: Path) -> dict:
    meta_path = checkpoint_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata.json in checkpoint dir: {checkpoint_dir}")
    return json.loads(meta_path.read_text())


def _safe_video_id(i: int) -> str:
    return f"video_{i:04d}"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate CogVideoX videos from saved continual LoRA for benchmark evaluation")
    p.add_argument("--model-path", type=str, default=None, help="Optional override for base CogVideoX model path")
    p.add_argument("--checkpoint-dir", type=str, required=True, help="Directory containing continual-training final checkpoint")
    p.add_argument("--prompt-file", type=str, required=True, help="Plain text file, one prompt per line")
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=720)
    p.add_argument("--num-frames", type=int, default=32)
    p.add_argument("--guidance-scale", type=float, default=7.5)
    p.add_argument("--num-inference-steps", type=int, default=50)
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-seeds", type=int, default=1)
    p.add_argument("--seed-stride", type=int, default=1)
    return p


def main() -> None:
    args = build_parser().parse_args()
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    prompt_file = Path(args.prompt_file).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    videos_dir = output_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    metadata = _load_metadata(checkpoint_dir)
    base_model_path = str(args.model_path or metadata.get("model_path"))
    lora_dir = checkpoint_dir / "lora_adapter"
    if not lora_dir.exists():
        raise FileNotFoundError(f"Missing LoRA adapter directory: {lora_dir}")

    prompts = _read_prompts(prompt_file)
    if not prompts:
        raise ValueError(f"No prompts found in {prompt_file}")

    print(f"Loading CogVideoX base model: {base_model_path}")
    pipe = CogVideoXPipeline.from_pretrained(base_model_path, torch_dtype=torch.bfloat16).to("cuda")
    transformer = pipe.transformer

    lora_rank = int(metadata.get("lora_rank", 4))
    lora_alpha = int(metadata.get("lora_alpha", 8))
    lora_blocks = metadata.get("lora_blocks", "last")
    blocks = getattr(transformer, "transformer_blocks", None)
    try:
        total_blocks = len(blocks) if blocks is not None else None
    except Exception:
        total_blocks = None
    target_blocks = resolve_lora_blocks(
        spec=lora_blocks,
        total_blocks=total_blocks,
        unfreeze_pct=float(metadata.get("unfreeze_percentage", 0.20)),
    )

    pipe.transformer, _ = apply_lora_to_transformer(
        transformer,
        rank=lora_rank,
        alpha=lora_alpha,
        target_blocks=target_blocks,
    )
    load_lora_weights(pipe.transformer, str(lora_dir))

    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    manifest = []
    for idx, prompt in enumerate(prompts, start=1):
        video_id = _safe_video_id(idx)
        manifest.append({"video_id": video_id, "prompt_en": prompt})
        print(f"[{idx}/{len(prompts)}] {prompt}")
        for seed_idx in range(int(args.num_seeds)):
            seed = int(args.seed) + seed_idx * int(args.seed_stride)
            frames = pipe(
                prompt=prompt,
                height=int(args.height),
                width=int(args.width),
                num_videos_per_prompt=1,
                num_inference_steps=int(args.num_inference_steps),
                num_frames=int(args.num_frames),
                use_dynamic_cfg=True,
                guidance_scale=float(args.guidance_scale),
                generator=torch.Generator(device="cpu").manual_seed(seed),
            ).frames[0]
            out_path = videos_dir / f"{video_id}__{seed}.mp4"
            export_to_video(frames, str(out_path), fps=int(args.fps))

    manifest_path = output_dir / "prompt_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"✅ Videos saved to: {videos_dir}")
    print(f"✅ Prompt manifest saved to: {manifest_path}")
    print("Example evaluation command:")
    print(
        f"python evaluate.py --mode custom_input --prompt_file \"{manifest_path}\" "
        f"--videos_path \"{videos_dir}\" --dimension aesthetic_quality motion_smoothness overall_consistency"
    )


if __name__ == "__main__":
    main()
