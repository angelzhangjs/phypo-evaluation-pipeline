#!/usr/bin/env python3
"""
Generate videos for benchmark/custom evaluation from a base LTX-Video model plus
saved GRPO checkpoint (LoRA or state_dict).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio
import numpy as np
import torch

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


def _save_ltx_video_tensor(video: torch.Tensor, path: Path, fps: int) -> None:
    """
    Save LTX output tensor [C, F, H, W] in [0,1] or [0,255]-like space.
    """
    v = video.detach().float().cpu()
    if v.ndim != 4:
        raise ValueError(f"Expected LTX video tensor [C,F,H,W], got {tuple(v.shape)}")
    if v.shape[0] != 3:
        raise ValueError(f"Expected channel-first RGB tensor, got {tuple(v.shape)}")
    frames = v.permute(1, 2, 3, 0).clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8).numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(path), fps=int(fps), codec="libx264", quality=8)
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate LTX-Video videos from saved GRPO checkpoints")
    p.add_argument("--checkpoint-dir", type=str, required=True)
    p.add_argument("--prompt-file", type=str, required=True)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--model-path", type=str, default=None, help="Optional override for base model repo/path")
    p.add_argument("--pipeline-config", type=str, default=None, help="Optional override for LTX pipeline YAML")
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=720)
    p.add_argument("--num-frames", type=int, default=32)
    p.add_argument("--frame-rate", type=int, default=8)
    p.add_argument("--num-inference-steps", type=int, default=50)
    p.add_argument("--guidance-scale", type=float, default=7.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-seeds", type=int, default=1)
    p.add_argument("--seed-stride", type=int, default=1)
    p.add_argument("--negative-prompt", type=str, default="")
    return p


def main() -> None:
    args = build_parser().parse_args()
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    prompt_file = Path(args.prompt_file).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    videos_dir = output_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    metadata = _load_metadata(checkpoint_dir)
    repo_root = Path(__file__).resolve().parents[2]

    pipeline_cfg_path = str(args.pipeline_config or (repo_root / "ltx_video" / "configs" / "ltxv-2b-0.9.6-dev.yaml"))

    from ltx_video.ltx_video.inference import create_ltx_video_pipeline, load_pipeline_config  # type: ignore
    try:
        from ltx_video.utils.skip_layer_strategy import SkipLayerStrategy  # type: ignore
    except Exception:
        from ltx_video.ltx_video.utils.skip_layer_strategy import SkipLayerStrategy  # type: ignore

    pipeline_config = load_pipeline_config(str(pipeline_cfg_path))
    ckpt_name_or_path = str(pipeline_config["checkpoint_path"])
    model_path = args.model_path or metadata.get("model_path") or "Lightricks/LTX-Video"

    if Path(str(model_path)).is_file():
        ckpt_path = str(model_path)
    else:
        from huggingface_hub import hf_hub_download  # type: ignore
        ckpt_path = hf_hub_download(repo_id=str(model_path), filename=ckpt_name_or_path, repo_type="model")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = create_ltx_video_pipeline(
        ckpt_path=str(ckpt_path),
        precision=str(pipeline_config.get("precision", "bfloat16")),
        text_encoder_model_name_or_path=str(pipeline_config["text_encoder_model_name_or_path"]),
        sampler=pipeline_config.get("sampler", None),
        device=str(device),
        enhance_prompt=False,
    )

    # Rebuild LoRA structure if needed, otherwise load state_dict.
    transformer = pipeline.transformer
    checkpoint_type = str(metadata.get("checkpoint_type", "lora"))
    if checkpoint_type == "lora":
        blocks = getattr(transformer, "transformer_blocks", None)
        try:
            total_blocks = len(blocks) if blocks is not None else None
        except Exception:
            total_blocks = None
        target_blocks = resolve_lora_blocks(
            spec=metadata.get("lora_blocks", None),
            total_blocks=total_blocks,
            unfreeze_pct=float(metadata.get("unfreeze_percentage", 0.20)),
        )
        pipeline.transformer, _ = apply_lora_to_transformer(
            transformer,
            rank=int(metadata.get("lora_rank", 4)),
            alpha=int(metadata.get("lora_alpha", 8)),
            target_blocks=target_blocks,
        )
        lora_dir = checkpoint_dir / "lora_adapter"
        lora_state = checkpoint_dir / "lora_state_dict.pt"
        if lora_dir.exists():
            load_lora_weights(pipeline.transformer, str(lora_dir))
        elif lora_state.exists():
            load_lora_weights(pipeline.transformer, str(lora_state))
        else:
            raise FileNotFoundError(f"No LoRA checkpoint found in {checkpoint_dir}")
    else:
        state_path = checkpoint_dir / "model_state_dict.pt"
        if not state_path.exists():
            raise FileNotFoundError(f"Missing model_state_dict.pt in {checkpoint_dir}")
        state = torch.load(state_path, map_location="cpu")
        pipeline.transformer.load_state_dict(state, strict=False)

    stg_mode = str(pipeline_config.get("stg_mode", "attention_values")).lower()
    if stg_mode in ("stg_av", "attention_values"):
        skip_layer_strategy = SkipLayerStrategy.AttentionValues
    elif stg_mode in ("stg_as", "attention_skip"):
        skip_layer_strategy = SkipLayerStrategy.AttentionSkip
    elif stg_mode in ("stg_r", "residual"):
        skip_layer_strategy = SkipLayerStrategy.Residual
    elif stg_mode in ("stg_t", "transformer_block"):
        skip_layer_strategy = SkipLayerStrategy.TransformerBlock
    else:
        skip_layer_strategy = SkipLayerStrategy.AttentionValues

    prompts = _read_prompts(prompt_file)
    manifest = []
    for idx, prompt in enumerate(prompts, start=1):
        video_id = _safe_video_id(idx)
        manifest.append({"video_id": video_id, "prompt_en": prompt})
        for seed_idx in range(int(args.num_seeds)):
            seed = int(args.seed) + seed_idx * int(args.seed_stride)
            sample = {
                "prompt": prompt,
                "prompt_attention_mask": None,
                "negative_prompt": str(args.negative_prompt),
                "negative_prompt_attention_mask": None,
            }
            generator = torch.Generator(device=device).manual_seed(seed)
            images = pipeline(
                **pipeline_config,
                skip_layer_strategy=skip_layer_strategy,
                generator=generator,
                output_type="pt",
                callback_on_step_end=None,
                height=int(args.height),
                width=int(args.width),
                num_frames=int(args.num_frames),
                frame_rate=int(args.frame_rate),
                **sample,
                media_items=None,
                conditioning_items=None,
                is_video=True,
                vae_per_channel_normalize=True,
                image_cond_noise_scale=float(pipeline_config.get("image_cond_noise_scale", 0.0)),
                mixed_precision=(str(pipeline_config.get("precision", "bfloat16")) == "mixed_precision"),
                offload_to_cpu=False,
                device=device,
                enhance_prompt=False,
            ).images
            video = images[0]  # [C, F, H, W]
            out_path = videos_dir / f"{video_id}__{seed}.mp4"
            _save_ltx_video_tensor(video, out_path, fps=int(args.frame_rate))

    manifest_path = output_dir / "prompt_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"✅ Videos saved to: {videos_dir}")
    print(f"✅ Prompt manifest saved to: {manifest_path}")


if __name__ == "__main__":
    main()
