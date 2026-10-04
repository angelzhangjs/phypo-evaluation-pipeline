#!/usr/bin/env python3
"""
Generate videos for benchmark/custom evaluation from a base Wan model plus
saved GRPO checkpoint (LoRA or state_dict).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio
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


def _save_wan_video_tensor(video: torch.Tensor, path: Path, fps: int) -> None:
    """
    Save WAN output tensor [C, F, H, W] or [F, C, H, W].
    """
    v = video.detach().float().cpu()
    if v.ndim != 4:
        raise ValueError(f"Expected WAN video tensor rank 4, got {tuple(v.shape)}")
    if v.shape[0] == 3 and v.shape[1] != 3:
        v = v.permute(1, 0, 2, 3).contiguous()
    elif v.shape[1] == 3:
        pass
    else:
        raise ValueError(f"Unexpected WAN video tensor shape: {tuple(v.shape)}")
    frames = v.permute(0, 2, 3, 1).clamp(-1.0, 1.0)
    frames = ((frames + 1.0) / 2.0).clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8).numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(path), fps=int(fps), codec="libx264", quality=8)
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate WAN videos from saved GRPO checkpoints")
    p.add_argument("--checkpoint-dir", type=str, required=True)
    p.add_argument("--prompt-file", type=str, required=True)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--model-path", type=str, default=None, help="Optional override for base Wan checkpoint dir or HF repo")
    p.add_argument("--wan-task", type=str, default="t2v-1.3B")
    p.add_argument("--wan-size", type=str, default="832*480")
    p.add_argument("--num-frames", type=int, default=33)
    p.add_argument("--guidance-scale", type=float, default=6.0)
    p.add_argument("--num-inference-steps", type=int, default=50)
    p.add_argument("--sample-shift", type=float, default=5.0)
    p.add_argument("--sample-solver", type=str, default="unipc")
    p.add_argument("--fps", type=int, default=16)
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
    wan_path = repo_root / "Wan2.1"
    import sys
    if str(wan_path) not in sys.path:
        sys.path.insert(0, str(wan_path))
    import wan  # type: ignore

    ckpt_dir = Path(str(args.model_path or metadata.get("model_path") or ""))
    if not ckpt_dir.exists():
        from huggingface_hub import snapshot_download  # type: ignore
        downloaded = snapshot_download(repo_id=str(args.model_path or metadata.get("model_path")))
        ckpt_dir = Path(downloaded)

    wan_task = str(args.wan_task)
    if wan_task not in wan.configs.WAN_CONFIGS:
        raise ValueError(f"Unknown wan task: {wan_task}")
    cfg = wan.configs.WAN_CONFIGS[wan_task]

    pipeline = wan.WanT2V(
        config=cfg,
        checkpoint_dir=str(ckpt_dir),
        device_id=torch.cuda.current_device() if torch.cuda.is_available() else 0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=True,
    )

    checkpoint_type = str(metadata.get("checkpoint_type", "lora"))
    if checkpoint_type == "lora":
        model = getattr(pipeline, "model", None)
        if model is None:
            raise RuntimeError("WAN pipeline has no .model for LoRA loading")
        blocks = getattr(model, "blocks", None)
        total_blocks = len(blocks) if blocks is not None else None
        target_blocks = resolve_lora_blocks(
            spec=metadata.get("lora_blocks", None),
            total_blocks=total_blocks,
            unfreeze_pct=float(metadata.get("unfreeze_percentage", 0.20)),
        )
        pipeline.model, _ = apply_lora_to_transformer(
            model,
            rank=int(metadata.get("lora_rank", 4)),
            alpha=int(metadata.get("lora_alpha", 8)),
            target_modules=["q", "k", "v", "o", "k_img", "v_img"],
            target_blocks=target_blocks,
        )
        lora_dir = checkpoint_dir / "lora_adapter"
        lora_state = checkpoint_dir / "lora_state_dict.pt"
        if lora_dir.exists():
            load_lora_weights(pipeline.model, str(lora_dir))
        elif lora_state.exists():
            load_lora_weights(pipeline.model, str(lora_state))
        else:
            raise FileNotFoundError(f"No LoRA checkpoint found in {checkpoint_dir}")
    else:
        state_path = checkpoint_dir / "model_state_dict.pt"
        if not state_path.exists():
            raise FileNotFoundError(f"Missing model_state_dict.pt in {checkpoint_dir}")
        state = torch.load(state_path, map_location="cpu")
        pipeline.model.load_state_dict(state, strict=False)

    prompts = _read_prompts(prompt_file)
    manifest = []
    width, height = [int(x) for x in str(args.wan_size).split("*", 1)]
    for idx, prompt in enumerate(prompts, start=1):
        video_id = _safe_video_id(idx)
        manifest.append({"video_id": video_id, "prompt_en": prompt})
        for seed_idx in range(int(args.num_seeds)):
            seed = int(args.seed) + seed_idx * int(args.seed_stride)
            video = pipeline.generate(
                prompt,
                size=(width, height),
                frame_num=int(args.num_frames),
                shift=float(args.sample_shift),
                sample_solver=str(args.sample_solver),
                sampling_steps=int(args.num_inference_steps),
                guide_scale=float(args.guidance_scale),
                n_prompt=str(args.negative_prompt),
                seed=seed,
                offload_model=True,
            )
            out_path = videos_dir / f"{video_id}__{seed}.mp4"
            _save_wan_video_tensor(video, out_path, fps=int(args.fps))

    manifest_path = output_dir / "prompt_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"✅ Videos saved to: {videos_dir}")
    print(f"✅ Prompt manifest saved to: {manifest_path}")


if __name__ == "__main__":
    main()
