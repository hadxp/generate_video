#!/usr/bin/uv
"""
LTX-2.5 local I2V: 1 start frame + 2 reference frames as keyframes.

Requires Lightricks/LTX-2 (ltx-pipelines) and the split LTX-2.5 bf16 pack.
Do not use *-comfy-int8-convrot.safetensors here.

Example (8s @ 24fps, 720p landscape):

  python ltx25_start_two_refs.py \
    --start start.jpg \
    --ref1 character.jpg \
    --ref2 end.jpg \
    --prompt "Slow orbit around the subject, wind in the coat, soft room tone" \
    --seconds 8 \
    --width 1280 --height 704
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
from ltx_pipelines.utils.model_paths import ModelPaths
from ltx_pipelines.utils.media_io.encode import encode_video

i=0

def frames_for_seconds(seconds: float, fps: float) -> int:
    """VAE grid: num_frames = 8k + 1."""
    raw = int(round(seconds * fps))
    k = max(1, round((raw - 1) / 8))
    return 8 * k + 1


def snap32(n: int) -> int:
    return max(32, n - (n % 32))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LTX-2.5 start + two reference keyframes")
    p.add_argument("--models", type=Path, default=Path("models/ltx-2.5"))
    p.add_argument("--start", type=Path, required=True, help="First-frame image")
    p.add_argument("--ref1", type=Path, required=True, help="Mid-clip reference still")
    p.add_argument("--ref2", type=Path, required=True, help="Last-frame / second reference still")
    p.add_argument("--prompt", required=True)
    p.add_argument("--output", type=Path, default=Path("out_ltx25.mp4"))
    p.add_argument("--seconds", type=float, default=8.0)
    p.add_argument("--fps", type=float, default=24.0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=704)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--start-strength", type=float, default=1.0)
    p.add_argument("--ref1-strength", type=float, default=0.85)
    p.add_argument("--ref2-strength", type=float, default=0.9)
    p.add_argument("--ref1-at", choices=("quarter", "mid", "three-quarter"), default="mid")
    p.add_argument("--enhance-prompt", action="store_true")
    p.add_argument("--quantization", default=None, help="e.g. fp8-cast")
    p.add_argument("--offload", default=None, help="e.g. cpu")
    return p.parse_args()


def model_pack(root: Path) -> tuple[ModelPaths, Path]:
    transformer = root / "diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors"
    text_enc = root / "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
    video_vae = root / "vae/ltx-2.5-video-vae-bf16.safetensors"
    audio_vae = root / "vae/ltx-2.5-audio-vae-bf16.safetensors"
    duration = root / "model_patches/ltx-2.5-duration-head-bf16.safetensors"
    upsampler = root / "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
    missing = [p for p in (transformer, text_enc, video_vae, audio_vae, duration, upsampler) if not p.is_file()]
    if missing:
        raise SystemExit("Missing weights:\n  " + "\n  ".join(str(m) for m in missing))
    paths = ModelPaths.from_split(
        transformer_path=str(transformer),
        text_encoder_path=str(text_enc),
        video_vae_path=str(video_vae),
        audio_vae_path=str(audio_vae),
        duration_head_path=str(duration),
    )
    return paths, upsampler


def ref1_index(num_frames: int, where: str) -> int:
    frac = {"quarter": 0.25, "mid": 0.5, "three-quarter": 0.75}[where]
    idx = int(round((num_frames - 1) * frac))
    # keep off the exact start/end slots
    return min(max(idx, 8), num_frames - 9)


def main() -> None:
    args = parse_args()
    for img in (args.start, args.ref1, args.ref2):
        if not img.is_file():
            raise SystemExit(f"Image not found: {img}")

    width, height = snap32(args.width), snap32(args.height)
    num_frames = frames_for_seconds(args.seconds, args.fps)
    last = num_frames - 1
    mid = ref1_index(num_frames, args.ref1_at)

    images: list[tuple[str, int, float]] = [
        (str(args.start.resolve()), 0, args.start_strength),
        (str(args.ref1.resolve()), mid, args.ref1_strength),
        (str(args.ref2.resolve()), last, args.ref2_strength),
    ]

    print(
        f"{width}x{height}  {num_frames} frames @ {args.fps} fps "
        f"(~{(num_frames - 1) / args.fps:.2f}s)\n"
        f"  start  frame 0     {args.start}\n"
        f"  ref1   frame {mid:<5} {args.ref1}\n"
        f"  ref2   frame {last:<5} {args.ref2}"
    )

    model_paths, upsampler = model_pack(args.models)
    pipe_kwargs = dict(model_paths=model_paths, spatial_upsampler_path=str(upsampler))
    # optional kwargs if your installed ltx-pipelines build exposes them
    if args.quantization or args.offload:
        try:
            pipe = TI2VidTwoStagesPipeline(
                **pipe_kwargs,
                quantization=args.quantization,
                offload_mode=args.offload,
            )
        except TypeError:
            pipe = TI2VidTwoStagesPipeline(**pipe_kwargs)
    else:
        pipe = TI2VidTwoStagesPipeline(**pipe_kwargs)

    video, audio = pipe(
        prompt=args.prompt,
		negative_prompt="blur, morphing, extra limbs, watermark",
		seed=args.seed,
		height=height,
		width=width,
		num_frames=num_frames,
		frame_rate=args.fps,
		num_inference_steps=30,
		images=images,  # same [(start,0,1.0), (ref1,mid,0.85), (ref2,last,0.9)]
		enhance_prompt=args.enhance_prompt,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        encode_video(video=video, audio=audio, fps=args.fps, output_path=str(args.output))
    except TypeError:
        encode_video(video=video, fps=args.fps, output_path=str(args.output))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()