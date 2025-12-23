import argparse
import math
import os

import numpy as np
import torch
from diffusers import WanPipeline

try:
    from diffusers import AutoPipelineForImage2Video
except ImportError:
    AutoPipelineForImage2Video = None

try:
    from diffusers import WanImageToVideoPipeline
except ImportError:
    WanImageToVideoPipeline = None
from diffusers.utils import export_to_video, load_image


def round_dimension(value, multiple):
    return int(math.floor(value / multiple) * multiple)


def prepare_size(image, target_area, mod_value):
    aspect_ratio = image.height / image.width
    height = round(np.sqrt(target_area * aspect_ratio))
    width = round(np.sqrt(target_area / aspect_ratio))
    height = max(mod_value, round_dimension(height, mod_value))
    width = max(mod_value, round_dimension(width, mod_value))
    return width, height


def parse_args():
    parser = argparse.ArgumentParser(description="Debug Wan2.2 TI2V inference.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="data/wan2.2-ti2v-5b-diffusers",
        help="Path to Wan2.2 diffusers checkpoint.",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Optional conditioning image path or URL. If omitted, pure text-to-video is run.",
    )
    parser.add_argument(
        "--prompt", type=str, required=True, help="Positive prompt text."
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="",
        help="Negative prompt text.",
    )
    parser.add_argument(
        "--output", type=str, default="wan2_2_debug.mp4", help="Output video path."
    )
    parser.add_argument(
        "--height", type=int, default=None, help="Optional override for output height."
    )
    parser.add_argument(
        "--width", type=int, default=None, help="Optional override for output width."
    )
    parser.add_argument(
        "--num_frames", type=int, default=81, help="Number of frames to sample."
    )
    parser.add_argument(
        "--num_steps", type=int, default=30, help="Denoising steps per video."
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=5.0,
        help="Classifier-free guidance scale.",
    )
    parser.add_argument(
        "--fps", type=int, default=16, help="FPS when saving the debug video."
    )
    parser.add_argument(
        "--target_area",
        type=int,
        default=480 * 832,
        help="Used to estimate width/height when not provided.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device for inference."
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed.")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    image = load_image(args.image) if args.image else None
    if image is not None:
        if WanImageToVideoPipeline is not None:
            pipe = WanImageToVideoPipeline.from_pretrained(
                args.model_path,
                torch_dtype=torch.bfloat16,
            ).to(device)
            print("Using WanImageToVideoPipeline for inference.")
        elif AutoPipelineForImage2Video is not None:
            pipe = AutoPipelineForImage2Video.from_pretrained(
                args.model_path,
                torch_dtype=torch.bfloat16,
            ).to(device)
            print("Using AutoPipelineForImage2Video for inference.")
        else:
            raise RuntimeError(
                "Image conditioning requires diffusers to expose WanImageToVideoPipeline "
                "or AutoPipelineForImage2Video. Please upgrade diffusers or drop --image."
            )
    else:
        pipe = WanPipeline.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
        ).to(device)

    patch_size = getattr(getattr(pipe, "transformer", None), "config", None)
    patch_size = getattr(patch_size, "patch_size", 1)
    if isinstance(patch_size, int):
        patch_multiple = patch_size
    elif isinstance(patch_size, (tuple, list)):
        patch_multiple = patch_size[1]
    else:
        patch_multiple = 1
    vae_scale = getattr(
        pipe, "vae_scale_factor_spatial", getattr(pipe, "vae_scale_factor", 8)
    )
    mod_value = max(1, vae_scale * patch_multiple)

    width, height = args.width, args.height
    if width is None or height is None:
        if image is None:
            raise ValueError(
                "Please provide --height/--width when no conditioning image is supplied."
            )
        width, height = prepare_size(image, args.target_area, mod_value)
    else:
        width = round_dimension(width, mod_value)
        height = round_dimension(height, mod_value)

    if image is not None:
        image = image.resize((width, height))

    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=device).manual_seed(args.seed)

    pipe_kwargs = {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "height": height,
        "width": width,
        "num_frames": args.num_frames,
        "num_inference_steps": args.num_steps,
        "guidance_scale": args.guidance_scale,
    }
    if generator is not None:
        pipe_kwargs["generator"] = generator
    if image is not None:
        pipe_kwargs["image"] = image

    output = pipe(**pipe_kwargs).frames[0]
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    export_to_video(output, args.output, fps=args.fps)
    print(f"Saved debug video to {args.output}")


if __name__ == "__main__":
    main()


# python scripts/inference/run_wan2_2_i2v.py   --model_path data/wan2.2-ti2v-5b-diffusers   --image ./assets/debug_frame.jpg   --prompt "A cinematic close-up of a golden retriever running through tall grass at sunset"   --negative_prompt ""   --num_steps 30   --guidance_scale 5.0   --output videos/wan22_debug.mp4   --seed 1234
