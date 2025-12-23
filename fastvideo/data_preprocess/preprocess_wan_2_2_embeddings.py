# Copyright (c) [2025] [FastVideo Team]
# Copyright (c) [2025] [ByteDance Ltd. and/or its affiliates.]
# SPDX-License-Identifier: [Apache License 2.0] 
#
# This file has been modified by [ByteDance Ltd. and/or its affiliates.] in 2025.
#
# Original file was released under [Apache License 2.0], with the full license text
# available at [https://github.com/hao-ai-lab/FastVideo/blob/main/LICENSE].
#
# This modified file is released under the same license.

import argparse
import contextlib
import json
import math
import os
import re

import numpy as np
import torch
import torch.distributed as dist
from accelerate.logging import get_logger
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from diffusers import WanPipeline

logger = get_logger(__name__)


def contains_chinese(text):
    """检查字符串是否包含中文字符"""
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def _round_to_multiple(value, multiple):
    if multiple <= 1:
        return int(value)
    return max(multiple, int(math.floor(value / multiple) * multiple))


def _prepare_conditioning_size(
    image,
    mod_value,
    target_area=None,
    forced_width=None,
    forced_height=None,
):
    width = forced_width
    height = forced_height
    if width is None or height is None:
        if target_area:
            aspect_ratio = image.height / max(image.width, 1)
            est_height = math.sqrt(target_area * aspect_ratio)
            est_width = math.sqrt(target_area / max(aspect_ratio, 1e-6))
        else:
            est_width = image.width
            est_height = image.height
        width = width or est_width
        height = height or est_height
    width = _round_to_multiple(int(width), mod_value)
    height = _round_to_multiple(int(height), mod_value)
    return width, height


def _get_resample_filter():
    if hasattr(Image, "Resampling"):
        return Image.Resampling.LANCZOS
    return Image.LANCZOS


def _get_vae_scaling_factor(vae):
    if hasattr(vae, "scaling_factor"):
        return vae.scaling_factor
    if getattr(getattr(vae, "config", None), "scaling_factor", None):
        return vae.config.scaling_factor
    return 1.0


def encode_conditioning_image(
    pipe,
    image_path,
    device,
    mod_value,
    target_area=None,
    forced_width=None,
    forced_height=None,
):
    image = Image.open(image_path).convert("RGB")
    width, height = _prepare_conditioning_size(
        image,
        mod_value,
        target_area=target_area,
        forced_width=forced_width,
        forced_height=forced_height,
    )
    resample = _get_resample_filter()
    image = image.resize((width, height), resample=resample)
    image = np.array(image).astype(np.float32) / 127.5 - 1.0
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    # Wan VAE expects a temporal dimension (B, C, T, H, W); use a single-frame stub.
    image = image.unsqueeze(2)
    image = image.to(device=device, dtype=pipe.vae.dtype)
    autocast_ctx = (
        torch.autocast(device.type, dtype=pipe.vae.dtype)
        if device.type == "cuda"
        else contextlib.nullcontext()
    )
    with autocast_ctx:
        latents = pipe.vae.encode(image)["latent_dist"].sample()
    latents = latents * _get_vae_scaling_factor(pipe.vae)
    return latents.to(torch.float32).cpu(), width, height


class T5dataset(Dataset):
    _DEFAULT_IMAGE_KEYS = ("image", "image_path", "source_image", "reference_image")
    _DEFAULT_CAPTION_KEYS = ("caption", "prompt", "instruction", "text", "future_caption")

    def __init__(self, txt_path, image_key=None, image_root=None, caption_key=None):
        self.txt_path = txt_path
        self.image_key = image_key
        self.image_root = image_root
        self.caption_key = caption_key
        _, ext = os.path.splitext(self.txt_path)
        self._is_jsonl = ext.lower() in {".json", ".jsonl"}
        self.train_dataset = []
        with open(self.txt_path, "r", encoding="utf-8") as f:
            for raw in f.read().splitlines():
                raw = raw.strip()
                if not raw:
                    continue
                sample = self._parse_line(raw)
                if not sample:
                    continue
                caption = sample["caption"]
                if contains_chinese(caption):
                    continue
                self.train_dataset.append(sample)
        self.has_image_conditioning = any(
            sample.get("image_path") for sample in self.train_dataset
        )

    def _parse_line(self, raw_line):
        payload = None
        if self._is_jsonl:
            try:
                payload = json.loads(raw_line)
            except json.JSONDecodeError:
                payload = None

        if payload is None:
            return {"caption": raw_line, "image_path": None}

        caption = None
        caption_candidates = [self.caption_key] if self.caption_key else []
        caption_candidates.extend(
            key for key in self._DEFAULT_CAPTION_KEYS if key not in caption_candidates
        )
        for key in caption_candidates:
            if key and payload.get(key):
                caption = payload[key]
                break
        if not caption:
            return None

        image_path = None
        key_candidates = [self.image_key] if self.image_key else []
        key_candidates.extend(k for k in self._DEFAULT_IMAGE_KEYS if k not in key_candidates)
        for key in key_candidates:
            if key and payload.get(key):
                image_path = payload[key]
                break
        if image_path and self.image_root and not os.path.isabs(image_path):
            image_path = os.path.join(self.image_root, image_path)

        return {"caption": caption, "image_path": image_path}

    def __len__(self):
        return len(self.train_dataset)

    def __getitem__(self, idx):
        sample = self.train_dataset[idx]
        return dict(
            caption=sample["caption"],
            filename=str(idx),
            conditioning_image=sample.get("image_path"),
        )


def main(args):
    local_rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    print("world_size", world_size, "local rank", local_rank)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl", init_method="env://", world_size=world_size, rank=local_rank
        )

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "prompt_embed"), exist_ok=True)
    os.makedirs(
        os.path.join(args.output_dir, "negative_prompt_embeds"), exist_ok=True
    )

    latents_txt_path = args.prompt_dir
    train_dataset = T5dataset(
        latents_txt_path,
        image_key=args.image_key,
        image_root=args.image_root,
        caption_key=args.caption_key,
    )
    pipe = WanPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).to(device)
    has_image_conditioning = train_dataset.has_image_conditioning
    if has_image_conditioning:
        os.makedirs(os.path.join(args.output_dir, "image_latents"), exist_ok=True)
        patch_size = getattr(getattr(pipe, "transformer", None), "config", None)
        patch_size = getattr(patch_size, "patch_size", 1)
        if isinstance(patch_size, int):
            patch_multiple = patch_size
        elif isinstance(patch_size, (tuple, list)):
            patch_multiple = patch_size[-1]
        else:
            patch_multiple = 1
        vae_scale = getattr(
            pipe, "vae_scale_factor_spatial", getattr(pipe, "vae_scale_factor", 8)
        )
        conditioning_mod_value = max(1, vae_scale * patch_multiple)
    else:
        conditioning_mod_value = None

    sampler = DistributedSampler(
        train_dataset, rank=local_rank, num_replicas=world_size, shuffle=False
    )
    train_dataloader = DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    json_data = []
    for _, data in tqdm(enumerate(train_dataloader), disable=local_rank != 0):
        with torch.inference_mode():
            prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
                prompt=list(data["caption"]),
                do_classifier_free_guidance=True,
            )
            prompt_embeds = prompt_embeds.to(torch.float32).cpu()
            negative_prompt_embeds = negative_prompt_embeds.to(torch.float32).cpu()
            batch_conditioning = data.get("conditioning_image") if has_image_conditioning else None
            for idx, video_name in enumerate(data["filename"]):
                prompt_embed_path = os.path.join(
                    args.output_dir, "prompt_embed", video_name + ".pt"
                )
                negative_prompt_embeds_path = os.path.join(
                    args.output_dir, "negative_prompt_embeds", video_name + ".pt"
                )
                # save latent
                torch.save(prompt_embeds[idx], prompt_embed_path)
                torch.save(negative_prompt_embeds[idx], negative_prompt_embeds_path)
                item = {}
                item["prompt_embed_path"] = video_name + ".pt"
                item["negative_prompt_embeds_path"] = video_name + ".pt"
                item["caption"] = data["caption"][idx]

                if batch_conditioning:
                    conditioning_path = batch_conditioning[idx]
                    if conditioning_path:
                        latents, cond_width, cond_height = encode_conditioning_image(
                            pipe,
                            conditioning_path,
                            device,
                            conditioning_mod_value,
                            target_area=args.conditioning_target_area,
                            forced_width=args.conditioning_width,
                            forced_height=args.conditioning_height,
                        )
                        image_latent_path = os.path.join(
                            args.output_dir, "image_latents", video_name + ".pt"
                        )
                        torch.save(latents, image_latent_path)
                        item["image_conditioning"] = {
                            "latents_path": video_name + ".pt",
                            "width": cond_width,
                            "height": cond_height,
                            "source_path": conditioning_path,
                        }

                json_data.append(item)
    dist.barrier()
    local_data = json_data
    gathered_data = [None] * world_size
    dist.all_gather_object(gathered_data, local_data)
    if local_rank == 0:
        # os.remove(latents_json_path)
        all_json_data = [item for sublist in gathered_data for item in sublist]
        with open(os.path.join(args.output_dir, "videos2caption.json"), "w") as f:
            json.dump(all_json_data, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="data/wan2.2-ti2v-5b-diffusers",
        help="Path to the Wan2.2 TI2V diffusers checkpoint.",
    )
    parser.add_argument(
        "--prompt_dir",
        type=str,
        required=True,
        help="Text file containing one caption per line or JSONL with caption/image info.",
    )
    parser.add_argument(
        "--image_root",
        type=str,
        default=None,
        help="Optional directory to prefix relative conditioning image paths.",
    )
    parser.add_argument(
        "--image_key",
        type=str,
        default=None,
        help="JSON key to read conditioning image path. Defaults to trying common names.",
    )
    parser.add_argument(
        "--caption_key",
        type=str,
        default=None,
        help="JSON key to read prompt text. Defaults to caption/prompt/instruction/text/future_caption.",
    )
    parser.add_argument(
        "--conditioning_width",
        type=int,
        default=None,
        help="Force conditioning images to this width before encoding (rounded by model).",
    )
    parser.add_argument(
        "--conditioning_height",
        type=int,
        default=None,
        help="Force conditioning images to this height before encoding (rounded by model).",
    )
    parser.add_argument(
        "--conditioning_target_area",
        type=int,
        default=None,
        help="Optional area heuristic for conditioning resize when width/height are not provided.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to store prompt embeddings/attention masks/json.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=1,
        help="Number of dataloader workers.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="Prompts processed per device.",
    )
    args = parser.parse_args()
    main(args)
