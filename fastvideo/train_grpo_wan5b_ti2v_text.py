import argparse
import logging
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from accelerate.utils import set_seed
from diffusers import UniPCMultistepScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.video_processor import VideoProcessor
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

import wandb
from fastvideo.dataset.latent_wan_2_2_rl_datasets import (
    LatentDataset,
    latent_collate_function,
)
from fastvideo.utils.checkpoint import (
    resume_lora_optimizer,
    save_checkpoint,
    save_lora_checkpoint,
)
from fastvideo.utils.communications import sp_parallel_dataloader_wrapper
from fastvideo.utils.fsdp_util import apply_fsdp_checkpointing, get_dit_fsdp_kwargs
from fastvideo.utils.load import load_transformer, load_vae
from fastvideo.utils.logging_ import main_print
from fastvideo.utils.parallel_states import (
    destroy_sequence_parallel_group,
    get_sequence_parallel_state,
    initialize_sequence_parallel_state,
    nccl_info,
)

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.31.0")
from collections import deque

from diffusers.utils import export_to_video

WAN5B_SPATIAL_DOWNSAMPLE = 8
WAN5B_TEMPORAL_DOWNSAMPLE = 4
WAN5B_CHANNELS = 48


def _unwrap_module(module):
    """
    Unwrap nested FSDP / DDP containers to access the underlying model.
    """
    while hasattr(module, "module"):
        module = module.module
    return module


def _reshape_vae_latent_stats(vae, latents):
    latents_mean = vae.config.latents_mean
    latents_std = vae.config.latents_std
    if latents_mean is None and hasattr(vae.config, "latents_mean"):
        latents_mean = vae.config.latents_mean
    if latents_std is None and hasattr(vae.config, "latents_std"):
        latents_std = vae.config.latents_std
    if latents_mean is None or latents_std is None:
        return None, None
    latents_mean = torch.as_tensor(
        latents_mean, device=latents.device, dtype=latents.dtype
    )
    latents_std = torch.as_tensor(
        latents_std, device=latents.device, dtype=latents.dtype
    )
    # Wan VAEs store precision (1/sigma); invert to get std.
    latents_std = torch.where(
        latents_std != 0,
        1.0 / latents_std,
        torch.ones_like(latents_std),
    )
    channel_dim = latents_mean.shape[0]
    view_shape = (1, channel_dim) + (1,) * (latents.ndim - 2)
    return latents_mean.view(view_shape), latents_std.view(view_shape)


def sd3_time_shift(shift, t):
    return (shift * t) / (1 + (shift - 1) * t)


def flux_step(
    model_output: torch.Tensor,
    latents: torch.Tensor,
    eta: float,
    sigmas: torch.Tensor,
    index: int,
    prev_sample: torch.Tensor,
    grpo: bool,
    sde_solver: bool,
):
    sigma = sigmas[index]
    dsigma = sigmas[index + 1] - sigma
    prev_sample_mean = latents + dsigma * model_output

    pred_original_sample = latents - sigma * model_output

    delta_t = sigma - sigmas[index + 1]
    std_dev_t = eta * math.sqrt(delta_t)

    if sde_solver:
        score_estimate = -(latents - pred_original_sample * (1 - sigma)) / sigma**2
        log_term = -0.5 * eta**2 * score_estimate
        prev_sample_mean = prev_sample_mean + log_term * dsigma

    if grpo and prev_sample is None:
        prev_sample = prev_sample_mean + torch.randn_like(prev_sample_mean) * std_dev_t

    if grpo:
        # log prob of prev_sample given prev_sample_mean and std_dev_t
        log_prob = (
            (
                -(
                    (
                        prev_sample.detach().to(torch.float32)
                        - prev_sample_mean.to(torch.float32)
                    )
                    ** 2
                )
                / (2 * (std_dev_t**2))
            )
            - math.log(std_dev_t)
            - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
        )

        # mean along all but batch dimension
        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
        return prev_sample, pred_original_sample, log_prob
    else:
        return prev_sample_mean, pred_original_sample


def assert_eq(x, y, msg=None):
    assert x == y, f"{msg or 'Assertion failed'}: {x} != {y}"


def _prepare_sampling_schedule(args, device):
    """
    Wan Ti2V checkpoints are distributed with a UniPC scheduler whose sigmas
    and timesteps need to be respected; otherwise the model receives noise
    levels it was never trained on.  We cache the scheduler on the args
    namespace to avoid reloading it from disk every sampling call.
    """
    scheduler = getattr(args, "_wan_scheduler", None)
    if scheduler is None:
        scheduler = UniPCMultistepScheduler.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="scheduler",
            local_files_only=True,
        )
        setattr(args, "_wan_scheduler", scheduler)
    scheduler.set_timesteps(args.sampling_steps, device=device)
    sigma_schedule = getattr(scheduler, "sigmas", None)
    if sigma_schedule is None:
        sigma_schedule = torch.linspace(
            1, 0, args.sampling_steps + 1, device=device, dtype=torch.float32
        )
        sigma_schedule = sd3_time_shift(args.shift, sigma_schedule)
        timesteps = torch.linspace(
            args.sampling_steps - 1,
            0,
            steps=args.sampling_steps,
            device=device,
            dtype=torch.float32,
        )
        timesteps = (timesteps / timesteps.max().clamp_min(1.0) * 1000).to(torch.long)
        init_sigma = float(sigma_schedule[0].item())
        return sigma_schedule, timesteps, init_sigma
    sigma_schedule = sigma_schedule.to(device=device, dtype=torch.float32)
    if sigma_schedule.shape[0] == args.sampling_steps:
        sigma_schedule = torch.cat([sigma_schedule, sigma_schedule.new_zeros(1)], dim=0)
    timesteps = scheduler.timesteps.to(device=device)
    init_sigma = float(getattr(scheduler, "init_noise_sigma", sigma_schedule[0].item()))
    return sigma_schedule, timesteps, init_sigma, scheduler


def run_sample_step(
    args,
    z,
    progress_bar,
    sigma_schedule,
    timestep_schedule,
    scheduler,
    transformer,
    encoder_hidden_states,
    negative_prompt_embeds,
    grpo_sample,
):
    if not grpo_sample:
        zero = torch.zeros(1, device=z.device)
        return z, z, zero, zero

    all_latents = [z]
    all_log_probs = []

    compare_scheduler = getattr(args, "compare_scheduler", False) and not getattr(
        args, "_scheduler_compare_done", False
    )
    # if compare_scheduler and dist.is_initialized() and dist.get_rank() != 0:
    #     compare_scheduler = False
    compare_scheduler_active = compare_scheduler
    cmp_scheduler = None
    cmp_latents = None
    if compare_scheduler:
        cmp_scheduler = UniPCMultistepScheduler.from_config(scheduler.config)
        cmp_scheduler.set_timesteps(args.sampling_steps, device=z.device)
        cmp_latents = z.clone().to(torch.float32)
        compare_limit = getattr(args, "compare_scheduler_max_steps", 4)
        tolerance = getattr(args, "compare_scheduler_tolerance", 1e-4)
    for i in progress_bar:
        B = encoder_hidden_states.shape[0]
        timestep_value = timestep_schedule[i]
        timestep_scalar = (
            timestep_value.to(device=z.device)
            if torch.is_tensor(timestep_value)
            else torch.tensor(
                timestep_value, device=z.device, dtype=timestep_schedule.dtype
            )
        )
        timesteps = torch.full(
            [B],
            timestep_scalar.item(),
            device=z.device,
            dtype=timestep_schedule.dtype,
        )
        transformer.eval()
        if args.cfg_infer > 1:
            with torch.autocast("cuda", torch.bfloat16):
                latent_z = torch.cat([z, z], dim=0)
                cond_states = torch.cat(
                    [encoder_hidden_states, negative_prompt_embeds], dim=0
                )
                timestep_inputs = torch.cat([timesteps, timesteps], dim=0)
                scaled_inputs = scheduler.scale_model_input(
                    latent_z.to(torch.float32), timestep_inputs
                ).to(torch.bfloat16)
                pred = transformer(
                    hidden_states=scaled_inputs,
                    timestep=timestep_inputs,
                    encoder_hidden_states=cond_states,
                    attention_kwargs=None,
                    return_dict=False,
                )[0]
                model_pred, uncond_pred = pred.chunk(2)
                pred = uncond_pred.to(torch.float32) + args.cfg_infer * (
                    model_pred.to(torch.float32) - uncond_pred.to(torch.float32)
                )
        else:
            with torch.autocast("cuda", torch.bfloat16):
                model_inputs = scheduler.scale_model_input(
                    z.to(torch.float32), timesteps
                ).to(torch.bfloat16)
                pred = transformer(
                    hidden_states=model_inputs,
                    timestep=timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_kwargs=None,
                    return_dict=False,
                )[0]
        z, pred_original, log_prob = flux_step(
            pred,
            z.to(torch.float32),
            args.eta,
            sigmas=sigma_schedule,
            index=i,
            prev_sample=None,
            grpo=True,
            sde_solver=True,
        )
        z = z.to(torch.bfloat16)
        if compare_scheduler and i < compare_limit:
            cmp_prev = cmp_scheduler.step(
                pred.to(torch.float32),
                timestep_scalar,
                cmp_latents,
                return_dict=False,
            )[0]
            diff = torch.max(torch.abs(cmp_prev - z.to(torch.float32))).item()
            main_print(
                f"[SchedulerCompare] step={i} max|Δ|={diff:.6e} (tol={tolerance})"
            )
            cmp_latents = cmp_prev
            if diff > tolerance:
                main_print(
                    "[SchedulerCompare] Detected mismatch between flux_step and scheduler.step."
                )
        if compare_scheduler and i + 1 == compare_limit:
            setattr(args, "_scheduler_compare_done", True)
            compare_scheduler = False
        all_latents.append(z)
        all_log_probs.append(log_prob)

    latents = pred_original
    all_latents = torch.stack(all_latents, dim=1)
    all_log_probs = torch.stack(all_log_probs, dim=1)
    if compare_scheduler_active:
        setattr(args, "_scheduler_compare_done", True)
    return z, latents, all_latents, all_log_probs


def grpo_one_step(
    args,
    latents,
    pre_latents,
    encoder_hidden_states,
    negative_prompt_embeds,
    scheduler,
    transformer,
    timesteps,
    i,
    sigma_schedule,
):
    transformer.train()
    if args.cfg_infer > 1:
        with torch.autocast("cuda", torch.bfloat16):
            latent_z = torch.cat([latents, latents], dim=0)
            timestep_inputs = torch.cat([timesteps, timesteps], dim=0)
            cond_states = torch.cat(
                [encoder_hidden_states, negative_prompt_embeds], dim=0
            )
            scaled_inputs = scheduler.scale_model_input(
                latent_z.to(torch.float32), timestep_inputs
            ).to(torch.bfloat16)
            pred = transformer(
                hidden_states=scaled_inputs,
                timestep=timestep_inputs,
                encoder_hidden_states=cond_states,
                attention_kwargs=None,
                return_dict=False,
            )[0]
            model_pred, uncond_pred = pred.chunk(2)
            pred = uncond_pred.to(torch.float32) + args.cfg_infer * (
                model_pred.to(torch.float32) - uncond_pred.to(torch.float32)
            )
    else:
        with torch.autocast("cuda", torch.bfloat16):
            model_inputs = scheduler.scale_model_input(
                latents.to(torch.float32), timesteps
            ).to(torch.bfloat16)
            pred = transformer(
                hidden_states=model_inputs,
                timestep=timesteps,
                encoder_hidden_states=encoder_hidden_states,
                attention_kwargs=None,
                return_dict=False,
            )[0]
    z, pred_original, log_prob = flux_step(
        pred,
        latents.to(torch.float32),
        args.eta,
        sigma_schedule,
        i,
        prev_sample=pre_latents.to(torch.float32),
        grpo=True,
        sde_solver=True,
    )
    return log_prob


def sample_reference_model(
    args,
    device,
    transformer,
    vae,
    encoder_hidden_states,
    negative_prompt_embeds,
    caption,
    inferencer=None,
):
    w, h, t = args.w, args.h, args.t
    sample_steps = args.sampling_steps
    (
        sigma_schedule,
        timestep_schedule,
        init_noise_sigma,
        scheduler,
    ) = _prepare_sampling_schedule(args, device)
    assert_eq(
        sigma_schedule.shape[0],
        sample_steps + 1,
        "sigma schedule must match step count",
    )
    noise_scale = torch.as_tensor(init_noise_sigma, device=device, dtype=torch.float32)

    transformer_config = getattr(_unwrap_module(transformer), "config", None)
    vae_scale_factor = vae.config.scale_factor_spatial // vae.config.patch_size
    if vae_scale_factor is None:
        vae_scale_factor = WAN5B_SPATIAL_DOWNSAMPLE
    vae_scale_factor = int(vae_scale_factor)
    temporal_downsample = vae.config.scale_factor_temporal
    if temporal_downsample in (None, 0):
        temporal_downsample = WAN5B_TEMPORAL_DOWNSAMPLE
    temporal_downsample = int(temporal_downsample)
    latent_channels = getattr(
        getattr(transformer_config, "config", transformer_config), "in_channels", None
    )
    if latent_channels is None:
        latent_channels = getattr(transformer_config, "in_channels", WAN5B_CHANNELS)
    latent_channels = int(latent_channels)

    latent_t = ((t - 1) // temporal_downsample) + 1
    latent_h = h // vae_scale_factor
    latent_w = w // vae_scale_factor

    vae.enable_tiling()
    video_processor = VideoProcessor(vae_scale_factor=vae_scale_factor)

    B = encoder_hidden_states.shape[0]
    batch_size = 1
    batch_indices = torch.chunk(torch.arange(B, device=device), max(1, B // batch_size))

    save_videos = getattr(args, "save_videos", True)
    video_output_dir = None
    video_filename_prefix = getattr(args, "video_filename_prefix", "wan5b")
    video_output_dir_arg = getattr(args, "video_output_dir", "./videos")
    video_fps = getattr(args, "video_fps", getattr(args, "fps", 24))
    if save_videos:
        video_output_dir = Path(video_output_dir_arg)
        video_output_dir.mkdir(parents=True, exist_ok=True)
        if not hasattr(args, "_video_save_counter"):
            setattr(args, "_video_save_counter", 0)

    def _sample_latents(batch_latent_shape):
        noise = torch.randn(
            batch_latent_shape,
            device=device,
            dtype=torch.float32,
        )
        noise = noise * noise_scale
        return noise.to(torch.bfloat16)

    if args.init_same_noise:
        input_latents = _sample_latents(
            (1, latent_channels, latent_t, latent_h, latent_w)
        )

    all_latents = []
    all_log_probs = []
    all_rewards = []
    for index, batch_idx in enumerate(batch_indices):
        batch_encoder_hidden_states = encoder_hidden_states[batch_idx]
        batch_negative_prompt_embeds = negative_prompt_embeds[batch_idx]
        batch_caption = [caption[i] for i in batch_idx.tolist()]

        if not args.init_same_noise:
            input_latents = _sample_latents(
                (1, latent_channels, latent_t, latent_h, latent_w)
            )

        progress_bar = tqdm(range(0, sample_steps), desc="Sampling Progress")
        with torch.no_grad():
            z, latents, batch_latents, batch_log_probs = run_sample_step(
                args,
                input_latents.clone(),
                progress_bar,
                sigma_schedule,
                timestep_schedule,
                scheduler,
                transformer,
                batch_encoder_hidden_states,
                batch_negative_prompt_embeds,
                True,
            )

        all_latents.append(batch_latents)
        all_log_probs.append(batch_log_probs)

        with torch.inference_mode():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                latents_mean, latents_std = _reshape_vae_latent_stats(vae, latents)
                decode_latents = latents
                if latents_mean is not None and latents_std is not None:
                    decode_latents = decode_latents / latents_std + latents_mean
                video = vae.decode(decode_latents, return_dict=False)[0]
                decoded_video = video_processor.postprocess_video(video)

        if len(decoded_video) == 0:
            continue

        video_frames = decoded_video[0]
        if not isinstance(video_frames, np.ndarray):
            video_frames = np.array(video_frames)

        color_format = getattr(args, "video_color_format", "rgb").lower()
        if video_frames.ndim == 5:
            video_frames = video_frames[0]
        if video_frames.ndim == 4 and video_frames.shape[1] in (1, 3, 4):
            video_frames = np.transpose(video_frames, (0, 2, 3, 1))
        if color_format == "bgr" and video_frames.shape[-1] == 3:
            video_frames = video_frames[..., ::-1]

        if np.issubdtype(video_frames.dtype, np.floating):
            video_frames = np.clip(video_frames, 0.0, 1.0)
        else:
            video_frames = video_frames.astype(np.float32) / 255.0

        video_path = None
        if save_videos and video_output_dir is not None:
            video_id = getattr(args, "_video_save_counter", 0)
            setattr(args, "_video_save_counter", video_id + 1)
            rank = int(os.environ.get("RANK", 0))
            video_filename = (
                f"{video_filename_prefix}_rank{rank}_sample{video_id}_idx{index}.mp4"
            )
            video_path = video_output_dir / video_filename
            export_to_video(video_frames, str(video_path), fps=video_fps)

        if args.use_videoalign and inferencer is not None:
            with torch.no_grad():
                try:
                    if video_path is not None:
                        absolute_path = os.path.abspath(str(video_path))
                        reward = inferencer.reward(
                            [absolute_path],
                            [batch_caption[0]],
                            use_norm=True,
                        )
                        reward = torch.tensor(reward[0]["MQ"]).to(device)
                    else:
                        reward = torch.tensor(0.0, device=device)
                except Exception:
                    reward = torch.tensor(-1.0, device=device)
        else:
            reward = torch.tensor(0.0, device=device)
        all_rewards.append(reward.unsqueeze(0))

    all_latents = torch.cat(all_latents, dim=0)
    all_log_probs = torch.cat(all_log_probs, dim=0)
    all_rewards = torch.cat(all_rewards, dim=0)

    return (
        all_rewards,
        all_latents,
        all_log_probs,
        sigma_schedule,
        timestep_schedule,
        scheduler,
    )


def gather_tensor(tensor):
    if not dist.is_initialized():
        return tensor
    world_size = dist.get_world_size()
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, tensor)
    return torch.cat(gathered_tensors, dim=0)


def train_one_step(
    args,
    device,
    transformer,
    vae,
    inferencer,
    optimizer,
    lr_scheduler,
    loader,
    max_grad_norm,
):
    total_loss = 0.0
    optimizer.zero_grad()
    batch = next(loader)
    if len(batch) == 4:
        (
            encoder_hidden_states,
            negative_prompt_embeds,
            caption,
            image_meta,
        ) = batch
    else:
        encoder_hidden_states, negative_prompt_embeds, caption = batch
        image_meta = None
    encoder_hidden_states = encoder_hidden_states.to(
        device=device, dtype=torch.bfloat16
    )
    negative_prompt_embeds = negative_prompt_embeds.to(
        device=device, dtype=torch.bfloat16
    )
    # device = latents.device
    if args.use_group:

        def repeat_tensor(tensor):
            if tensor is None:
                return None
            return torch.repeat_interleave(tensor, args.num_generations, dim=0)

        encoder_hidden_states = repeat_tensor(encoder_hidden_states)
        negative_prompt_embeds = repeat_tensor(negative_prompt_embeds)

        if isinstance(caption, str):
            caption = [caption] * args.num_generations
        elif isinstance(caption, list):
            caption = [item for item in caption for _ in range(args.num_generations)]
        else:
            raise ValueError(f"Unsupported caption type: {type(caption)}")

    B = encoder_hidden_states.shape[0]
    negative_prompt_embeds = negative_prompt_embeds[:B]

    (
        reward,
        all_latents,
        all_log_probs,
        sigma_schedule,
        timestep_schedule,
        scheduler,
    ) = sample_reference_model(
        args,
        device,
        transformer,
        vae,
        encoder_hidden_states,
        negative_prompt_embeds,
        caption,
        inferencer,
    )
    batch_size = all_latents.shape[0]
    timestep_values = timestep_schedule.to(all_latents.device)
    timesteps = timestep_values.unsqueeze(0).repeat(batch_size, 1)
    samples = {
        "timesteps": timesteps.detach().clone()[:, :-1],
        "latents": all_latents[:, :-1][
            :, :-1
        ],  # each entry is the latent before timestep t
        "next_latents": all_latents[:, 1:][
            :, :-1
        ],  # each entry is the latent after timestep t
        "log_probs": all_log_probs[:, :-1],
        "rewards": reward.to(torch.float32),
        "encoder_hidden_states": encoder_hidden_states,
        "negative_prompt_embeds": negative_prompt_embeds,
    }
    gathered_reward = gather_tensor(samples["rewards"])
    if dist.get_rank() == 0:
        print("gathered_reward", gathered_reward)
        with open("./reward.txt", "a") as f:
            f.write(f"{gathered_reward.mean().item()}\n")

    # 计算advantage
    if args.use_group:
        n = len(samples["rewards"]) // (args.num_generations)
        advantages = torch.zeros_like(samples["rewards"])

        for i in range(n):
            start_idx = i * args.num_generations
            end_idx = (i + 1) * args.num_generations
            group_rewards = samples["rewards"][start_idx:end_idx]
            group_mean = group_rewards.mean()
            group_std = group_rewards.std() + 1e-8
            advantages[start_idx:end_idx] = (group_rewards - group_mean) / group_std

        samples["advantages"] = advantages
    else:
        advantages = (samples["rewards"] - gathered_reward.mean()) / (
            gathered_reward.std() + 1e-8
        )
        samples["advantages"] = advantages

    perms = torch.stack(
        [torch.randperm(len(samples["timesteps"][0])) for _ in range(batch_size)]
    ).to(device)
    for key in ["timesteps", "latents", "next_latents", "log_probs"]:
        samples[key] = samples[key][
            torch.arange(batch_size).to(device)[:, None],
            perms,
        ]
    samples_batched = {k: v.unsqueeze(1) for k, v in samples.items()}
    # dict of lists -> list of dicts for easier iteration
    samples_batched_list = [
        dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
    ]
    train_timesteps = int(len(samples["timesteps"][0]) * args.timestep_fraction)

    for i, sample in list(enumerate(samples_batched_list)):
        for _ in range(train_timesteps):
            clip_range = 1e-4
            adv_clip_max = 5.0
            new_log_probs = grpo_one_step(
                args,
                sample["latents"][:, _],
                sample["next_latents"][:, _],
                sample["encoder_hidden_states"],
                sample["negative_prompt_embeds"],
                scheduler,
                transformer,
                sample["timesteps"][:, _],
                perms[i][_],
                sigma_schedule,
            )

            advantages = torch.clamp(
                sample["advantages"],
                -adv_clip_max,
                adv_clip_max,
            )

            ratio = torch.exp(new_log_probs - sample["log_probs"][:, _])

            unclipped_loss = -advantages * ratio
            clipped_loss = -advantages * torch.clamp(
                ratio,
                1.0 - clip_range,
                1.0 + clip_range,
            )
            loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss)) / (
                args.gradient_accumulation_steps * train_timesteps
            )

            loss.backward()
            avg_loss = loss.detach().clone()
            dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
            total_loss += avg_loss.item()
        if (i + 1) % args.gradient_accumulation_steps == 0:
            grad_norm = transformer.clip_grad_norm_(max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        if dist.get_rank() % 8 == 0:
            print("reward", sample["rewards"].item())
            print("ratio", ratio)
            print("advantage", sample["advantages"].item())
            print("final loss", loss.item())
        dist.barrier()
    return total_loss, grad_norm.item()


def main(args):
    torch.backends.cuda.matmul.allow_tf32 = True

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()
    initialize_sequence_parallel_state(args.sp_size)

    # If passed along, set the training seed now. On GPU...
    if args.seed is not None:
        # TODO: t within the same seq parallel group should be the same. Noise should be different.
        set_seed(args.seed + rank)
    # We use different seeds for the noise generation in each process to ensure that the noise is different in a batch.
    noise_random_generator = None

    # Handle the repository creation
    if rank <= 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    # For mixed precision training we cast all non-trainable weigths to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    inferencer = None
    if args.use_videoalign:
        from fastvideo.models.videoalign.inference import VideoVLMRewardInference

        load_from_pretrained = "./videoalign_ckpt"
        dtype = torch.bfloat16
        inferencer = VideoVLMRewardInference(
            load_from_pretrained, device=f"cuda:{device}", dtype=dtype
        )

    # reward_model =
    main_print(f"--> loading model from {args.pretrained_model_name_or_path}")
    # keep the master weight to float32

    main_print(f"--> loading model from {args.model_type}")

    transformer = load_transformer(
        args.model_type,
        args.dit_model_name_or_path,
        args.pretrained_model_name_or_path,
        torch.float32 if args.master_weight_type == "fp32" else torch.bfloat16,
    )

    main_print(
        f"  Total training parameters = {sum(p.numel() for p in transformer.parameters() if p.requires_grad) / 1e6} M"
    )
    main_print(
        f"--> Initializing FSDP with sharding strategy: {args.fsdp_sharding_startegy}"
    )
    fsdp_kwargs, no_split_modules = get_dit_fsdp_kwargs(
        transformer,
        args.fsdp_sharding_startegy,
        False,
        args.use_cpu_offload,
        args.master_weight_type,
    )

    transformer = FSDP(
        transformer,
        **fsdp_kwargs,
    )

    # reference_transformer = load_reference_model(args)
    main_print(f"--> model loaded")

    if args.gradient_checkpointing:
        apply_fsdp_checkpointing(
            transformer, no_split_modules, args.selective_checkpointing
        )

    # Set model as trainable.
    transformer.train()

    params_to_optimize = transformer.parameters()
    params_to_optimize = list(filter(lambda p: p.requires_grad, params_to_optimize))

    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
        eps=1e-8,
    )

    init_steps = 0
    main_print(f"optimizer: {optimizer}")

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=1000000,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
        last_epoch=init_steps - 1,
    )

    train_dataset = LatentDataset(args.data_json_path, args.num_latent_t, args.cfg)
    sampler = DistributedSampler(
        train_dataset,
        rank=rank,
        num_replicas=world_size,
        shuffle=True,
        seed=args.sampler_seed,
    )

    train_dataloader = DataLoader(
        train_dataset,
        sampler=sampler,
        collate_fn=latent_collate_function,
        pin_memory=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        drop_last=True,
    )

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader)
        / args.gradient_accumulation_steps
        * args.sp_size
        / args.train_sp_batch_size
    )
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    vae, autocast_type, fps = load_vae(args.model_type, args.vae_model_path)
    # vae.enable_tiling()

    print("vae scale factor", vae.config.scale_factor_spatial)
    print("vae latent mean", vae.config.latents_mean)
    print("vae latent std", vae.config.latents_std)

    if rank <= 0:
        project = args.tracker_project_name or "fastvideo"
        wandb.init(project=project, config=args)

    # Train!
    total_batch_size = (
        world_size
        * args.gradient_accumulation_steps
        / args.sp_size
        * args.train_sp_batch_size
    )
    main_print("***** Running training *****")
    main_print(f"  Num examples = {len(train_dataset)}")
    main_print(f"  Dataloader size = {len(train_dataloader)}")
    main_print(f"  Resume training from step {init_steps}")
    main_print(f"  Instantaneous batch size per device = {args.train_batch_size}")
    main_print(
        f"  Total train batch size (w. data & sequence parallel, accumulation) = {total_batch_size}"
    )
    main_print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    main_print(f"  Total optimization steps per epoch = {args.max_train_steps}")
    main_print(
        f"  Total training parameters per FSDP shard = {sum(p.numel() for p in transformer.parameters() if p.requires_grad) / 1e9} B"
    )
    # print dtype
    main_print(f"  Master weight dtype: {transformer.parameters().__next__().dtype}")

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        assert NotImplementedError("resume_from_checkpoint is not supported now.")
        # TODO

    progress_bar = tqdm(
        range(0, 100000),
        initial=init_steps,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=local_rank > 0,
    )

    loader = sp_parallel_dataloader_wrapper(
        train_dataloader,
        device,
        args.train_batch_size,
        args.sp_size,
        args.train_sp_batch_size,
    )

    step_times = deque(maxlen=100)
    # todo future
    # for i in range(init_steps):
    #    next(loader)
    for epoch in range(1):
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch)  # Crucial for distributed shuffling per epoch
        for step in range(init_steps + 1, args.max_train_steps + 1):
            start_time = time.time()
            if step % args.checkpointing_steps == 0:
                save_checkpoint(transformer, rank, args.output_dir, step, epoch)

                dist.barrier()
            loss, grad_norm = train_one_step(
                args,
                device,
                transformer,
                vae,
                inferencer,
                optimizer,
                lr_scheduler,
                loader,
                args.max_grad_norm,
            )

            step_time = time.time() - start_time
            step_times.append(step_time)
            avg_step_time = sum(step_times) / len(step_times)

            progress_bar.set_postfix(
                {
                    "loss": f"{loss:.4f}",
                    "step_time": f"{step_time:.2f}s",
                    "grad_norm": grad_norm,
                }
            )
            progress_bar.update(1)
            if rank <= 0:
                wandb.log(
                    {
                        "train_loss": loss,
                        "learning_rate": lr_scheduler.get_last_lr()[0],
                        "step_time": step_time,
                        "avg_step_time": avg_step_time,
                        "grad_norm": grad_norm,
                    },
                    step=step,
                )

    if get_sequence_parallel_state():
        destroy_sequence_parallel_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        type=str,
        default="wan2.2_ti2v",
        help="The type of model to train.",
    )
    # dataset & dataloader
    parser.add_argument("--data_json_path", type=str, required=True)
    parser.add_argument("--num_frames", type=int, default=121)
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=10,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--num_latent_t", type=int, default=31, help="Number of latent timesteps."
    )
    parser.add_argument("--group_frame", action="store_true")  # TODO
    parser.add_argument("--group_resolution", action="store_true")  # TODO

    # text encoder & vae & diffusion model
    parser.add_argument(
        "--pretrained_model_name_or_path", type=str, default="data/Wan2.2-5B"
    )
    parser.add_argument("--reference_model_path", type=str, default="data/flux")
    parser.add_argument(
        "--conditioning_image_path",
        type=str,
        default=None,
        help="Optional path to a conditioning image for I2V first frame; if set, Flux is not used.",
    )
    parser.add_argument("--dit_model_name_or_path", type=str, default=None)
    parser.add_argument(
        "--vae_model_path",
        type=str,
        default="data/Wan2.2-5B/Wan2.2_VAE.pth",
        help="vae model.",
    )
    parser.add_argument("--cache_dir", type=str, default="./cache_dir")

    # diffusion setting
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--ema_start_step", type=int, default=0)
    parser.add_argument("--cfg", type=float, default=0.1)
    parser.add_argument(
        "--precondition_outputs",
        action="store_true",
        help="Whether to precondition the outputs of the model.",
    )

    # validation & logs
    parser.add_argument("--validation_prompt_dir", type=str)
    parser.add_argument("--uncond_prompt_dir", type=str)
    parser.add_argument(
        "--validation_sampling_steps",
        type=str,
        default="64",
        help="use ',' to split multi sampling steps",
    )
    parser.add_argument(
        "--validation_guidance_scale",
        type=str,
        default="4.5",
        help="use ',' to split multi scale",
    )
    parser.add_argument("--validation_steps", type=int, default=50)
    parser.add_argument("--log_validation", action="store_true")
    parser.add_argument("--tracker_project_name", type=str, default=None)
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--resume_from_lora_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous lora checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )

    # optimizer & scheduler & Training
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=10,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--max_grad_norm", default=2.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument("--selective_checkpointing", type=float, default=1.0)
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--use_cpu_offload",
        action="store_true",
        help="Whether to use CPU offload for param & gradient & optimizer states.",
    )

    parser.add_argument("--sp_size", type=int, default=1, help="For sequence parallel")
    parser.add_argument(
        "--train_sp_batch_size",
        type=int,
        default=1,
        help="Batch size for sequence parallel training",
    )

    parser.add_argument(
        "--use_lora",
        action="store_true",
        default=False,
        help="Whether to use LoRA for finetuning.",
    )
    parser.add_argument(
        "--lora_alpha", type=int, default=256, help="Alpha parameter for LoRA."
    )
    parser.add_argument(
        "--lora_rank", type=int, default=128, help="LoRA rank parameter. "
    )
    parser.add_argument("--fsdp_sharding_startegy", default="full")

    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="uniform",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "uniform"],
    )
    parser.add_argument(
        "--logit_mean",
        type=float,
        default=0.0,
        help="mean to use when using the `'logit_normal'` weighting scheme.",
    )
    parser.add_argument(
        "--logit_std",
        type=float,
        default=1.0,
        help="std to use when using the `'logit_normal'` weighting scheme.",
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )
    # lr_scheduler
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant_with_warmup",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of cycles in the learning rate scheduler.",
    )
    parser.add_argument(
        "--lr_power",
        type=float,
        default=1.0,
        help="Power factor of the polynomial scheduler.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="Weight decay to apply."
    )
    parser.add_argument(
        "--master_weight_type",
        type=str,
        default="fp32",
        help="Weight type to use - fp32 or bf16.",
    )
    parser.add_argument(
        "--weight_path",
        type=str,
        default=704,
        help="Reward model path",
    )
    parser.add_argument(
        "--h",
        type=int,
        default=1280,
        help="video height",
    )
    parser.add_argument(
        "--w",
        type=int,
        default=121,
        help="video width",
    )
    parser.add_argument(
        "--t",
        type=int,
        default=30,
        help="video length",
    )
    parser.add_argument(
        "--sampling_steps",
        type=int,
        default=0.3,
        help="sampling steps",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=24,
        help="noise eta",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=1223627,
        help="fps of stored video",
    )
    sampling_group = parser.add_mutually_exclusive_group()
    sampling_group.add_argument(
        "--simple_sampling",
        dest="simple_sampling",
        action="store_true",
        help="Use Wan2.1-style sampling without first-frame conditioning.",
    )
    sampling_group.add_argument(
        "--no_simple_sampling",
        dest="simple_sampling",
        action="store_false",
        help="Use the original scheduler + first-frame conditioning.",
    )
    sampling_group.set_defaults(simple_sampling=True)
    parser.add_argument(
        "--video_output_dir",
        type=str,
        default="./videos",
        help="Directory where generated videos are stored.",
    )
    parser.add_argument(
        "--sampler_seed",
        type=int,
        default=None,
        help="seed of sampler",
    )
    parser.add_argument(
        "--use_group",
        action="store_true",
        default=False,
        help="whether to use group",
    )
    parser.add_argument(
        "--num_generations",
        type=int,
        default=8,
        help="num_generations per prompt",
    )
    parser.add_argument(
        "--use_videoalign",
        action="store_true",
        default=False,
        help="whether to use group",
    )
    parser.add_argument(
        "--init_same_noise",
        action="store_true",
        default=False,
        help="whether to use the same noise",
    )
    parser.add_argument(
        "--timestep_fraction",
        type=float,
        default=1.0,
        help="timestep_fraction",
    )
    parser.add_argument(
        "--cfg_infer",
        type=float,
        default=5.0,
        help="cfg",
    )
    parser.add_argument(
        "--compare_scheduler",
        action="store_true",
        help="Compare custom sampler updates against UniPC scheduler.step and log differences.",
    )
    parser.add_argument(
        "--compare_scheduler_tolerance",
        type=float,
        default=1e-4,
        help="Tolerance for max absolute difference when comparing against UniPC scheduler.",
    )
    parser.add_argument(
        "--compare_scheduler_max_steps",
        type=int,
        default=4,
        help="Number of steps to compare when --compare_scheduler is enabled.",
    )
    parser.add_argument(
        "--shift",
        type=float,
        default=1.0,
        help="sampling shift",
    )

    args = parser.parse_args()
    main(args)
