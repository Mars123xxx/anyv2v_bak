import os
import sys
from pathlib import Path

import numpy as np
import torch
import argparse
import logging
from omegaconf import OmegaConf
from PIL import Image
import json

SV3D_DIFFUSERS = "chenguolin/sv3d-diffusers"

# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "~/.cache/huggingface"

# HF imports
from diffusers import (
    DDIMInverseScheduler,
    DDIMScheduler, AutoencoderKL, EulerDiscreteScheduler,
)
from diffusers.utils import load_image, export_to_video, export_to_gif
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

from diffusers_sv3d import SV3DUNetSpatioTemporalConditionModel, StableVideo3DDiffusionPipeline

# Project imports
from utils import (
    seed_everything,
    load_video_frames,
    convert_video_to_frames,
    load_ddim_latents_at_T,
    load_ddim_latents_at_t,
)

from pnp_utils import (
    register_time,
    register_conv_injection,
    register_spatial_attention_pnp,
    register_temp_attention_pnp,
)


def init_pnp(pipe, scheduler, config):
    conv_injection_t = int(config.n_steps * config.pnp_f_t)
    spatial_attn_qk_injection_t = int(config.n_steps * config.pnp_spatial_attn_t)
    temp_attn_qk_injection_t = int(config.n_steps * config.pnp_temp_attn_t)
    conv_injection_timesteps = scheduler.timesteps[:conv_injection_t] if conv_injection_t >= 0 else []
    spatial_attn_qk_injection_timesteps = (
        scheduler.timesteps[:spatial_attn_qk_injection_t] if spatial_attn_qk_injection_t >= 0 else []
    )
    temp_attn_qk_injection_timesteps = (
        scheduler.timesteps[:temp_attn_qk_injection_t] if temp_attn_qk_injection_t >= 0 else []
    )
    register_conv_injection(pipe, conv_injection_timesteps)
    register_spatial_attention_pnp(pipe, spatial_attn_qk_injection_timesteps)
    register_temp_attention_pnp(pipe, temp_attn_qk_injection_timesteps)

    logger = logging.getLogger(__name__)
    logger.debug(f"conv_injection_t: {conv_injection_t}")
    logger.debug(f"spatial_attn_qk_injection_t: {spatial_attn_qk_injection_t}")
    logger.debug(f"temp_attn_qk_injection_t: {temp_attn_qk_injection_t}")
    logger.debug(f"conv_injection_timesteps: {conv_injection_timesteps}")
    logger.debug(f"spatial_attn_qk_injection_timesteps: {spatial_attn_qk_injection_timesteps}")
    logger.debug(f"temp_attn_qk_injection_timesteps: {temp_attn_qk_injection_timesteps}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="out/", type=str, help="Output filepath")
    parser.add_argument("--image_path", default="assets/bubble_mart_blue.png", type=str, help="Image filepath")
    parser.add_argument("--elevation", default=10, type=float, help="Camera elevation of the input image")
    parser.add_argument("--half_precision", action="store_true", help="Use fp16 half precision")
    parser.add_argument("--seed", default=-1, type=int, help="Random seed")
    args = parser.parse_args()

    unet = SV3DUNetSpatioTemporalConditionModel.from_pretrained(SV3D_DIFFUSERS, subfolder="unet")
    vae = AutoencoderKL.from_pretrained(SV3D_DIFFUSERS, subfolder="vae")
    scheduler = EulerDiscreteScheduler.from_pretrained(SV3D_DIFFUSERS, subfolder="scheduler")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(SV3D_DIFFUSERS, subfolder="image_encoder")
    feature_extractor = CLIPImageProcessor.from_pretrained(SV3D_DIFFUSERS, subfolder="feature_extractor")
    # Initialize the pipeline
    pipe = StableVideo3DDiffusionPipeline(
        image_encoder=image_encoder, feature_extractor=feature_extractor,
        unet=unet, vae=vae,
        scheduler=scheduler,
    )
    num_frames, sv3d_res = 21, 576
    elevations_deg = [args.elevation] * num_frames
    polars_rad = [np.deg2rad(90 - e) for e in elevations_deg]
    azimuths_deg = np.linspace(0, 360, num_frames + 1)[1:] % 360
    azimuths_rad = [np.deg2rad((a - azimuths_deg[-1]) % 360) for a in azimuths_deg]
    azimuths_rad[:-1].sort()
    pipe.to("cuda")


    # Initialize the DDIM scheduler
    ddim_scheduler = pipe.scheduler
    ddim_scheduler.set_timesteps(50)

    ddim_latents_at_t = load_ddim_latents_at_t(
        ddim_scheduler.timesteps[0], ddim_latents_path='ddim_latents'
    )

    print(ddim_latents_at_t)
    print('----------->', ddim_scheduler.timesteps[0])
    print(ddim_latents_at_t)

    random_latents = torch.randn_like(ddim_latents_at_t)

    # Blend the latents
    random_latents = torch.randn_like(ddim_latents_at_t)
    mixed_latents = random_latents * 0.0 + ddim_latents_at_t * (1 - 0.0)

    config = {
        'n_steps': 50,
        'pnp_f_t': 0.2,
        'pnp_spatial_attn_t': 0.2,
        'pnp_temp_attn_t': 0.5
    }

    init_pnp(pipe, ddim_scheduler, config)

    # Edit video
    # 补充实现pipe的register_modules方法
    # pipe.register_modules(scheduler=ddim_scheduler)

    # -> run_group_pnp_edit.py 示例 --> 补充实现pipe的sample_with_pnp方法
    # edited_video = pipe.sample_with_pnp(
    #     prompt=config.editing_prompt,
    #     image=edited_1st_frame,
    #     height=config.image_size[1],
    #     width=config.image_size[0],
    #     num_frames=config.n_frames,
    #     num_inference_steps=config.n_steps,
    #     guidance_scale=config.cfg,
    #     negative_prompt=config.editing_negative_prompt,
    #     target_fps=config.target_fps,
    #     latents=mixed_latents,
    #     generator=torch.manual_seed(config.seed),
    #     return_dict=True,
    #     ddim_init_latents_t_idx=ddim_init_latents_t_idx,
    #     ddim_inv_latents_path=config.ddim_latents_path,
    #     ddim_inv_prompt=config.ddim_inv_prompt,
    #     ddim_inv_1st_frame=src_1st_frame,
    # ).frames[0]

    #######
    # edited_video -> 保存为gif或者mp4导出



if __name__ == "__main__":
    main()
