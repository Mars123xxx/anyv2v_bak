import argparse
import logging

import numpy as np
import torch
from PIL import Image
from diffusers import AutoencoderKL, EulerDiscreteScheduler
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

from diffusers_sv3d import SV3DUNetSpatioTemporalConditionModel, StableVideo3DDiffusionPipeline

SV3D_DIFFUSERS = "chenguolin/sv3d-diffusers"



def ddim_inversion():
    # ? EulerDiscreteScheduler的inverseScheduler怎么实现 ? -> 有没有
    # ddim_inversion怎么给sv3d反演

    # invert输入参数 仰角、方位角、多视角图像的latents -> 保证输入符合逻辑
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

    pipeline = StableVideo3DDiffusionPipeline(
        image_encoder=image_encoder, feature_extractor=feature_extractor,
        unet=unet, vae=vae,
        scheduler=scheduler,
    )


    # 模型预参数准备
    num_frames, sv3d_res = 21, 576
    elevations_deg = [args.elevation] * num_frames
    polars_rad = [np.deg2rad(90 - e) for e in elevations_deg]
    azimuths_deg = np.linspace(0, 360, num_frames + 1)[1:] % 360
    azimuths_rad = [np.deg2rad((a - azimuths_deg[-1]) % 360) for a in azimuths_deg]
    azimuths_rad[:-1].sort()
    pipeline.to("cuda")
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.float16 if args.half_precision else torch.float32, enabled=True):
            image = Image.open(args.image_path)
            image.load()  # required for `.split()`
            if len(image.split()) == 4:  # RGBA
                input_image = Image.new("RGB", image.size, (255, 255, 255))  # pure white bg
                input_image.paste(image, mask=image.split()[3])  # 3rd is the alpha channel
            else:
                input_image = image

            ddim_latents = pipeline.invert(
                input_image.resize((sv3d_res, sv3d_res)),
                height=sv3d_res,
                width=sv3d_res,
                num_frames=num_frames,
                decode_chunk_size=8,  # smaller to save memory
                polars_rad=polars_rad,
                azimuths_rad=azimuths_rad,
                generator=torch.manual_seed(args.seed) if args.seed >= 0 else None,
            )
            print(ddim_latents)
            ddim_latents = ddim_latents[0]
            return ddim_latents


if __name__ == "__main__":
    d_l = ddim_inversion()
    print(d_l)