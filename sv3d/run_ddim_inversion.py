import argparse

import numpy as np
import torch
from PIL import Image
from diffusers import AutoencoderKL, EulerDiscreteScheduler,DDIMScheduler,DDIMInverseScheduler
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

from diffusers_sv3d import SV3DUNetSpatioTemporalConditionModel, StableVideo3DDiffusionPipeline

from sv3d.utils import load_video_frames

SV3D_DIFFUSERS = "chenguolin/sv3d-diffusers"

# 生成inversion的latents
def inversion(image,inverse_scheduler,pipe,frame_list,config):
    # 这里由于EulerDiscreteScheduler本身并没有提供inverse操作获得latents
    # 故这里的inverse_scheduler暂时搁置
    pipe.scheduler = inverse_scheduler

    video_latents_at_0 = pipe.encode_vae_video(
        frame_list,
        device='cuda',
        height=config.height,
        width=config.width,
    )

    ddim_latents = pipe.invert(
        image.resize((config.width, config.height)),
        height=config.height,
        width=config.width,
        num_frames=config.num_frames,
        decode_chunk_size=8,  # smaller to save memory
        polars_rad=config.polars_rad,
        azimuths_rad=config.azimuths_rad,
        generator=torch.manual_seed(config.seed) if config.seed >= 0 else None,
        latents=video_latents_at_0
    )  # [b, num_inference_steps, c, num_frames, h, w]

    ddim_latents = ddim_latents[0]  # [num_inference_steps, c, num_frames, h, w]
    return ddim_latents

def sampling(image,pipe,scheduler,ddim_latents_at_T,config):
    # 这里scheduler是xt -> x0的调度器
    # 目前为 -> EulerDiscreteSchedulers
    pipe.scheduler = scheduler
    reconstructed_video = pipe(
        image.resize((config.width, config.height)),
        height=config.height,
        width=config.width,
        num_frames=config.num_frames,
        decode_chunk_size=8,  # smaller to save memory
        polars_rad=config.polars_rad,
        azimuths_rad=config.azimuths_rad,
        generator=torch.manual_seed(config.seed) if config.seed >= 0 else None,
        latents=ddim_latents_at_T,
        return_dict=True
    ).frames[0]
    return reconstructed_video


def main():
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
    num_frames, sv3d_res = 9, 576
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


            config = {
                'height':576,
                'width':576,
                'num_frames':21,
                'azimuths_rad':azimuths_rad,
                'polars_rad':polars_rad,
                'seed':args.seed,
                'video_frames_path':'../demo/inversion_test/img_1',
                'image_size':[576,576]
            }

            _, frame_list = load_video_frames(config.video_frames_path, config.num_frames, config.image_size)

            # 更换scheduler
            scheduler = DDIMInverseScheduler.from_config(pipeline.scheduler.config)
            # 调用inversion方法
            latents = inversion(input_image,pipeline,scheduler,frame_list, config)

            scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
            # 调用sampling方法
            reconstruted_video = sampling(input_image,pipeline,scheduler,latents,config)
            return reconstruted_video


if __name__ == "__main__":
    d_l = main()
    print(d_l)