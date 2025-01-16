import os

from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import _append_dims
from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import *
from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import retrieve_timesteps
from diffusers.utils import (
    BaseOutput
)
from typing import Any, Dict, List, Optional, Tuple, Union


def _convert_pt_to_pil(image: Union[torch.Tensor, List[torch.Tensor]]):
    if isinstance(image, list) and isinstance(image[0], torch.Tensor):
        image = torch.cat(image, 0)

    if isinstance(image, torch.Tensor):
        if image.ndim == 3:
            image = image.unsqueeze(0)

        image_numpy = VaeImageProcessor.pt_to_numpy(image)
        image_pil = VaeImageProcessor.numpy_to_pil(image_numpy)
        image = image_pil

    return image

def _center_crop_wide(
        image: Union[torch.Tensor, List[torch.Tensor], PIL.Image.Image, List[PIL.Image.Image]],
        resolution: Tuple[int, int]
):
    # First convert the images to PIL in case they are float tensors (only relevant for tests now).
    image = _convert_pt_to_pil(image)

    if isinstance(image, list):
        scale = min(image[0].size[0] / resolution[0], image[0].size[1] / resolution[1])
        image = [u.resize((round(u.width // scale), round(u.height // scale)), resample=PIL.Image.BOX) for u in image]

        # center crop
        x1 = (image[0].width - resolution[0]) // 2
        y1 = (image[0].height - resolution[1]) // 2
        image = [u.crop((x1, y1, x1 + resolution[0], y1 + resolution[1])) for u in image]
        return image
    else:
        scale = min(image.size[0] / resolution[0], image.size[1] / resolution[1])
        image = image.resize((round(image.width // scale), round(image.height // scale)), resample=PIL.Image.BOX)
        x1 = (image.width - resolution[0]) // 2
        y1 = (image.height - resolution[1]) // 2
        image = image.crop((x1, y1, x1 + resolution[0], y1 + resolution[1]))
        return image


@dataclass
class StableVideoDiffusionInversionPipelineOutput(BaseOutput):
    """
    Output class for Stable Diffusion pipelines.

    Args:
        latents (`torch.FloatTensor`)
            inverted latents tensor
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `num_timesteps * batch_size` or numpy array of shape `(num_timesteps,
            batch_size, height, width, num_channels)`. PIL images or numpy array present the denoised images of the
            diffusion pipeline.
    """

    inverted_latents: torch.FloatTensor
    # images: Union[List[PIL.Image.Image], np.ndarray] # TODO: we can return the noisy video.


# Copied from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion.StableVideoDiffusionPipeline
class StableVideo3DDiffusionPipeline(StableVideoDiffusionPipeline):
    def __init__(
        self,
        vae: AutoencoderKLTemporalDecoder,
        image_encoder: CLIPVisionModelWithProjection,
        unet: UNetSpatioTemporalConditionModel,
        scheduler: EulerDiscreteScheduler, # xt -> x0
        feature_extractor: CLIPImageProcessor,
    ):
        super().__init__(
            vae,
            image_encoder,
            unet,
            scheduler,
            feature_extractor,
        )

    def encode_vae_video(
            self,
            video: List[PIL.Image.Image],
            device,
            height: int = 576,
            width: int = 576,
    ):
        # video is a list of PIL images
        # [batch*frames] while batch is always 1  TODO: generalize to batch > 1
        dtype = next(self.vae.parameters()).dtype
        n_frames = len(video)
        video_latents = []
        for i in range(0, n_frames):
            frame = video[i]
            resized_frame = _center_crop_wide(frame, (width, height))
            frame = self.image_processor.preprocess(resized_frame)
            frame = frame.to(device=device, dtype=dtype)
            image_latents = self.vae.encode(frame).latent_dist.sample()  # [1, channels, height, width]
            image_latents = image_latents * self.vae.config.scaling_factor
            logger.debug(f"image_latents.shape: {image_latents.shape}")
            image_latents = image_latents.squeeze(0)  # [channels, height, width]
            video_latents.append(image_latents)
        video_latents = torch.stack(video_latents)  # [batch*frames, channels, height, width]
        video_latents = video_latents.reshape(1, n_frames, *video_latents.shape[1:])
        video_latents = video_latents.permute(0, 2, 1, 3, 4)  # [batch, channels, frames, height, width]
        video_latents = video_latents.to(device=device, dtype=dtype)
        # [batch, channels, frames, height, width]
        return video_latents

    def _get_add_time_ids(
        self,
        noise_aug_strength: float,
        polars_rad: List[float],
        azimuths_rad: List[float],
        dtype: torch.dtype,
        batch_size: int,
        num_videos_per_prompt: int,
        do_classifier_free_guidance: bool,
    ):
        cond_aug = torch.tensor([noise_aug_strength]*len(polars_rad), dtype=dtype).repeat(batch_size * num_videos_per_prompt, 1)
        polars_rad = torch.tensor(polars_rad, dtype=dtype).repeat(batch_size * num_videos_per_prompt, 1)
        azimuths_rad = torch.tensor(azimuths_rad, dtype=dtype).repeat(batch_size * num_videos_per_prompt, 1)

        if do_classifier_free_guidance:
            cond_aug = torch.cat([cond_aug, cond_aug])
            polars_rad = torch.cat([polars_rad, polars_rad])
            azimuths_rad = torch.cat([azimuths_rad, azimuths_rad])

        add_time_ids = [cond_aug, polars_rad, azimuths_rad]

        return add_time_ids

    @torch.no_grad()
    def __call__(
        self,
        image: Union[PIL.Image.Image, List[PIL.Image.Image], torch.Tensor],

        polars_rad: List[float],
        azimuths_rad: List[float],
        triangle_cfg_scaling: bool = True,

        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: Optional[int] = None,
        num_inference_steps: int = 25,
        sigmas: Optional[List[float]] = None,
        min_guidance_scale: float = 1.0,
        max_guidance_scale: float = 2.5,
        noise_aug_strength: float = 1e-5,
        decode_chunk_size: Optional[int] = None,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        return_dict: bool = True,
    ):
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        num_frames = num_frames if num_frames is not None else self.unet.config.num_frames
        decode_chunk_size = decode_chunk_size if decode_chunk_size is not None else num_frames

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(image, height, width)

        # 2. Define call parameters
        if isinstance(image, PIL.Image.Image):
            batch_size = 1
        elif isinstance(image, list):
            batch_size = len(image)
        else:
            batch_size = image.shape[0]
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        self._guidance_scale = max_guidance_scale

        # 3. Encode input image
        image_embeddings = self._encode_image(image, device, num_videos_per_prompt, self.do_classifier_free_guidance)

        # 4. Encode input image using VAE
        image = self.video_processor.preprocess(image, height=height, width=width).to(device)
        noise = randn_tensor(image.shape, generator=generator, device=device, dtype=image.dtype)
        image = image + noise_aug_strength * noise

        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
        if needs_upcasting:
            self.vae.to(dtype=torch.float32)

        image_latents = self._encode_vae_image(
            image,
            device=device,
            num_videos_per_prompt=num_videos_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
        )
        image_latents = image_latents.to(image_embeddings.dtype)

        # cast back to fp16 if needed
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)

        # Repeat the image latents for each frame so we can concatenate them with the noise
        # image_latents [batch, channels, height, width] ->[batch, num_frames, channels, height, width]
        image_latents = image_latents.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)

        # 5. Get Added Time IDs
        added_time_ids = self._get_add_time_ids(
            noise_aug_strength,
            polars_rad,
            azimuths_rad,
            image_embeddings.dtype,
            batch_size,
            num_videos_per_prompt,
            self.do_classifier_free_guidance,
        )
        added_time_ids = [a.to(device) for a in added_time_ids]  # (cond_aug, polars_rad, azimuths_rad)

        # 6. Prepare timesteps
        # timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, None, sigmas)

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 7. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_frames,
            num_channels_latents,
            height,
            width,
            image_embeddings.dtype,
            device,
            generator,
            latents,
        )
        print(latents.shape)

        # 8. Prepare guidance scale
        if triangle_cfg_scaling:
            # Triangle CFG scaling; the last view is input condition
            guidance_scale = torch.cat([
                torch.linspace(min_guidance_scale, max_guidance_scale, num_frames//2 + 1)[1:].unsqueeze(0),
                torch.linspace(max_guidance_scale, min_guidance_scale, num_frames - num_frames//2 + 1)[1:].unsqueeze(0)
            ], dim=-1)
        else:
            guidance_scale = torch.linspace(min_guidance_scale, max_guidance_scale, num_frames).unsqueeze(0)
        guidance_scale = guidance_scale.to(device, latents.dtype)
        guidance_scale = guidance_scale.repeat(batch_size * num_videos_per_prompt, 1)
        guidance_scale = _append_dims(guidance_scale, latents.ndim)

        self._guidance_scale = guidance_scale

        # 9. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # Concatenate image_latents over channels dimension
                latent_model_input = torch.cat([latent_model_input, image_latents], dim=2)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=image_embeddings,
                    added_time_ids=added_time_ids,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample

                if output_type == 'latent':
                    os.makedirs('ddim_latents', exist_ok=True)
                    torch.save(
                        latents.detach().clone(),
                        os.path.join('ddim_latents', f"ddim_latents_{t}.pt"),
                    )

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        if not output_type == "latent":
            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
            frames = self.decode_latents(latents, num_frames, decode_chunk_size)
            frames = self.video_processor.postprocess_video(video=frames, output_type=output_type)
        else:
            print('latents output')
            frames = latents

        self.maybe_free_model_hooks()

        if not return_dict:
            return frames

        return StableVideoDiffusionPipelineOutput(frames=frames)

    @torch.no_grad()
    def invert(
            self,
            image: Union[PIL.Image.Image, List[PIL.Image.Image], torch.Tensor],
            polars_rad: List[float],
            azimuths_rad: List[float],
            triangle_cfg_scaling: bool = True,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_frames: Optional[int] = None,
            num_inference_steps: int = 50,
            sigmas: Optional[List[float]] = None,
            min_guidance_scale: float = 1.0,
            max_guidance_scale: float = 2.5,
            noise_aug_strength: float = 1e-5,
            decode_chunk_size: Optional[int] = None,
            num_videos_per_prompt: Optional[int] = 1,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.Tensor] = None,
            output_type: Optional[str] = "pil",
            callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],
            return_dict: bool = True,
        ):
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        num_frames = num_frames if num_frames is not None else self.unet.config.num_frames
        decode_chunk_size = decode_chunk_size if decode_chunk_size is not None else num_frames

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(image, height, width)

        # 2. Define call parameters
        if isinstance(image, PIL.Image.Image):
            batch_size = 1
        elif isinstance(image, list):
            batch_size = len(image)
        else:
            batch_size = image.shape[0]
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        self._guidance_scale = max_guidance_scale

        # 3. Encode input image
        image_embeddings = self._encode_image(image, device, num_videos_per_prompt, self.do_classifier_free_guidance)

        # 4. Encode input image using VAE
        image = self.video_processor.preprocess(image, height=height, width=width).to(device)
        noise = randn_tensor(image.shape, generator=generator, device=device, dtype=image.dtype)
        image = image + noise_aug_strength * noise

        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
        if needs_upcasting:
            self.vae.to(dtype=torch.float32)

        image_latents = self._encode_vae_image(
            image,
            device=device,
            num_videos_per_prompt=num_videos_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
        )
        image_latents = image_latents.to(image_embeddings.dtype)

        # cast back to fp16 if needed
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)

        # Repeat the image latents for each frame so we can concatenate them with the noise
        # image_latents [batch, channels, height, width] ->[batch, num_frames, channels, height, width]
        image_latents = image_latents.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)

        # 5. Get Added Time IDs
        added_time_ids = self._get_add_time_ids(
            noise_aug_strength,
            polars_rad,
            azimuths_rad,
            image_embeddings.dtype,
            batch_size,
            num_videos_per_prompt,
            self.do_classifier_free_guidance,
        )
        added_time_ids = [a.to(device) for a in added_time_ids]  # (cond_aug, polars_rad, azimuths_rad)

        # 6. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, None, sigmas)

        # 7. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_frames,
            num_channels_latents,
            height,
            width,
            image_embeddings.dtype,
            device,
            generator,
            latents,
        )
        print(latents.shape)

        print('timesteps -->',timesteps)

        # 8. Prepare guidance scale
        if triangle_cfg_scaling:
            # Triangle CFG scaling; the last view is input condition
            guidance_scale = torch.cat([
                torch.linspace(min_guidance_scale, max_guidance_scale, num_frames // 2 + 1)[1:].unsqueeze(0),
                torch.linspace(max_guidance_scale, min_guidance_scale, num_frames - num_frames // 2 + 1)[1:].unsqueeze(
                    0)
            ], dim=-1)
        else:
            guidance_scale = torch.linspace(min_guidance_scale, max_guidance_scale, num_frames).unsqueeze(0)
        guidance_scale = guidance_scale.to(device, latents.dtype)
        guidance_scale = guidance_scale.repeat(batch_size * num_videos_per_prompt, 1)
        guidance_scale = _append_dims(guidance_scale, latents.ndim)

        self._guidance_scale = guidance_scale

        inverted_latents = []

        # 9. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # Concatenate image_latents over channels dimension
                latent_model_input = torch.cat([latent_model_input, image_latents], dim=2)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=image_embeddings,
                    added_time_ids=added_time_ids,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample

                inverted_latents.append(latents.detach().clone())

                os.makedirs('ddim_latents', exist_ok=True)
                torch.save(
                    latents.detach().clone(),
                    os.path.join('ddim_latents', f"ddim_latents_{t}.pt"),
                )
                print(f"saved noisy latents at t={t} to ddim_latents")

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        # assert len(inverted_latents) == len(timesteps)
        inverted_latents = torch.stack(list(reversed(inverted_latents)), 1)

        if not return_dict:
            return inverted_latents

        # if output_type == "latent":
        #     return I2VGenXLPipelineOutput(frames=latents)
        #
        # video_tensor = self.decode_latents(latents, decode_chunk_size=decode_chunk_size)
        # video = tensor2vid(video_tensor, self.image_processor, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        return StableVideoDiffusionInversionPipelineOutput(inverted_latents=inverted_latents)

