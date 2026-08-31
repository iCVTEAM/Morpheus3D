from contextlib import contextmanager
from dataclasses import dataclass

import os
import sys
from PIL import Image
import torchvision.transforms.functional as TF

import torch
import torch.nn.functional as F
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
)
from diffusers.utils.import_utils import is_xformers_available

from extern.zero123 import Zero123Pipeline

import threestudio
from threestudio.utils.base import BaseModule
from threestudio.utils.misc import C, cleanup, parse_version, enable_gradient
from threestudio.utils.typing import *
import torch_dct as dct


@threestudio.register("zero123-sds-guidance")
class Zero123SDSGuidance(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        pretrained_model_name_or_path: str = "./diffusion_ckpt/zero123-xl-diffusers-base"
        enable_memory_efficient_attention: bool = False
        enable_sequential_cpu_offload: bool = False
        enable_attention_slicing: bool = False
        enable_channels_last_format: bool = False
        guidance_scale: float = 7.5
        grad_clip: Optional[Any] = None
        half_precision_weights: bool = True

        min_step_percent: float = 0.02
        max_step_percent: float = 0.98

        view_dependent_prompting: bool = True
        camera_condition_type: str = "extrinsics"

        cond_image_path: str = ""
        cond_elevation_deg: float = 0.0
        cond_azimuth_deg: float = 0.0
        cond_camera_distance: float = 1.2

        cal_sds_in_freq: bool = False
        use_freq_in_low: bool = False
        freq_bound_ratio: float = 0.5

        vis_sds: bool = False

    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading Zero123...")

        self.weights_dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )

        sys.path.append("extern/")

        pipe_kwargs = {
            "safety_checker": None,
            "requires_safety_checker": False,
            "variant": "fp16" if self.cfg.half_precision_weights else None,
            "torch_dtype": self.weights_dtype,
        }

        @dataclass
        class SubModules:
            pipe: Zero123Pipeline

        pipe = Zero123Pipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path, **pipe_kwargs,
        ).to(self.device)
        self.prepare_pipe(pipe)

        if self.cfg.camera_condition_type in ["extrinsics", "mvp"]:
            self.camera_embedding_dim = 16
        elif self.cfg.camera_condition_type == "spherical":
            self.camera_embedding_dim = 4
        else:
            raise ValueError("Invalid camera condition type!")

        self.submodules = SubModules(pipe=pipe)

        if self.cfg.enable_memory_efficient_attention:
            if parse_version(torch.__version__) >= parse_version("2"):
                threestudio.info(
                    "PyTorch2.0 uses memory efficient attention by default."
                )
            elif not is_xformers_available():
                threestudio.warn(
                    "xformers is not available, memory efficient attention is not enabled."
                )
            else:
                self.pipe.enable_xformers_memory_efficient_attention()

        if self.cfg.enable_sequential_cpu_offload:
            self.pipe.enable_sequential_cpu_offload()

        if self.cfg.enable_attention_slicing:
            self.pipe.enable_attention_slicing(1)

        if self.cfg.enable_channels_last_format:
            self.pipe.unet.to(memory_format=torch.channels_last)

        self.scheduler = DDPMScheduler.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            subfolder="scheduler",
            torch_dtype=self.weights_dtype,
        )

        self.pipe.scheduler = self.scheduler

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.set_min_max_steps()  # set to default value

        self.alphas: Float[Tensor, "..."] = self.scheduler.alphas_cumprod.to(
            self.device
        )

        self.grad_clip_val: Optional[float] = None

        self.prepare_image_embeddings()

        threestudio.info(f"Loaded Zero123!")

    @property
    def pipe(self):
        return self.submodules.pipe

    @property
    def unet(self):
        return self.submodules.pipe.unet

    @property
    def vae(self):
        return self.submodules.pipe.vae

    def prepare_pipe(self, pipe: Zero123Pipeline):
        cleanup()

        pipe.image_encoder.eval()
        pipe.vae.eval()
        pipe.unet.eval()
        pipe.clip_camera_projection.eval()

        enable_gradient(pipe.image_encoder, enabled=False)
        enable_gradient(pipe.vae, enabled=False)
        enable_gradient(pipe.unet, enabled=False)
        enable_gradient(pipe.clip_camera_projection, enabled=False)

        # disable progress bar
        pipe.set_progress_bar_config(disable=True)

    def prepare_image_embeddings(self) -> None:
        if not os.path.exists(self.cfg.cond_image_path):
            raise RuntimeError(
                f"Condition image not found at {self.cfg.cond_image_path}"
            )
        image = Image.open(self.cfg.cond_image_path).convert("RGBA").resize((256, 256))
        image = (
            TF.to_tensor(image)
            .unsqueeze(0)
            .to(device=self.device, dtype=self.weights_dtype)
        )
        # rgba -> rgb, apply white background
        image = image[:, :3] * image[:, 3:4] + (1 - image[:, 3:4])

        with torch.no_grad():
            self.clip_image_embeddings: Float[
                Tensor, "1 1 D"
            ] = self.extract_clip_image_embeddings(image)

            # encoded latents should be multiplied with vae.config.scaling_factor
            # but zero123 was not trained this way
            self.image_latents: Float[Tensor, "1 4 Hl Wl"] = (
                self.vae_encode(self.pipe.vae, image * 2.0 - 1.0, mode=True)
                / self.pipe.vae.config.scaling_factor
            )

    def extract_clip_image_embeddings(
        self, images: Float[Tensor, "B 3 H W"]
    ) -> Float[Tensor, "B 1 D"]:
        # expect images in [0, 1]
        images_pil = [TF.to_pil_image(image) for image in images]
        images_processed = self.pipe.feature_extractor(
            images=images_pil, return_tensors="pt"
        ).pixel_values.to(device=self.device, dtype=self.weights_dtype)
        clip_image_embeddings = self.pipe.image_encoder(images_processed).image_embeds
        return clip_image_embeddings.to(images.dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def set_min_max_steps(self, min_step_percent=0.02, max_step_percent=0.98):
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)

    @torch.cuda.amp.autocast(enabled=False)
    def forward_unet(
        self,
        unet: UNet2DConditionModel,
        latents: Float[Tensor, "..."],
        t: Int[Tensor, "..."],
        encoder_hidden_states: Float[Tensor, "..."],
        class_labels: Optional[Float[Tensor, "..."]] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        down_block_additional_residuals: Optional[Float[Tensor, "..."]] = None,
        mid_block_additional_residual: Optional[Float[Tensor, "..."]] = None,
        velocity_to_epsilon: bool = False,
    ) -> Float[Tensor, "..."]:
        input_dtype = latents.dtype
        pred = unet(
            latents.to(unet.dtype),
            t.to(unet.dtype),
            encoder_hidden_states=encoder_hidden_states.to(unet.dtype),
            class_labels=class_labels,
            cross_attention_kwargs=cross_attention_kwargs,
            down_block_additional_residuals=down_block_additional_residuals,
            mid_block_additional_residual=mid_block_additional_residual,
        ).sample
        if velocity_to_epsilon:
            pred = latents * self.sigmas[t].view(-1, 1, 1, 1) + pred * self.alphas[
                t
            ].view(-1, 1, 1, 1)
        return pred.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def decode_latents(
        self,
        latents: Float[Tensor, "B 4 H W"],
        latent_height: int = 64,
        latent_width: int = 64,
    ) -> Float[Tensor, "B 3 512 512"]:
        input_dtype = latents.dtype
        latents = F.interpolate(
            latents, (latent_height, latent_width), mode="bilinear", align_corners=False
        )
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents.to(self.weights_dtype)).sample
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return image.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def vae_encode(
        self, vae: AutoencoderKL, imgs: Float[Tensor, "B 3 H W"], mode=False
    ) -> Float[Tensor, "B 4 Hl Wl"]:
        # expect input in [-1, 1]
        input_dtype = imgs.dtype
        posterior = vae.encode(imgs.to(vae.dtype)).latent_dist
        if mode:
            latents = posterior.mode()
        else:
            latents = posterior.sample()
        latents = latents * vae.config.scaling_factor
        return latents.to(input_dtype)

    @contextmanager
    def disable_unet_class_embedding(self, unet: UNet2DConditionModel):
        class_embedding = unet.class_embedding
        try:
            unet.class_embedding = None
            yield unet
        finally:
            unet.class_embedding = class_embedding

    def compute_grad_sds(
        self,
        latents: Float[Tensor, "B 4 Hl Wl"],
        image_camera_embeddings: Float[Tensor, "B 1 D"],
        camera_condition: Float[Tensor, "B ..."],
    ):
        B = latents.shape[0]

        with torch.no_grad():
            # random timestamp
            t = torch.randint(
                self.min_step,
                self.max_step + 1,
                [B],
                dtype=torch.long,
                device=self.device,
            )
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
            with self.disable_unet_class_embedding(self.unet) as unet:
                noise_pred_pretrain = self.forward_unet(
                    unet,
                    torch.cat(
                        [
                            latent_model_input,
                            torch.cat(
                                [
                                    self.image_latents.repeat(B, 1, 1, 1),
                                    torch.zeros_like(self.image_latents).repeat(
                                        B, 1, 1, 1
                                    ),
                                ],
                                dim=0,
                            ),
                        ],
                        dim=1,
                    ),
                    torch.cat([t] * 2, dim=0),
                    encoder_hidden_states=torch.cat(
                        [
                            image_camera_embeddings,
                            torch.zeros_like(image_camera_embeddings),
                        ],
                        dim=0,
                    ),
                    velocity_to_epsilon=self.pipe.scheduler.config.prediction_type
                    == "v_prediction",
                )

            (
                noise_pred_pretrain_image,
                noise_pred_pretrain_uncond,
            ) = noise_pred_pretrain.chunk(2)
            noise_pred_pretrain = (
                noise_pred_pretrain_uncond
                + self.cfg.guidance_scale
                * (noise_pred_pretrain_image - noise_pred_pretrain_uncond)
            )

        w = (1 - self.alphas[t]).view(-1, 1, 1, 1)  # weighting strategy: dreamfusion

        grad = w * (noise_pred_pretrain - noise)

        if self.cfg.vis_sds:
            import cv2
            import numpy as np

            outp_root_dir = "./draw_exp/print_freq_latents_sds_grad/0123"
            case_name = self.cfg.cond_image_path.split("/")[-1].split(".")[0]
            outp_dir = os.path.join(outp_root_dir, case_name)
            if not os.path.exists(outp_dir):
                os.makedirs(outp_dir, exist_ok=True)
            for idx in range(int(grad.shape[1])):
                grad_print = grad[0][idx].detach().clone()
                npy_out_dir = os.path.join(outp_dir, f"{str(idx).zfill(4)}_spatial.npy")
                grad_npy = grad_print.detach().cpu().numpy().astype(np.float32)
                np.save(npy_out_dir, grad_npy)
                grad_print = grad_print ** 2
                grad_print = (grad_print - grad_print.min()) / (
                    grad_print.max() - grad_print.min()
                )
                grad_print *= 255.0
                grad_print = grad_print.cpu().numpy().astype(np.uint8)
                grad_print = cv2.resize(grad_print, (512, 512))
                img_out_dir = os.path.join(outp_dir, f"{str(idx).zfill(4)}_spatial.png")
                grad_print = cv2.applyColorMap(grad_print, cv2.COLORMAP_JET)
                cv2.imwrite(img_out_dir, grad_print)

        if self.cfg.cal_sds_in_freq:
            freq_H, freq_W = grad.shape[-2], grad.shape[-1]
            freq_bound_pos_H = int(freq_H * self.cfg.freq_bound_ratio)
            freq_bound_pos_W = int(freq_W * self.cfg.freq_bound_ratio)
            freq_grad = dct.dct_2d(grad, norm="ortho")
            if self.cfg.use_freq_in_low:
                freq_grad[:, :, freq_bound_pos_H:, freq_bound_pos_W:] = 0.0
            else:
                freq_grad[:, :, :freq_bound_pos_H, :freq_bound_pos_W] = 0.0
            grad = dct.idct_2d(freq_grad, norm="ortho")

        if self.cfg.vis_sds:
            import cv2
            import numpy as np

            outp_root_dir = "./draw_exp/print_freq_latents_sds_grad/0123"
            case_name = self.cfg.cond_image_path.split("/")[-1].split(".")[0]
            outp_dir = os.path.join(outp_root_dir, case_name)
            if not os.path.exists(outp_dir):
                os.makedirs(outp_dir, exist_ok=True)
            for idx in range(int(grad.shape[1])):
                grad_print = grad[0][idx].detach().clone()
                npy_out_dir = os.path.join(outp_dir, f"{str(idx).zfill(4)}.npy")
                grad_npy = grad_print.detach().cpu().numpy().astype(np.float32)
                np.save(npy_out_dir, grad_npy)
                grad_print = grad_print ** 2
                grad_print = (grad_print - grad_print.min()) / (
                    grad_print.max() - grad_print.min()
                )
                grad_print *= 255.0
                grad_print = grad_print.cpu().numpy().astype(np.uint8)
                grad_print = cv2.resize(grad_print, (512, 512))
                img_out_dir = os.path.join(outp_dir, f"{str(idx).zfill(4)}.png")
                grad_print = cv2.applyColorMap(grad_print, cv2.COLORMAP_JET)
                cv2.imwrite(img_out_dir, grad_print)
        return grad

    def get_image_camera_embeddings(
        self,
        elevation_deg: Float[Tensor, "B"],
        azimuth_deg: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
    ) -> Float[Tensor, "B 1 D"]:
        batch_size = elevation_deg.shape[0]
        camera_embeddings: Float[Tensor, "B 1 4"] = torch.stack(
            [
                torch.deg2rad(self.cfg.cond_elevation_deg - elevation_deg),
                torch.sin(torch.deg2rad(azimuth_deg - self.cfg.cond_azimuth_deg)),
                torch.cos(torch.deg2rad(azimuth_deg - self.cfg.cond_azimuth_deg)),
                camera_distances - self.cfg.cond_camera_distance,
            ],
            dim=-1,
        )[:, None, :]

        image_camera_embeddings = self.pipe.clip_camera_projection(
            torch.cat(
                [
                    self.clip_image_embeddings.repeat(batch_size, 1, 1),
                    camera_embeddings,
                ],
                dim=-1,
            ).to(self.weights_dtype)
        )

        return image_camera_embeddings

    def __call__(
        self,
        rgb: Float[Tensor, "B H W C"],
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        mvp_mtx: Float[Tensor, "B 4 4"],
        c2w: Float[Tensor, "B 4 4"],
        rgb_as_latents=False,
        **kwargs,
    ):
        batch_size = rgb.shape[0]

        rgb_BCHW = rgb.permute(0, 3, 1, 2)
        latents: Float[Tensor, "B 4 32 32"]
        if rgb_as_latents:
            # treat input rgb as latents
            # input rgb should be in range [-1, 1]
            latents = F.interpolate(
                rgb_BCHW, (32, 32), mode="bilinear", align_corners=False
            )
        else:
            # treat input rgb as rgb
            # input rgb should be in range [0, 1]
            rgb_BCHW = F.interpolate(
                rgb_BCHW, (256, 256), mode="bilinear", align_corners=False
            )
            # encode image into latents with vae
            latents = self.vae_encode(self.pipe.vae, rgb_BCHW * 2.0 - 1.0)

        # image-camera feature condition
        image_camera_embeddings = self.get_image_camera_embeddings(
            elevation, azimuth, camera_distances
        )

        if self.cfg.camera_condition_type == "extrinsics":
            camera_condition = c2w
        elif self.cfg.camera_condition_type == "mvp":
            camera_condition = mvp_mtx
        elif self.cfg.camera_condition_type == "spherical":
            camera_condition = torch.stack(
                [
                    torch.deg2rad(elevation),
                    torch.sin(torch.deg2rad(azimuth)),
                    torch.cos(torch.deg2rad(azimuth)),
                    camera_distances,
                ],
                dim=-1,
            )
        else:
            raise ValueError(
                f"Unknown camera_condition_type {self.cfg.camera_condition_type}"
            )

        grad = self.compute_grad_sds(latents, image_camera_embeddings, camera_condition)

        grad = torch.nan_to_num(grad)
        # clip grad for stable training?
        if self.grad_clip_val is not None:
            grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)

        # reparameterization trick
        # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
        target = (latents - grad).detach()
        loss_sds = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size

        return {
            "loss_3d_sds": loss_sds,
            "grad_norm": grad.norm(),
            "min_step": self.min_step,
            "max_step": self.max_step,
        }

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        # clip grad for stable training as demonstrated in
        # Debiasing Scores and Prompts of 2D Diffusion for Robust Text-to-3D Generation
        # http://arxiv.org/abs/2303.15413
        if self.cfg.grad_clip is not None:
            self.grad_clip_val = C(self.cfg.grad_clip, epoch, global_step)

        self.set_min_max_steps(
            min_step_percent=C(self.cfg.min_step_percent, epoch, global_step),
            max_step_percent=C(self.cfg.max_step_percent, epoch, global_step),
        )
