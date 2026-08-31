from dataclasses import dataclass

import torch
import torch.nn.functional as F
from diffusers.loaders import AttnProcsLayers

import threestudio
from threestudio.utils.base import BaseModule
from threestudio.utils.misc import C, cleanup
from threestudio.utils.typing import *

import os
from PIL import Image
from torchvision import transforms

from extern.pfd.pfd import PromptFreeDiffusionPipeline
# Importing this module registers the OpenAI UNet variants used by PFD.
from extern.pfd.lib.model_zoo.openaimodel import UNetModel2D_Next
from extern.pfd.lib.model_zoo.attention import LoRAAttnProcessor

import torch_dct as dct


@threestudio.register("pfd-vsd-guidance")
class PromptFreeDiffusionVSDGuidance(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        pretrained_model_name_or_path: str = "./diffusion_ckpt/prompt-free-diffusion-base"
        pretrained_model_name_or_path_lora: str = "./diffusion_ckpt/prompt-free-diffusion"
        preprocessor_realtive_path: str = "controlnet/control_sd15_canny_slimmed.safetensors"
        preprocessor_method_name: str = "canny"
        diffuser_realtive_path: str = "pfd/diffuser/SD-v1-5.safetensors"
        diffuser_name: str = "SD-v1.5"
        ctxencoder_realtive_path: str = "pfd/seecoder/seecoder-v1-0.safetensors"
        ctxencoder_name: str = "SeeCoder"
        guidance_scale: float = 2.0
        guidance_scale_lora: float = 1.0
        grad_clip: Optional[Any] = None
        half_precision_weights: bool = True
        lora_cfg_training: bool = True
        lora_n_timestamp_samples: int = 1

        min_step_percent: float = 0.02
        max_step_percent: float = 0.98

        camera_condition_type: str = "extrinsics"
        cond_image_path: str = ""

        cal_vsd_in_freq: bool = False
        cal_lora_in_freq: bool = False
        use_freq_in_low: bool = False
        freq_bound_ratio: float = 0.0

        cross_attention_scale: float = 0.5

        vis_vsd: bool = False

    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading Prompt Free Diffusion...")

        self.unet_cross_attention_dim = (
            768  # FIXME: hard-coded for SD-v-1-5 given by pfd
        )
        self.unet_block_out_channels = [
            320,
            320,
            640,
            640,
            1280,
            1280,
            1280,
            1280,
            1280,
            1280,
            640,
            640,
            640,
            320,
            320,
            320,
        ]  # FIXME: hard-coded for SD-v-1-5 given by pfd

        self.weights_dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )

        @dataclass
        class SubModules:
            pipe: PromptFreeDiffusionPipeline
            pipe_lora: PromptFreeDiffusionPipeline

        ctxencoder_path = os.path.join(
            self.cfg.pretrained_model_name_or_path, self.cfg.ctxencoder_realtive_path
        )
        diffuser_path = os.path.join(
            self.cfg.pretrained_model_name_or_path, self.cfg.diffuser_realtive_path
        )
        preprocessor_path = os.path.join(
            self.cfg.pretrained_model_name_or_path, self.cfg.preprocessor_realtive_path
        )

        pipe = PromptFreeDiffusionPipeline(
            fp16=self.cfg.half_precision_weights,
            ctx_path=ctxencoder_path,
            diffuser_path=diffuser_path,
            ctl_path=preprocessor_path,
            tag_ctx=self.cfg.ctxencoder_name,
            tag_diffuser=self.cfg.diffuser_name,
            tag_ctl=self.cfg.preprocessor_method_name,
            device=self.device,
        )

        if (
            self.cfg.pretrained_model_name_or_path
            == self.cfg.pretrained_model_name_or_path_lora
        ):
            self.single_model = True
            pipe_lora = pipe
        else:
            self.single_model = False
            ctxencoder_path_lora = os.path.join(
                self.cfg.pretrained_model_name_or_path_lora,
                self.cfg.ctxencoder_realtive_path,
            )
            diffuser_path_lora = os.path.join(
                self.cfg.pretrained_model_name_or_path_lora,
                self.cfg.diffuser_realtive_path,
            )
            preprocessor_path_lora = os.path.join(
                self.cfg.pretrained_model_name_or_path_lora,
                self.cfg.preprocessor_realtive_path,
            )
            pipe_lora = PromptFreeDiffusionPipeline(
                fp16=self.cfg.half_precision_weights,
                ctx_path=ctxencoder_path_lora,
                diffuser_path=diffuser_path_lora,
                ctl_path=preprocessor_path_lora,
                tag_ctx=self.cfg.ctxencoder_name,
                tag_diffuser=self.cfg.diffuser_name,
                tag_ctl=self.cfg.preprocessor_method_name,
                device=self.device,
            )
        self.submodules = SubModules(pipe=pipe, pipe_lora=pipe_lora)
        cleanup()
        self.vae.eval()
        self.ctx.eval()
        self.ctl.eval()
        self.unet.eval()
        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.ctx.parameters():
            p.requires_grad_(False)
        for p in self.ctl.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)
        cleanup()
        self.vae_lora.eval()
        self.ctx_lora.eval()
        self.ctl_lora.eval()
        self.unet_lora.eval()
        for p in self.vae_lora.parameters():
            p.requires_grad_(False)
        for p in self.ctx_lora.parameters():
            p.requires_grad_(False)
        for p in self.ctl_lora.parameters():
            p.requires_grad_(False)
        for p in self.unet_lora.parameters():
            p.requires_grad_(False)

        # set up LoRA layers
        lora_attn_procs = {}
        for name in self.unet_lora.attn_processors.keys():
            cross_attention_dim = (
                None
                if name.endswith("attn1.processor")
                else self.unet_cross_attention_dim  # pfd_inference.net.diffuser.config.cross_attention_dim
            )
            block_id = int(name.split(".")[1])
            hidden_size = self.unet_block_out_channels[block_id]
            lora_attn_procs[name] = LoRAAttnProcessor(
                hidden_size=hidden_size, cross_attention_dim=cross_attention_dim
            )

        self.unet_lora.set_attn_processor(lora_attn_procs)

        self.lora_layers = AttnProcsLayers(self.unet_lora.attn_processors).to(
            self.device
        )
        self.lora_layers._load_state_dict_pre_hooks.clear()
        self.lora_layers._state_dict_hooks.clear()

        self.num_train_timesteps = self.pipe.net.num_timesteps
        self.set_min_max_steps()  # set to default value

        self.alphas: Float[Tensor, "..."] = self.pipe.net.alphas_cumprod.to(
            self.device, self.weights_dtype
        )

        self.grad_clip_val: Optional[float] = None

        self.prepare_reference_embeddings()

        threestudio.info(f"Loaded Prompt Free Diffusion!")

    def prepare_reference_embeddings(self) -> None:
        if not os.path.exists(self.cfg.cond_image_path):
            raise RuntimeError(
                f"Condition image not found at {self.cfg.cond_image_path}"
            )
        image = Image.open(self.cfg.cond_image_path).convert("RGB")
        craw = transforms.ToTensor()(image)[None].to(self.device).to(self.weights_dtype)
        with torch.no_grad():
            craw = craw.to(self.device).to(self.weights_dtype)
            self.reference_embeddings = (
                self.pipe.net.ctx_encode_trainable(craw, which="image")
                .to(self.device)
                .to(self.weights_dtype)
            )

    @torch.cuda.amp.autocast(enabled=False)
    def set_min_max_steps(self, min_step_percent=0.02, max_step_percent=0.98):
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)

    @property
    def pipe(self):
        return self.submodules.pipe

    @property
    def pipe_lora(self):
        return self.submodules.pipe_lora

    @property
    def unet(self):
        return self.submodules.pipe.net.diffuser["image"]

    @property
    def unet_lora(self):
        return self.submodules.pipe_lora.net.diffuser["image"]

    @property
    def vae(self):
        return self.submodules.pipe.net.vae["image"]

    @property
    def vae_lora(self):
        return self.submodules.pipe_lora.net.vae["image"]

    @property
    def ctl(self):
        return self.submodules.pipe.net.ctl

    @property
    def ctl_lora(self):
        return self.submodules.pipe_lora.net.ctl

    @property
    def ctx(self):
        return self.submodules.pipe.net.ctx["image"]

    @property
    def ctx_lora(self):
        return self.submodules.pipe_lora.net.ctx["image"]

    @torch.cuda.amp.autocast(enabled=False)
    def forward_unet(
        self,
        reference_image_encode: Float[Tensor, "..."],
        rendered_image: Float[Tensor, "..."],
        latents_noisy: Float[Tensor, "..."],
        t: Float[Tensor, "..."],
        tag_ctl: str,
        do_preprocess: bool,
        is_lora: bool,
        is_with_uncond: bool,
    ) -> Float[Tensor, "..."]:
        if is_lora:
            pipe = self.pipe_lora
        else:
            pipe = self.pipe

        h, w = (
            rendered_image.shape[-2],
            rendered_image.shape[-1],
        )  # FIXME: must be the times based on 64 (h % 64 == 0), but without checking

        c = reference_image_encode
        u = torch.zeros_like(c)

        ccraw = rendered_image.to(self.device, self.weights_dtype)
        if do_preprocess:
            if tag_ctl in ["canny", "canny_v11p"]:
                cc = pipe.net.ctl.preprocess(
                    ccraw,
                    type=tag_ctl,
                    size=[h, w],
                    low_threshold=150,
                    high_threshold=250,
                )
                cc = cc.to(self.weights_dtype)
            else:
                cc = pipe.net.ctl.preprocess(ccraw, type=tag_ctl, size=[h, w])
                cc = cc.to(self.weights_dtype)
        else:
            cc = ccraw

        if is_with_uncond:
            x_in = torch.cat([latents_noisy] * 2)
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([u, c])
        else:
            x_in = latents_noisy
            t_in = t
            c_in = c
        x_info = {"type": "image", "x": x_in}
        c_info = {"type": "image", "c": c_in, "control": cc}
        model_output = pipe.net.apply_model(
            x_info=x_info, timesteps=t_in, c_info=c_info
        )
        return model_output

    def compute_grad_vsd(
        self,
        reference_image_encode: Float[Tensor, "..."],
        rendered_image: Float[Tensor, "..."],
        latents: Float[Tensor, "..."],
        tag_ctl: str,
        do_preprocess: bool,
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

            noise = torch.randn_like(latents)
            latents_noisy = self.pipe.net.q_sample(x_start=latents, t=t, noise=noise)

            model_output = self.forward_unet(
                reference_image_encode=reference_image_encode,
                rendered_image=rendered_image,
                latents_noisy=latents_noisy,
                t=t,
                tag_ctl=tag_ctl,
                do_preprocess=do_preprocess,
                is_lora=False,
                is_with_uncond=True,
            )

            model_output_lora = self.forward_unet(
                reference_image_encode=reference_image_encode,
                rendered_image=rendered_image,
                latents_noisy=latents_noisy,
                t=t,
                tag_ctl=tag_ctl,
                do_preprocess=do_preprocess,
                is_lora=True,
                is_with_uncond=True,
            )

        noise_pred_pretrain_uncond, noise_pred_pretrain_cond = model_output.chunk(2)
        noise_pred_pretrain = noise_pred_pretrain_uncond + self.cfg.guidance_scale * (
            noise_pred_pretrain_cond - noise_pred_pretrain_uncond
        )
        noise_pred_lora_uncond, noise_pred_lora_cond = model_output_lora.chunk(2)
        noise_pred_lora = noise_pred_lora_uncond + self.cfg.guidance_scale_lora * (
            noise_pred_lora_cond - noise_pred_lora_uncond
        )
        w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
        grad = w * (noise_pred_pretrain - noise_pred_lora)

        if self.cfg.vis_vsd:
            import cv2
            import numpy as np

            outp_root_dir = "./draw_exp/print_freq_latents_vsd_grad/pfd"
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

        if self.cfg.cal_vsd_in_freq:
            freq_H, freq_W = grad.shape[-2], grad.shape[-1]
            freq_bound_pos_H = int(freq_H * self.cfg.freq_bound_ratio)
            freq_bound_pos_W = int(freq_W * self.cfg.freq_bound_ratio)
            assert freq_bound_pos_H == freq_bound_pos_W
            freq_grad = dct.dct_2d(grad, norm="ortho")
            if self.cfg.use_freq_in_low:
                freq_grad[:, :, freq_bound_pos_H:, freq_bound_pos_W:] = 0.0
            else:
                freq_grad[:, :, :freq_bound_pos_H, :freq_bound_pos_W] = 0.0
            grad = dct.idct_2d(freq_grad, norm="ortho")

        if self.cfg.vis_vsd:
            import cv2
            import numpy as np

            outp_root_dir = "./draw_exp/print_freq_latents_vsd_grad/pfd"
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

    def train_lora(
        self,
        reference_image_encode: Float[Tensor, "..."],
        rendered_image: Float[Tensor, "..."],
        latents: Float[Tensor, "..."],
        tag_ctl: str,
        do_preprocess: bool,
    ):
        B = latents.shape[0]
        latents = latents.detach().repeat(self.cfg.lora_n_timestamp_samples, 1, 1, 1)

        t = torch.randint(
            int(self.num_train_timesteps * 0.0),
            int(self.num_train_timesteps * 1.0),
            [B * self.cfg.lora_n_timestamp_samples],
            dtype=torch.long,
            device=self.device,
        )

        noise = torch.randn_like(latents)
        latents_noisy = self.pipe_lora.net.q_sample(latents, t, noise)

        model_output_lora = self.forward_unet(
            reference_image_encode=reference_image_encode,
            rendered_image=rendered_image,
            latents_noisy=latents_noisy,
            t=t,
            tag_ctl=tag_ctl,
            do_preprocess=do_preprocess,
            is_lora=True,
            is_with_uncond=False,
        )

        if self.cfg.cal_lora_in_freq:
            freq_H, freq_W = model_output_lora.shape[-2], model_output_lora.shape[-1]
            freq_bound_pos_H = int(freq_H * self.cfg.freq_bound_ratio)
            freq_bound_pos_W = int(freq_W * self.cfg.freq_bound_ratio)
            freq_model_output_lora = dct.dct_2d(model_output_lora, norm="ortho")
            freq_noise = dct.dct_2d(noise, norm="ortho")
            if self.cfg.use_freq_in_low:
                freq_model_output_lora[:, :, freq_bound_pos_H:, freq_bound_pos_W:] = 0.0
                freq_noise[:, :, freq_bound_pos_H:, freq_bound_pos_W:] = 0.0
            else:
                freq_model_output_lora[:, :, :freq_bound_pos_H, :freq_bound_pos_W] = 0.0
                freq_noise[:, :, :freq_bound_pos_H, :freq_bound_pos_W] = 0.0
            return F.mse_loss(
                freq_model_output_lora.float(), freq_noise.float(), reduction="mean"
            )

        return F.mse_loss(model_output_lora.float(), noise.float(), reduction="mean")

    def __call__(
        self,
        rendered_image: Float[Tensor, "B H W C"],
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        mvp_mtx: Float[Tensor, "B 4 4"],
        c2w: Float[Tensor, "B 4 4"],
        rgb_as_latents=False,
        tag_ctl="canny",
        do_preprocess=True,
        **kwargs,
    ):
        batch_size = rendered_image.shape[0]

        rgb_BCHW = rendered_image.permute(0, 3, 1, 2).to(
            self.device, self.weights_dtype
        )
        latents = self.pipe.net.vae_encode_trainable(rgb_BCHW, which="image").to(
            self.device, self.weights_dtype
        )

        if self.cfg.camera_condition_type == "extrinsics":
            camera_condition = c2w
        elif self.cfg.camera_condition_type == "mvp":
            camera_condition = mvp_mtx
        else:
            raise ValueError(
                f"Unknown camera_condition_type {self.cfg.camera_condition_type}"
            )

        grad = self.compute_grad_vsd(
            reference_image_encode=self.reference_embeddings,
            rendered_image=rgb_BCHW,
            latents=latents,
            tag_ctl=tag_ctl,
            do_preprocess=do_preprocess,
        )

        grad = torch.nan_to_num(grad)
        # clip grad for stable training?
        if self.grad_clip_val is not None:
            grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)

        # reparameterization trick
        # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
        target = (latents - grad).detach()
        loss_vsd = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size

        loss_lora = self.train_lora(
            reference_image_encode=self.reference_embeddings,
            rendered_image=rgb_BCHW,
            latents=latents,
            tag_ctl=tag_ctl,
            do_preprocess=do_preprocess,
        )

        return {
            "loss_vsd": loss_vsd,
            "loss_lora": loss_lora,
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
