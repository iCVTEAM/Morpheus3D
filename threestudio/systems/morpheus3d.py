from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
import torch_dct as dct

import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *


@threestudio.register("morpheus3d-system")
class Morpheus3D(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        stage: str = "coarse"
        guidance_type: str = ""
        guidance: dict = field(default_factory=dict)
        guidance_3d_type: str = ""
        guidance_3d: dict = field(default_factory=dict)
        guidance_clip_type: str = ""
        guidance_clip: dict = field(default_factory=dict)
        guidance_depth_type: str = ""
        guidance_depth: dict = field(default_factory=dict)

    cfg: Config

    def configure(self) -> None:
        super().configure()

        if self.cfg.guidance_type != "None":
            self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
        if self.cfg.guidance_3d_type != "None":
            self.guidance_3d = threestudio.find(self.cfg.guidance_3d_type)(
                self.cfg.guidance_3d
            )
        if self.cfg.guidance_clip_type != "None":
            self.guidance_clip = threestudio.find(self.cfg.guidance_clip_type)(
                self.cfg.guidance_clip
            )
        if self.cfg.guidance_depth_type != "None":
            self.guidance_depth = threestudio.find(self.cfg.guidance_depth_type)(
                self.cfg.guidance_depth
            )
        if self.cfg.prompt_processor_type != "None":
            self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
                self.cfg.prompt_processor
            )
            self.prompt_utils = self.prompt_processor()

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        if self.cfg.stage == "geometry":
            render_out = self.renderer(**batch, render_rgb=False)
        else:
            render_out = self.renderer(**batch)
        return {
            **render_out,
        }

    def on_fit_start(self) -> None:
        super().on_fit_start()

    def training_step(self, batch, batch_idx):
        out_input = self(batch)
        out = self(batch["random_camera"])

        if self.cfg.stage == "geometry":
            guidance_inp = out["comp_normal"]
        else:
            guidance_inp = out["comp_rgb"]

        loss = 0.0

        if self.cfg.guidance_type != "None":
            if self.cfg.prompt_processor_type != "None":
                guidance_out = self.guidance(
                    guidance_inp,
                    self.prompt_utils,
                    **batch["random_camera"],
                    rgb_as_latents=False,
                )
            else:
                guidance_out = self.guidance(
                    guidance_inp, **batch["random_camera"], rgb_as_latents=False
                )
            for name, value in guidance_out.items():
                self.log(f"train/{name}", value)
                if name.startswith("loss_"):
                    loss += (
                        value
                        * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])
                        * self.C(self.cfg.loss.lambda_c)
                    )
        if self.cfg.guidance_3d_type != "None":
            guidance_3d_out = self.guidance_3d(
                guidance_inp, **batch["random_camera"], rgb_as_latents=False
            )
            for name, value in guidance_3d_out.items():
                self.log(f"train/{name}", value)
                if name.startswith("loss_"):
                    loss += (
                        value
                        * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])
                        * self.C(self.cfg.loss.lambda_c)
                    )
        loss_rgb = F.mse_loss(
            out_input["comp_rgb"],
            batch["rgb"] * batch["mask"].float()
            + out_input["comp_rgb_bg"] * (1.0 - batch["mask"].float()),
        )
        self.log("train/loss_rgb", loss_rgb)
        loss += loss_rgb * (
            1.0
            - self.C(self.cfg.loss.lambda_c) / (self.C(self.cfg.loss.lambda_sigma) ** 2)
        )

        if self.cfg.guidance_type != "None":
            if self.cfg.guidance_type == "pfd-sds-guidance":
                if self.cfg.guidance.cal_sds_in_freq == True:
                    x1 = out_input["comp_rgb"]
                    x2 = batch["rgb"] * batch["mask"].float() + out_input[
                        "comp_rgb_bg"
                    ] * (1.0 - batch["mask"].float())
                    u1 = dct.dct_2d(x1, norm="ortho")
                    u2 = dct.dct_2d(x2, norm="ortho")
                    bound = int(x1.shape[-1] * self.cfg.guidance.freq_bound_ratio)
                    if self.cfg.guidance.use_freq_in_low == False:
                        u1[..., :bound, :bound] = 0.0
                        u2[..., :bound, :bound] = 0.0
                    else:
                        u1[..., bound:, bound:] = 0.0
                        u2[..., bound:, bound:] = 0.0
                    loss_freq = F.mse_loss(u1, u2)
                    loss += (
                        loss_freq
                        * self.C(self.cfg.loss.lambda_sds)
                        * (
                            1.0
                            - self.C(self.cfg.loss.lambda_c)
                            / (self.C(self.cfg.loss.lambda_sigma) ** 2)
                        )
                    )
                else:
                    loss += (
                        loss_rgb
                        * self.C(self.cfg.loss.lambda_sds)
                        * (
                            1.0
                            - self.C(self.cfg.loss.lambda_c)
                            / (self.C(self.cfg.loss.lambda_sigma) ** 2)
                        )
                    )
            elif self.cfg.guidance_type == "pfd-vsd-guidance":
                if self.cfg.guidance.cal_vsd_in_freq == True:
                    x1 = out_input["comp_rgb"]
                    x2 = batch["rgb"] * batch["mask"].float() + out_input[
                        "comp_rgb_bg"
                    ] * (1.0 - batch["mask"].float())
                    u1 = dct.dct_2d(x1, norm="ortho")
                    u2 = dct.dct_2d(x2, norm="ortho")
                    bound = int(x1.shape[-1] * self.cfg.guidance.freq_bound_ratio)
                    if self.cfg.guidance.use_freq_in_low == False:
                        u1[..., :bound, :bound] = 0.0
                        u2[..., :bound, :bound] = 0.0
                    else:
                        u1[..., bound:, bound:] = 0.0
                        u2[..., bound:, bound:] = 0.0
                    loss_freq = F.mse_loss(u1, u2)
                    loss += (
                        loss_freq
                        * self.C(self.cfg.loss.lambda_vsd)
                        * (
                            1.0
                            - self.C(self.cfg.loss.lambda_c)
                            / (self.C(self.cfg.loss.lambda_sigma) ** 2)
                        )
                    )
                else:
                    loss += (
                        loss_rgb
                        * self.C(self.cfg.loss.lambda_vsd)
                        * (
                            1.0
                            - self.C(self.cfg.loss.lambda_c)
                            / (self.C(self.cfg.loss.lambda_sigma) ** 2)
                        )
                    )
            else:
                print("error")
                exit(1)

        loss_mask = F.binary_cross_entropy(
            out_input["opacity"].clamp(1.0e-5, 1.0 - 1.0e-5), batch["mask"].float(),
        )
        self.log("train/loss_mask", loss_mask)
        loss += loss_mask * (
            1.0
            - self.C(self.cfg.loss.lambda_c)
            / (self.C(self.cfg.loss.lambda_sigma) ** 2)
            / 10.0
        )

        if self.cfg.guidance_clip_type != "None":
            for name, value in self.guidance_clip(out["comp_rgb"]).items():
                self.log(f"train/{name}", value)
                if name.startswith("loss_"):
                    loss += value * self.C(self.cfg.loss.lambda_clip)

        if self.cfg.stage == "coarse":
            if (
                self.cfg.guidance_depth_type != "None"
                and self.C(self.cfg.loss.lambda_depth) > 0.0
            ):
                loss_depth = self.guidance_depth(
                    out_input["depth"], batch["mask"], batch["ref_depth"]
                )["loss_depth"]
                self.log(f"train/loss_depth", loss_depth)
                loss += (
                    (
                        1.0
                        - self.C(self.cfg.loss.lambda_c)
                        / (self.C(self.cfg.loss.lambda_sigma) ** 2)
                    )
                    * loss_depth
                    * self.C(self.cfg.loss.lambda_depth)
                )

            if self.C(self.cfg.loss.lambda_orient) > 0:
                if "normal" not in out:
                    raise ValueError(
                        "Normal is required for orientation loss, no normal is found in the output."
                    )
                loss_orient = (
                    out["weights"].detach()
                    * dot(out["normal"], out["t_dirs"]).clamp_min(0.0) ** 2
                ).sum() / (out["opacity"] > 0).sum()
                self.log("train/loss_orient", loss_orient)
                loss += loss_orient * self.C(self.cfg.loss.lambda_orient)

            if "comp_normal" in out:
                normal = out["comp_normal"]
                loss_normal_smoothness_2d = (
                    (normal[:, 1:, :, :] - normal[:, :-1, :, :]).square().mean()
                    + (normal[:, :, 1:, :] - normal[:, :, :-1, :]).square().mean()
                )
                self.log("trian/loss_normal_smoothness_2d", loss_normal_smoothness_2d)
                loss += loss_normal_smoothness_2d * self.C(
                    self.cfg.loss.lambda_normal_smoothness
                )

                loss_sparsity = (out["opacity"] ** 2 + 0.01).sqrt().mean()
                self.log("train/loss_sparsity", loss_sparsity)
                loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)

                opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
                loss_opaque = binary_cross_entropy(opacity_clamped, opacity_clamped)
                self.log("train/loss_opaque", loss_opaque)
                loss += loss_opaque * self.C(self.cfg.loss.lambda_opaque)

            if "z_variance" in out:
                loss_z_variance = out["z_variance"][out["opacity"] > 0.5].mean()
                self.log("train/loss_z_variance", loss_z_variance)
                loss += loss_z_variance * self.C(self.cfg.loss.lambda_z_variance)

            if "sdf_grad" in out:
                loss_eikonal = (
                    (torch.linalg.norm(out["sdf_grad"], ord=2, dim=-1) - 1.0) ** 2
                ).mean()
                self.log("train/loss_eikonal", loss_eikonal)
                loss += loss_eikonal * self.C(self.cfg.loss.lambda_eikonal)
                self.log("train/inv_std", out["inv_std"], prog_bar=True)
        elif self.cfg.stage == "refine":
            loss_normal_consistency = out["mesh"].normal_consistency()
            self.log("train/loss_normal_consistency", loss_normal_consistency)
            loss += loss_normal_consistency * self.C(
                self.cfg.loss.lambda_normal_consistency
            )

            if self.C(self.cfg.loss.lambda_laplacian_smoothness) > 0:
                loss_laplacian_smoothness = out["mesh"].laplacian()
                self.log("train/loss_laplacian_smoothness", loss_laplacian_smoothness)
                loss += loss_laplacian_smoothness * self.C(
                    self.cfg.loss.lambda_laplacian_smoothness
                )
        elif self.cfg.stage == "texture":
            pass
        else:
            raise ValueError(f"Unknown stage {self.cfg.stage}")

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        loss *= self.cfg.loss.lambda_scale

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                if "comp_rgb" in out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ],
            name="validation_step",
            step=self.true_global_step,
        )

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-test/{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                if "comp_rgb" in out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ],
            name="test_step",
            step=self.true_global_step,
        )

    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"it{self.true_global_step}-test",
            f"it{self.true_global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.true_global_step,
        )
