# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict

import torch
from einops import rearrange

from app.robo_jepa.trainable_model import TrainableModel
from app.robo_jepa.robo_jepa.diffusion import create_diffusion
from app.robo_jepa.robo_jepa.diffusion.model import SimpleDiT


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


class WorldModelDiTHead(TrainableModel):
    def __init__(
        self,
        head_config,
        train_config=None,
        device="cpu",
        diffusion_kwargs=None,
    ):
        model = SimpleDiT(**head_config, diffusion_kwargs=diffusion_kwargs)
        super().__init__(model, train_config=train_config, device=device)
        self.unwrapped = model
        self.diffusion_kwargs = diffusion_kwargs if diffusion_kwargs is not None else {}
        self.ema_model = SimpleDiT(**head_config, diffusion_kwargs=diffusion_kwargs).to(device)
        self.ema_model.eval()
        for p in self.ema_model.parameters():
            p.requires_grad = False
        update_ema(self.ema_model, self.unwrapped, decay=0.0)

    def compute_loss(self, features, rgb_target, padding_mask):
        # TODO: use padding mask here
        B, T, V, H, W, C = features.shape
        _, t_pix, _, _, h_pix, w_pix = rgb_target.shape
        if self.unwrapped.decode_video:
            rgb_target = rearrange(rgb_target, "b t v c h w -> (b v) t c h w")
            features = rearrange(features, "b t v h w c -> (b v) (t h w) c")
        else:
            rgb_target = rearrange(rgb_target, "b t v c h w -> (b t v) c h w")
            features = rearrange(features, "b t v h w c -> (b t v) (h w) c")
        rgb_encoder_losses = self.unwrapped.compute_loss(rgb_target, features, model=self.model)
        if self.unwrapped.decode_video and not self.unwrapped.gaussian_diffusion.diffusion_forcing:
            rgb_encoder_losses = {k: v.reshape(B, V).mean(-1) for k, v in rgb_encoder_losses.items()}
        else:
            rgb_encoder_losses = {k: v.reshape(B, T, V).mean(-1) for k, v in rgb_encoder_losses.items()}
        return rgb_encoder_losses

    def optimization_step(self):
        result = super().optimization_step()
        # turn off decay for now
        update_ema(self.ema_model, self.unwrapped, decay=0.0)
        return result

    def decode(
        self,
        features,
        denoising_steps=128,
        cfg_scale=1.0,
        samples_prefix=None,
        verbose=False,
        cache=None,
        use_cache=False,
        return_raw=False,
        prefix_noise_timestamp=0,
    ):
        assert int(return_raw) + int(use_cache) <= 1, "return_raw and use_cache are mutually exclusive. "
        if self.ema_model.gaussian_diffusion.original_num_steps != denoising_steps:
            self.ema_model.gaussian_diffusion = create_diffusion(str(denoising_steps), **self.diffusion_kwargs)
        B, T, V, H, W, C = features.shape
        if self.unwrapped.decode_video:
            features = rearrange(features, "b t v h w c -> (b v) (t h w) c")
        else:
            features = rearrange(features, "b t v h w c -> (b t v) (h w) c")
        samples = self.ema_model.sample(
            features,
            cfg_scale=cfg_scale,
            prefix=samples_prefix,
            prefix_noise_timestamp=prefix_noise_timestamp,
            verbose=verbose,
            use_cache=use_cache,
            cache=cache,
            condition_grid=(H, W),
        )
        if use_cache:
            samples, cache = samples
        T = T * self.ema_model.tubelet_size
        samples_ = samples
        samples = rearrange(samples, "(b v) t c h w -> b t v c h w", b=B, v=V)
        rgb = self.postprocess_rgb(samples, online=use_cache)
        if return_raw:
            return rgb, samples_
        elif use_cache:
            return rgb, cache
        return rgb

    def preprocess_rgb(self, rgb_video):
        mean = torch.tensor((0.485, 0.456, 0.406), dtype=rgb_video.dtype, device=rgb_video.device)[
            None, None, None, None, None, :
        ]
        std = torch.tensor((0.229, 0.224, 0.225), dtype=rgb_video.dtype, device=rgb_video.device)[
            None, None, None, None, None, :
        ]
        rgb_video = (rgb_video / 255.0 - mean) / std
        rgb_video = rgb_video.permute(0, 1, 2, 5, 3, 4)
        return rgb_video

    def postprocess_rgb(self, rgb_output, **kwargs):
        rgb_output = rgb_output.float()
        mean = torch.tensor((0.485, 0.456, 0.406), dtype=rgb_output.dtype, device=rgb_output.device)[
            None, None, None, :, None, None
        ]
        std = torch.tensor((0.229, 0.224, 0.225), dtype=rgb_output.dtype, device=rgb_output.device)[
            None, None, None, :, None, None
        ]
        rgb_ = (255.0 * (rgb_output * std + mean)).clip(0.0, 255.0).to(torch.uint8)
        rgb_ = rearrange(rgb_, "b t v c h w -> b t v h w c")
        return rgb_

    def to(self, device):
        self.model.to(device)
        self.ema_model.to(device)
        return self

    def cpu(self):
        return self.to("cpu")

    def cuda(self):
        return self.to("cuda")


class WorldModelDiTCosmosHead(WorldModelDiTHead):
    """DiT diffusion decoder: denoises Cosmos video tokens conditioned on V-JEPA features.

    See ``docs/decoder_architecture.md`` for detailed documentation
    covering the end-to-end pipeline, DDPM training with x₀ prediction, the
    cosine_simple_diffusion noise schedule, diffusion forcing, local temporal
    attention, and KV-cache inference.
    """

    def __init__(
        self,
        head_config,
        train_config=None,
        device="cpu",
        diffusion_kwargs=None,
    ):
        head_config["regression_dim"] = 16
        encoder_path = head_config.get("encoder_path")
        if "encoder_path" in head_config:
            del head_config["encoder_path"]
        decoder_path = head_config.get("decoder_path")
        if "decoder_path" in head_config:
            del head_config["decoder_path"]
        super().__init__(head_config, train_config, device, diffusion_kwargs=diffusion_kwargs)
        self.diffusion_kwargs = diffusion_kwargs if diffusion_kwargs is not None else {}
        self.image_tokenizer = False
        from cosmos_tokenizer.image_lib import ImageTokenizer

        self.last_output = None
        self.online = True
        from cosmos_tokenizer.video_lib import CausalVideoTokenizer

        self.tokenizer_cls = ImageTokenizer if self.image_tokenizer else CausalVideoTokenizer
        self.tokenizer = self.tokenizer_cls(
            checkpoint_enc=encoder_path,
            checkpoint_dec=decoder_path,
            device=device,
            dtype="bfloat16",
        )
        self.tokenizer.eval()
        for p in self.tokenizer.parameters():
            p.requires_grad = False

    def preprocess_rgb(self, rgb_target):
        rgb_target = rgb_target / 255.0
        rgb_target = (rgb_target - 0.5) * 2
        b, _, v = rgb_target.shape[:3]
        if self.image_tokenizer:
            rgb_target = rearrange(rgb_target, "b t v h w c -> (b t v) c h w")
        else:
            rgb_target = rearrange(rgb_target, "b t v h w c -> (b v) c t h w")
        tokenized = self.tokenizer.encode(rgb_target)[0]
        B = tokenized.shape[0]
        t = B // (b * v)
        if self.image_tokenizer:
            tokenized = rearrange(tokenized, "(b t v) c h w -> b t v c h w", b=b, t=t, v=v)
        else:
            tokenized = rearrange(tokenized, "(b v) c t h w -> b t v c h w", b=b, v=v)
        return tokenized

    def postprocess_rgb(self, rgb_output, online=False):
        b, _, v = rgb_output.shape[:3]
        if self.image_tokenizer:
            rgb_output = rearrange(rgb_output, "b t v c h w -> (b t v) c h w")
        else:
            rgb_output = rearrange(rgb_output, "b t v c h w -> (b v) c t h w")
        self.online = online
        if self.online:
            if self.last_output is None:
                self.last_output = rgb_output
            else:
                self.last_output = torch.cat([self.last_output, rgb_output], dim=2)
                self.last_output = self.last_output[:, :, -8:]
            decoded = self.tokenizer.decode(self.last_output)[:, :, -4:]
        else:
            decoded = self.tokenizer.decode(rgb_output)
        B = decoded.shape[0]
        if self.image_tokenizer:
            t = B // (b * v)
            decoded = rearrange(decoded, "(b t v) c h w -> b t v h w c", b=b, t=t, v=v)
        else:
            decoded = rearrange(decoded, "(b v) c t h w -> b t v h w c", b=b, v=v)
        decoded = (decoded + 1) / 2
        decoded = decoded * 255.0
        return decoded.clip(0.0, 255.0).to(torch.uint8)

    def to(self, device):
        self.model.to(device)
        self.ema_model.to(device)
        self.tokenizer.to(device)
        return self
