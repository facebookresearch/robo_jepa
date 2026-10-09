# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""AOT compile infrastructure for the DiT diffusion decoder.

Provides DecoderStepModule — a flat-tensor wrapper around SimpleDiT for
torch.export / AOT Inductor compilation. Two variants per cache_step:
  - update_cache=False (denoising steps 0..N-2): read-only cache, no mutations
  - update_cache=True  (last denoising step):    functional cache, returns new KV

Program matrix: 5 cache_steps × 2 update_cache × 6 batch_sizes = 60 programs.
"""

import gc
import os
import sys
import time

import torch
import torch.nn as nn

from app.robo_jepa.robo_jepa.models.utils.modules import KeyValueCache


class DecoderStepModule(nn.Module):
    """Wraps SimpleDiT for AOT compilation with flat tensor I/O.

    Bakes cache_size and update_cache as compile-time constants.
    For update_cache=True: uses functional_cache_tensors to avoid mutations.
    For update_cache=False: uses standard read-only cache path (no mutations).
    """

    def __init__(self, dit_model, cache_size, update_cache):
        super().__init__()
        self.dit = dit_model
        self.cache_size = cache_size
        self.update_cache = update_cache

    def forward(self, noisy_x, timestep_t, condition, cache_keys, cache_values, time_offset):
        """
        Args:
            noisy_x:     [B, T_frame, C_in, H_cosmos, W_cosmos] noisy latent
            timestep_t:  [B, T_frame] diffusion timestep (long)
            condition:   [B, N_cond, cond_dim] V-JEPA features
            cache_keys:  [depth, B, num_kv_heads, cache_len, frame_tokens, head_dim]
            cache_values: same shape
            time_offset: 0-dim int64, frames rolled out of the cache before this call (KeyValueCache.evicted)

        Returns:
            (output, new_cache_keys, new_cache_values)
        """
        depth = cache_keys.shape[0]
        target_dtype = cache_keys.dtype
        cache = KeyValueCache(
            layers=depth,
            batch=cache_keys.shape[1],
            frame_tokens=cache_keys.shape[4],
            cache_len=cache_keys.shape[3],
            num_heads=cache_keys.shape[2],
            head_dim=cache_keys.shape[5],
            dtype=target_dtype,
            device=cache_keys.device,
            keys=[cache_keys[i] for i in range(depth)],
            values=[cache_values[i] for i in range(depth)],
            pointer=[self.cache_size] * depth,
            evicted=time_offset,
        )

        if self.update_cache:
            result = self.dit(
                noisy_x,
                timestep_t,
                condition,
                cache=cache,
                use_cache=True,
                update_cache=True,
                cache_size=self.cache_size,
                functional_cache_tensors=(cache_keys, cache_values),
            )
            # functional path returns (pixels, cache, new_cache_keys, new_cache_values)
            output = result[0]
            new_cache_keys = result[2]
            new_cache_values = result[3]
        else:
            result = self.dit(
                noisy_x,
                timestep_t,
                condition,
                cache=cache,
                use_cache=True,
                update_cache=False,
                cache_size=self.cache_size,
            )
            # read-only path returns (pixels, cache)
            output = result[0]
            new_cache_keys = cache_keys
            new_cache_values = cache_values

        return output, new_cache_keys, new_cache_values


def get_decoder_config(wm, grid_height=None, grid_width=None):
    """Extract decoder config needed for building example inputs.

    Args:
        wm: WorldModel instance.
        grid_height: Override encoder grid height (for cross-resolution compilation).
            Decoder grid and condition dims are scaled proportionally.
        grid_width: Override encoder grid width (for cross-resolution compilation).
    """
    head = wm.heads["image_head"]
    dit = head.ema_model
    num_kv_heads = dit.num_kv_heads if dit.num_kv_heads is not None else dit.num_heads

    # If overrides provided, scale decoder grid and condition dims proportionally
    if grid_height is not None and grid_width is not None:
        scale_h = grid_height / wm.model.grid_height
        scale_w = grid_width / wm.model.grid_width
        dec_grid_h = int(dit.grid_height * scale_h)
        dec_grid_w = int(dit.grid_width * scale_w)
        cond_h = int(dit.condition_h * scale_h)
        cond_w = int(dit.condition_w * scale_w)
        img_h = dec_grid_h * dit.patch_size
        img_w = dec_grid_w * dit.patch_size
    else:
        dec_grid_h = dit.grid_height
        dec_grid_w = dit.grid_width
        cond_h = dit.condition_h
        cond_w = dit.condition_w
        img_h, img_w = dit.img_size

    return {
        "depth": dit.depth,
        "decoder_embed_dim": dit.decoder_embed_dim,
        "num_heads": dit.num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": dit.decoder_embed_dim // dit.num_heads,
        "grid_height": dec_grid_h,
        "grid_width": dec_grid_w,
        "in_chans": dit.in_chans,
        "condition_dim": dit.cond_mixing.in_features - dit.decoder_embed_dim,
        "condition_h": cond_h,
        "condition_w": cond_w,
        "cache_len": dit.attn_depth,
        "img_size": (img_h, img_w),
        "tubelet_size": dit.tubelet_size,
        "channels_mult": dit.channels_mult,
    }


def build_decoder_example_inputs(decoder_config, cache_step, batch_size, device="cuda"):
    """Build concrete example inputs for a given (cache_step, batch_size)."""
    B = batch_size
    depth = decoder_config["depth"]
    num_kv_heads = decoder_config["num_kv_heads"]
    head_dim = decoder_config["head_dim"]
    cache_len = decoder_config["cache_len"]
    grid_h = decoder_config["grid_height"]
    grid_w = decoder_config["grid_width"]
    in_chans = decoder_config["in_chans"]
    img_h, img_w = decoder_config["img_size"]
    cond_dim = decoder_config["condition_dim"]
    cond_h = decoder_config["condition_h"]
    cond_w = decoder_config["condition_w"]
    frame_tokens = grid_h * grid_w

    # noisy_x: one frame of Cosmos latent
    noisy_x = torch.randn(B, 1, in_chans, img_h, img_w, device=device, dtype=torch.bfloat16)

    # timestep_t: diffusion timestep per frame
    timestep_t = torch.randint(0, 1000, (B, 1), device=device, dtype=torch.long)

    # condition: V-JEPA features for one frame (pre-interpolation)
    # condition_d = 1 frame → N_cond = 1 * cond_h * cond_w
    n_cond = cond_h * cond_w
    condition = torch.randn(B, n_cond, cond_dim, device=device, dtype=torch.bfloat16)

    # KV cache
    cache_keys = torch.zeros(
        depth,
        B,
        num_kv_heads,
        cache_len,
        frame_tokens,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    cache_values = torch.zeros(
        depth,
        B,
        num_kv_heads,
        cache_len,
        frame_tokens,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )

    time_offset = torch.zeros((), device=device, dtype=torch.long)

    return noisy_x, timestep_t, condition, cache_keys, cache_values, time_offset
