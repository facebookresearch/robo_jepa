# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import math

import torch
import torch.nn as nn
from einops import rearrange

from src.utils.tensors import trunc_normal_

try:
    from app.robo_jepa.robo_jepa.models.utils.xformer_utils import (
        build_local_attention_mask,
        make_sparse_mask,
        mask_to_blockwise,
    )
except Exception:
    pass

from app.robo_jepa.robo_jepa.diffusion import create_diffusion
from app.robo_jepa.robo_jepa.models.utils.modules import (
    MLP,
    Attention,
    DropPath,
    KeyValueCache,
    RoPEAttention,
    SwiGLUFFN,
)
from app.robo_jepa.robo_jepa.models.utils.patch_embed import PatchEmbed


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


# code is inspired by: https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/nets/PixArt_blocks.py
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.SiLU(),
        )
        self.output = nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(self.dtype)
        t = self.mlp(t_freq)
        return self.output(t), t

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


class CondLayerNormBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        num_kv_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        wide_silu=True,
        norm_layer=nn.LayerNorm,
        attn_impl="fmha",
        is_causal=False,
        grid_height=16,
        grid_width=16,
        grid_depth=1,
        grid_views=1,
        use_rope=False,
        layer_id=None,
        **kwargs,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if use_rope:
            self.self_attn = RoPEAttention(
                dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                block_size=64,
                is_causal=is_causal,
                attn_impl=attn_impl,
                proj_drop=drop,
                layer_id=layer_id,
                grid_height=grid_height,
                grid_width=grid_width,
                grid_depth=grid_depth,
                grid_views=grid_views,
            )
        else:
            self.self_attn = Attention(
                dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                block_size=64,
                is_causal=is_causal,
                attn_impl=attn_impl,
                proj_drop=drop,
                layer_id=layer_id,
                grid_size=grid_size,  # noqa: F821
                grid_depth=grid_depth,
                grid_views=grid_views,
            )
        self.layer_id = layer_id
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        # self.cross_attn = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias)

        if act_layer is nn.SiLU:
            self.mlp = SwiGLUFFN(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                wide_silu=wide_silu,
                drop=drop,
            )
        else:
            self.mlp = MLP(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                drop=drop,
            )
        self.scale_shift_table = nn.Parameter(torch.randn(6, dim) / dim**0.5)

    def forward(
        self,
        x,
        t,
        y,
        attn_mask=None,
        use_cache=False,
        cache=None,
        cache_size=0,
        update_cache=False,
        functional_cache=None,
    ):
        """
        Args:
            x: [B, h*w, C] (image diffusion) or [B, t*h*w, C] (video diffusion) - tokens for denoising
            t: [B, C * 6] (normal diffusion) or [B, t, C * 6] (diffusion forcing) - embedded timestep conditioning
            y: [B, N, C] - conditioning (not used right now as it is mixed in the input)
            attn_mask: the attention mask for self-attention over x
            functional_cache: optional (keys_in, values_in) for mutation-free cache path.
                When provided, the attention returns (attn_out, (keys_out, values_out))
                and this block returns (x, (keys_out, values_out)).
        """
        B = x.shape[0]
        C = x.shape[-1]
        cache_kwargs = dict(
            cache=cache,
            cache_size=cache_size,
            update_cache=update_cache,
            functional_cache=functional_cache,
        )
        new_layer_cache = None
        if len(t.shape) == 3:
            T = t.shape[1]
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.scale_shift_table[None, None] + t.reshape(B, T, 6, -1)
            ).chunk(6, dim=2)  # every shape is [B, T, C]; the shape of x here is [B, T * N, C] where N == H*W
            y = x
            x = modulate(self.norm1(x.reshape(B, T, -1, C)), shift_msa, scale_msa).flatten(1, 2)  # [B, T * N, C]
            attn_out = self.self_attn(x, attn_mask=attn_mask, **cache_kwargs)
            if functional_cache is not None:
                attn_out, new_layer_cache = attn_out
            x = y + self.drop_path((gate_msa * attn_out.reshape(B, T, -1, C)).flatten(1, 2))
            # x = x + self.cross_attn(x, y)
            y = x
            x = modulate(self.norm2(x.reshape(B, T, -1, C)), shift_mlp, scale_mlp).flatten(1, 2)  # [B, T * N, C]
            x = self.mlp(x)
            x = y + self.drop_path((gate_mlp * x.reshape(B, T, -1, C)).flatten(1, 2))
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.scale_shift_table[None] + t.reshape(B, 6, -1)
            ).chunk(6, dim=1)
            attn_out = self.self_attn(
                modulate(self.norm1(x), shift_msa, scale_msa),
                attn_mask=attn_mask,
                **cache_kwargs,
            )
            if functional_cache is not None:
                attn_out, new_layer_cache = attn_out
            x = x + self.drop_path(gate_msa * attn_out)
            # x = x + self.cross_attn(x, y)
            x = x + self.drop_path(gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp)))
        if new_layer_cache is not None:
            return x, new_layer_cache
        return x


class FinalLinear(nn.Module):
    """
    The final layer of PixArt.
    """

    def __init__(self, hidden_size, patch_size, tubelet_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size,
            tubelet_size * patch_size * patch_size * out_channels,
            bias=True,
        )
        self.scale_shift_table = nn.Parameter(torch.randn(2, hidden_size) / hidden_size**0.5)
        self.out_channels = out_channels

    def forward(self, x, t):
        """

        Args:
            x ([B, H*W, C] or [B, T*H*W, C]):
            t ([B, C] or [B, T, C]):
        """
        if len(t.shape) == 3:
            B, T, C = t.shape
            shift, scale = (self.scale_shift_table[None, None] + t[:, :, None]).chunk(2, dim=2)
            x = modulate(self.norm_final(x).reshape(B, T, -1, C), shift, scale).flatten(1, 2)
        else:
            shift, scale = (self.scale_shift_table[None] + t[:, None]).chunk(2, dim=1)
            x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class SimpleDiT(nn.Module):
    """Vision Transformer"""

    def __init__(
        self,
        patch_size=8,
        tubelet_size=1,
        in_chans=3,  # input channels
        img_size=(256, 256),
        num_frames=16,
        decode_video=True,
        use_cosmos=False,
        condition_dim=1408,
        decoder_embed_dim=1024,
        condition_dropout=0.1,
        depth=3,
        num_heads=32,
        num_kv_heads=32,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        uniform_power=False,
        num_views=3,
        condition_h=16,
        condition_w=16,
        condition_d=1,
        zero_init_mask_tokens=True,
        use_silu=False,
        wide_silu=True,
        is_causal=False,
        use_activation_checkpointing=False,
        attn_impl="fmha",
        use_rope=False,
        local_attn_window=(-1, -1, -1),
        diffusion_kwargs=None,
        **kwargs,
    ):
        super().__init__()
        self.diffusion_kwargs = diffusion_kwargs if diffusion_kwargs is not None else {}

        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            in_chans=in_chans,
            tubelet_size=tubelet_size,
            embed_dim=decoder_embed_dim,
        )
        self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.decode_video = decode_video
        self.num_frames = num_frames
        self.use_cosmos = use_cosmos
        self.attn_impl = attn_impl

        self.condition_h = condition_h
        self.condition_w = condition_w
        self.condition_d = condition_d
        if self.decode_video:
            if self.use_cosmos:
                null_tokens = (condition_d + 1) * condition_h * condition_w
            else:
                null_tokens = condition_d * condition_h * condition_w
        else:
            null_tokens = condition_h * condition_w
        if not self.decode_video:
            assert self.tubelet_size == 1 and self.num_frames == 1, "Switch to video mode"

        # self.condition_embedding = nn.Linear(condition_dim, decoder_embed_dim)
        self.decoder_embed_dim = decoder_embed_dim
        self.cond_mixing = nn.Linear(decoder_embed_dim + condition_dim, decoder_embed_dim)
        self.timestep_embedding = TimestepEmbedder(decoder_embed_dim)
        self.channels_mult = 2 if self.diffusion_kwargs.get("learn_sigma", True) else 1
        self.null_y = nn.Parameter(torch.randn(null_tokens, condition_dim) / condition_dim**0.5)
        self.out_proj = FinalLinear(decoder_embed_dim, patch_size, tubelet_size, in_chans * self.channels_mult)
        self.gaussian_diffusion = create_diffusion(timestep_respacing="", **self.diffusion_kwargs)

        # Determine positional embedding
        # --
        grid_height = img_size[0] // patch_size
        grid_width = img_size[1] // patch_size
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.grid_views = num_views
        self.grid_depth = self.num_frames // self.tubelet_size
        self.img_size = img_size
        self.in_chans = in_chans
        self.depth = depth
        self.use_activation_checkpointing = use_activation_checkpointing

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        if self.decode_video:
            self.num_patches = num_patches = (self.grid_height) * (self.grid_width) * (self.grid_depth + 1)
        else:
            self.num_patches = num_patches = (self.grid_height) * (self.grid_width)  # noqa: F841
        # Position embedding
        self.uniform_power = uniform_power
        # Attention Blocks
        self.use_rope = use_rope
        self.num_kv_heads = num_kv_heads
        self.num_heads = num_heads
        self.blocks = nn.ModuleList(
            [
                CondLayerNormBlock(
                    use_rope=use_rope,
                    grid_height=self.grid_height,
                    grid_width=self.grid_width,
                    grid_depth=self.grid_depth,
                    grid_views=self.grid_views,
                    dim=decoder_embed_dim,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    act_layer=nn.SiLU if use_silu else nn.GELU,
                    is_causal=is_causal,
                    attn_impl=attn_impl,
                    wide_silu=wide_silu,
                    block_size=None,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    layer_id=i,
                )
                for i in range(depth)
            ]
        )
        self.depth = depth
        self.condition_dropout = condition_dropout

        # Normalize & project back to input dimension
        # self.decoder_norm = norm_layer(decoder_embed_dim)
        # self.decoder_proj = nn.Linear(decoder_embed_dim, patch_size*patch_size*in_chans * 2, bias=True)

        # ------ initialize weights
        self.init_std = init_std
        if not zero_init_mask_tokens:
            for mt in self.mask_tokens:
                trunc_normal_(mt, std=init_std)
        self.apply(self._init_weights)
        self._rescale_blocks()
        self.attn_depth, self.attn_height, self.attn_width = local_attn_window
        self.attn_mask_cache = {}
        attn_mask = None
        self.attn_block_size = 64
        if self.attn_depth > 0 or self.attn_height > 0 or self.attn_width > 0:
            grid_depth = self.num_frames // self.tubelet_size
            grid_height = self.grid_height
            grid_width = (
                self.grid_width
            )  # TODO: previous impl may be broken by this -> double check (`// patch_size` is removed)
            attn_mask = self.generate_attn_mask(
                grid_depth,
                grid_height,
                grid_width,
                1,
                self.attn_depth,
                self.attn_height,
                self.attn_width,
                blockwise=self.attn_impl in ["fmha", "flex"],
            )
            self.attn_mask_cache[
                (
                    grid_depth,
                    grid_height,
                    grid_width,
                    self.attn_depth,
                    self.attn_height,
                    self.attn_width,
                )
            ] = attn_mask
        self.attn_mask = attn_mask

    def generate_attn_mask(
        self,
        grid_depth,
        grid_height,
        grid_width,
        grid_views,
        attn_depth,
        attn_width,
        attn_height,
        blockwise=True,
    ):
        attn_mask = build_local_attention_mask(
            grid_depth,
            grid_height,
            grid_width,
            grid_views,
            action_tokens=0,
            state_tokens=0,
            local_window_time=attn_depth,
            local_window_h=attn_height,
            local_window_w=attn_width,
            block_size_for_padding=self.attn_block_size if blockwise else None,
        )
        if blockwise:
            attn_mask = mask_to_blockwise(attn_mask, self.attn_block_size)
            mask_blocks, mask_indices = attn_mask
            mask_blocks = mask_blocks.cuda()
            row_offsets, column_indices, mask_indices_new = make_sparse_mask(mask_indices.cuda())  # to(q.device))
            return mask_blocks, mask_indices_new, row_offsets, column_indices
        return attn_mask.cuda()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.elementwise_affine:
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        # for layer_id, layer in enumerate(self.decoder_blocks):
        #     rescale(layer.attn.proj.weight.data, layer_id + 1)
        #     rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _infer_condition_grid(self, cond_tokens):
        """Infer (cond_d, cond_h, cond_w) from the number of condition tokens,
        preserving the stored aspect ratio. Returns (condition_d, cond_h, cond_w)."""
        stored_spatial = self.condition_h * self.condition_w
        if cond_tokens % stored_spatial == 0:
            return cond_tokens // stored_spatial, self.condition_h, self.condition_w
        aspect = self.condition_h / self.condition_w
        for d in range(1, cond_tokens + 1):
            if cond_tokens % d != 0:
                continue
            spatial = cond_tokens // d
            w = round((spatial / aspect) ** 0.5)
            if w <= 0:
                continue
            h = spatial // w
            if h * w == spatial and h > 0:
                return d, h, w
        return cond_tokens // stored_spatial, self.condition_h, self.condition_w

    # @partial(torch.compile, dynamic=False, fullgraph=True)
    def forward(
        self,
        x,
        t,
        condition,
        cache=None,
        use_cache=False,
        update_cache=False,
        cache_size=None,
        functional_cache_tensors=None,
        condition_grid=None,
    ):
        """
        :param x: [B, 3, h_pixel, w_pixel] noisy input
        :param t: float time
        :param functional_cache_tensors: optional (cache_keys, cache_values) for
            mutation-free cache path. When provided, each block receives its
            per-layer slice and returns new cache tensors instead of mutating.
        :param condition_grid: optional (cond_h, cond_w) tuple to override
            ``_infer_condition_grid`` for cross-resolution inference.
        :params masks_y: indices of target tokens in input
        """
        x = self.patch_embed(x)
        B = x.shape[0]
        t_shape = t.shape
        t, t0 = self.timestep_embedding(t.flatten())
        t, t0 = t.unflatten(0, t_shape), t0.unflatten(0, t_shape)
        cond_dim = condition.shape[-1]
        if condition_grid is not None:
            cond_h, cond_w = condition_grid
            condition_d = condition.shape[1] // (cond_h * cond_w)
        else:
            condition_d, cond_h, cond_w = self._infer_condition_grid(condition.shape[1])
        T = condition_d
        scale_h = self.grid_height / self.condition_h
        scale_w = self.grid_width / self.condition_w
        H = int(cond_h * scale_h)
        W = int(cond_w * scale_w)

        is_cross_resolution = cond_h != self.condition_h or cond_w != self.condition_w
        is_full_sequence_no_cache = self.attn_mask is not None and cache is None and not use_cache
        if is_full_sequence_no_cache and not is_cross_resolution:
            cache_key = (T, H, W, self.attn_depth, self.attn_height, self.attn_width)
            if cache_key in self.attn_mask_cache:
                attn_mask = self.attn_mask_cache[cache_key]
            else:
                grid_depth = T
                grid_height = H
                grid_width = W
                grid_views = 1
                attn_mask = self.generate_attn_mask(
                    grid_depth,
                    grid_height,
                    grid_width,
                    grid_views,
                    self.attn_depth,
                    self.attn_height,
                    self.attn_width,
                    blockwise=self.attn_impl == "fmha",
                )
                self.attn_mask_cache[cache_key] = attn_mask
        else:
            attn_mask = None

        if is_full_sequence_no_cache:
            if self.decode_video and self.use_cosmos:
                if condition_d != (self.num_frames + 1):
                    condition = condition.reshape(B, condition_d, cond_h, cond_w, cond_dim)
                    condition = torch.cat([condition[:, :1], condition], dim=1).reshape(B, -1, cond_dim)
                    condition_d = condition_d + 1
        condition = self.drop_condition(condition)
        if condition.shape[1] != x.shape[1]:
            scale_sh = H / cond_h
            scale_sw = W / cond_w
            if self.decode_video:
                grid_depth = x.shape[1] // (H * W)
                scale_t = grid_depth / condition_d
                condition = (
                    nn.functional.interpolate(
                        condition.reshape(B, condition_d, cond_h, cond_w, cond_dim).permute(0, 4, 1, 2, 3),
                        scale_factor=(scale_t, scale_sh, scale_sw),
                        mode="nearest",
                    )
                    .permute(0, 2, 3, 4, 1)
                    .reshape(B, x.shape[1], cond_dim)
                )
            else:
                condition = (
                    nn.functional.interpolate(
                        condition.reshape(B, cond_h, cond_w, cond_dim).permute(0, 3, 1, 2),
                        scale_factor=(scale_sh, scale_sw),
                        mode="nearest",
                    )
                    .permute(0, 2, 3, 1)
                    .reshape(B, x.shape[1], cond_dim)
                )
        # condition = self.condition_embedding(condition)
        x = self.cond_mixing(torch.cat([x, condition], dim=2))

        if cache is None and use_cache:
            cache = KeyValueCache(
                layers=self.depth,
                batch=B,
                frame_tokens=H * W,
                cache_len=self.attn_depth,
                num_heads=self.num_heads if self.num_kv_heads is None else self.num_kv_heads,
                head_dim=self.decoder_embed_dim // self.num_heads,
                dtype=torch.bfloat16,
                device=x.device,  # NOTE: `a` is used as a tmp workaround. Since `x` is always fp32
            )
            cache_size = 0

        # Fwd prop — functional cache collects per-layer outputs
        new_cache_keys_list = []
        new_cache_values_list = []
        for i, blk in enumerate(self.blocks):
            func_cache_i = None
            if functional_cache_tensors is not None:
                all_keys, all_values = functional_cache_tensors
                func_cache_i = (all_keys[i], all_values[i])
            if self.use_activation_checkpointing:
                blk_out = torch.utils.checkpoint.checkpoint(
                    blk,
                    x,
                    t,
                    condition,
                    attn_mask,
                    use_cache,
                    cache,
                    cache_size,
                    update_cache,
                    func_cache_i,
                    use_reentrant=False,
                )
            else:
                blk_out = blk(
                    x,
                    t,
                    condition,
                    attn_mask=attn_mask,
                    cache=cache,
                    cache_size=cache_size,
                    use_cache=use_cache,
                    update_cache=update_cache,
                    functional_cache=func_cache_i,
                )
            if func_cache_i is not None:
                x, (ko, vo) = blk_out
                new_cache_keys_list.append(ko)
                new_cache_values_list.append(vo)
            else:
                x = blk_out
        x = self.out_proj(x, t0)

        if self.decode_video:
            grid_depth = x.shape[1] // (H * W)
            pixels = rearrange(
                x,
                "b (t h w) (p q l c) -> b (t l) c (h p) (w q)",
                p=self.patch_size,
                q=self.patch_size,
                c=self.in_chans * self.channels_mult,
                h=H,
                w=W,
                t=grid_depth,
                l=self.tubelet_size,
            )
        else:
            pixels = rearrange(
                x,
                "b (h w) (p q c) -> b c (h p) (w q)",
                p=self.patch_size,
                q=self.patch_size,
                c=self.in_chans * self.channels_mult,
                h=H,
                w=W,
            )
        if functional_cache_tensors is not None and new_cache_keys_list:
            new_cache_keys = torch.stack(new_cache_keys_list, dim=0)
            new_cache_values = torch.stack(new_cache_values_list, dim=0)
            return pixels, cache, new_cache_keys, new_cache_values
        if use_cache:
            return pixels, cache
        return pixels

    def drop_condition(self, condition):
        B = condition.shape[0]
        if self.training:
            drop_ids = torch.rand(B, device=condition.device) < self.condition_dropout
        else:
            drop_ids = torch.zeros(B, dtype=torch.bool, device=condition.device)
        null_y = self.null_y
        if condition.shape[1] > null_y.shape[0]:
            repeats = (condition.shape[1] + null_y.shape[0] - 1) // null_y.shape[0]
            null_y = null_y.repeat(repeats, 1)
        condition = torch.where(
            drop_ids[:, None, None],
            null_y[: condition.shape[1]].repeat(B, 1, 1),
            condition,
        )
        return condition

    def forward_with_cfg(self, x, t, y, cfg_scale, condition_grid=None):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y, condition_grid=condition_grid)
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

    def compute_loss(self, x, condition, model=None):
        t = torch.randint(
            0,
            self.gaussian_diffusion.num_timesteps,
            (x.shape[0], x.shape[1]),
            device=x.device,
        )
        model = model if model is not None else self
        return self.gaussian_diffusion.training_losses(model, x, t, model_kwargs={"condition": condition})

    def sample(
        self,
        condition,
        cfg_scale=1.0,
        verbose=False,
        prefix=None,
        use_cache=False,
        cache=None,
        prefix_noise_timestamp=1,
        condition_grid=None,
    ):
        B = condition.shape[0]
        batch_mul = 2 if cfg_scale != 1.0 else 1

        # Infer actual pixel dims from the condition tensor
        if condition_grid is not None:
            cond_h, cond_w = condition_grid
        else:
            _, cond_h, cond_w = self._infer_condition_grid(condition.shape[1])
        scale_h = self.grid_height / self.condition_h
        scale_w = self.grid_width / self.condition_w
        pixel_h = int(cond_h * scale_h) * self.patch_size
        pixel_w = int(cond_w * scale_w) * self.patch_size

        if self.decode_video:
            if not use_cache:
                num_frames = self.num_frames
                if self.use_cosmos:
                    num_frames += 1
            else:
                num_frames = 1
                assert prefix is None, "prefix of samples cannot be used with cache"
            z = torch.zeros(
                B * batch_mul,
                num_frames,
                self.in_chans,
                pixel_h,
                pixel_w,
                device=condition.device,
            )
            t = torch.zeros((B * batch_mul, num_frames), device=z.device).long()
            t += self.gaussian_diffusion.num_timesteps - 1
            if prefix is not None:
                z[:, : prefix.shape[1]] = prefix
                t[:, : prefix.shape[1]] = prefix_noise_timestamp
            z = self.gaussian_diffusion.q_sample(z, t)

            def next_t(t_):
                t_ = t_.clone()
                if prefix is not None:
                    t_[:, prefix.shape[1] :] -= 1
                else:
                    t_ -= 1
                # t_.clamp_(min=0)
                return t_

        else:
            z = torch.randn(
                B * batch_mul,
                self.in_chans,
                self.img_size[0],
                self.img_size[1],
                device=condition.device,
            )
        if cfg_scale != 1.0:
            y = torch.cat([condition, self.null_y.repeat(B, 1, 1)], dim=0)
            kwargs = {"y": y, "cfg_scale": cfg_scale, "condition_grid": (cond_h, cond_w)}
            func = self.forward_with_cfg
        else:
            kwargs = {"condition": condition, "condition_grid": (cond_h, cond_w)}
            func = self.forward
        if use_cache:
            kwargs["use_cache"] = True
            kwargs["cache"] = cache
            if cache is not None:
                kwargs["cache_size"] = (
                    cache.pointer[0] if isinstance(cache.pointer[0], int) else cache.pointer[0].item()
                )
        # print('running ddim')
        samples, kv_cache = self.gaussian_diffusion.ddim_sample_loop(
            # samples, kv_cache = self.gaussian_diffusion.p_sample_loop(
            func,
            z.shape,
            z,
            clip_denoised=False,
            model_kwargs=kwargs,
            progress=verbose,
            device=condition.device,
            t_init=t,
            next_t_func=next_t,
        )
        if cfg_scale != 1.0:
            samples, _ = samples.chunk(2, dim=0)
        if use_cache:
            return samples, kv_cache
        return samples
