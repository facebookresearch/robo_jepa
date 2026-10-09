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
from functools import partial

import einops
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from src.utils.tensors import trunc_normal_

try:
    from app.robo_jepa.robo_jepa.models.utils.xformer_utils import (
        build_blockwise_local_attention_mask,
        build_local_attention_mask,
        make_sparse_mask,
    )
except Exception:
    pass

from app.robo_jepa.prop_embedding import ProprioceptiveEmbedding
from app.robo_jepa.robo_jepa.models.utils.modules import Block, KeyValueCache


class VisionTransformerPredictor(nn.Module):
    """Vision Transformer Predictor"""

    def __init__(
        self,
        grid_height=16,
        grid_width=16,
        patch_size=1,
        num_frames=16,
        tubelet_size=1,
        embed_dim=768,
        predictor_embed_dim=384,
        depth=6,
        num_heads=12,
        num_kv_heads=None,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        uniform_power=False,
        num_views=3,
        view_map=None,
        action_conditioning="additive",
        action_tokens=1,
        action_dim=7,
        state_tokens=0,
        state_dim=7,
        camera_extrinsics_tokens=0,
        camera_extrinsics_dim=4,
        decode_state=False,
        use_alibi=False,
        use_dt=False,
        use_silu=False,
        wide_silu=True,
        use_activation_checkpointing=False,
        use_multiembodiment_proj=False,
        local_attn_window=(-1, -1, -1),
        attn_block_size=64,
        use_rope=False,
        attn_impl="fmha",
        embodiment_list=None,
        encoder_layernorm=False,
        use_view_tokens=True,
        view_groups_v=None,
        view_groups_hw=None,
        rope_interleave=False,
        use_strong_view_conditioning=False,
        use_qk_norm=False,
        qk_norm_type="layernorm",
        # ── μP (maximal update parameterization) -----------------------------
        # When mup_base_width is None (default), μP is OFF and behaviour is
        # bit-stable with the standard parameterization.  When set, the
        # predictor:
        #   - tags every Linear sub-module with `_mup_role` ∈
        #     {"input", "hidden", "head"};
        #   - re-scales hidden-trunk init by 1/√m (m = embed_dim/base_width);
        #   - re-scales output-head init by 1/m and/or multiplies output by
        #     1/m at forward time, depending on `mup_head_mode`;
        #   - threads `mup_attn_scale_mult = 1/√head_dim` into every
        #     Block.attn so the softmax scale becomes 1/d_k (instead of
        #     the standard 1/√d_k);
        #   - skips `_rescale_blocks` (the residual-1/√(2L) scheme that
        #     conflicts with μP's output 1/m treatment);
        #   - exposes `self.mup_width_mult` so the optimizer / scheduler
        #     can apply a `1/m` LR multiplier to hidden + head param
        #     groups (see app/robo_jepa/trainable_model.py).
        mup_base_width=None,
        mup_head_mode="init_only",
        mup_disable_residual_rescale=True,
    ):
        super().__init__()
        # ── μP state (must be computed BEFORE submodules are constructed
        # so they can be tagged with `_mup_role` immediately after build,
        # and BEFORE `self.apply(self._init_weights)` so role-aware
        # init reads the right multipliers).
        self.mup_enabled = mup_base_width is not None
        if self.mup_enabled:
            if int(mup_base_width) <= 0:
                raise ValueError(f"mup_base_width must be > 0, got {mup_base_width}")
            self.mup_width_mult = float(predictor_embed_dim) / float(mup_base_width)
            if mup_head_mode not in {"init_only", "fwd_mult", "both"}:
                raise ValueError(
                    f"mup_head_mode must be one of {{'init_only','fwd_mult','both'}}, got {mup_head_mode!r}"
                )
            self.mup_head_mode = mup_head_mode
            self.mup_disable_residual_rescale = bool(mup_disable_residual_rescale)
            # Forward-time output multiplier on the predictor head.
            if mup_head_mode in ("fwd_mult", "both"):
                self.mup_output_mult = 1.0 / self.mup_width_mult
            else:
                self.mup_output_mult = 1.0
            # Pre-softmax attention logit re-scaling: standard scale is
            # 1/√d_k; μP wants 1/d_k, i.e. an extra factor of 1/√d_k.
            # The per-head dim is `predictor_embed_dim // num_heads`.
            _head_dim = predictor_embed_dim // num_heads
            self.mup_attn_scale_mult = 1.0 / math.sqrt(_head_dim)
        else:
            self.mup_width_mult = 1.0
            self.mup_head_mode = mup_head_mode  # stored but unused
            self.mup_disable_residual_rescale = False
            self.mup_output_mult = 1.0
            self.mup_attn_scale_mult = 1.0
        self.mup_base_width = mup_base_width
        self.attn_depth, self.attn_height, self.attn_width = local_attn_window
        self.attn_block_size = attn_block_size

        # Determine positional embedding
        # --
        self.num_frames = num_frames
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.attn_impl = attn_impl
        self.camera_extrinsics_dim = camera_extrinsics_dim
        self.camera_extrinsics_tokens = camera_extrinsics_tokens
        self.is_video = num_frames > 1 and not use_alibi
        if use_rope and use_alibi:
            raise ValueError("RoPE and AliBi bias cannot be used at the same time!")
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.grid_depth = num_frames // self.tubelet_size
        self.use_activation_checkpointing = use_activation_checkpointing

        if self.is_video:
            self.num_patches = num_patches = (
                (num_frames // tubelet_size) * (grid_height // patch_size) * (grid_width // patch_size)
            )
        else:
            self.num_patches = num_patches = (grid_height // patch_size) * (grid_width // patch_size)  # noqa: F841
        # Position embedding
        self.uniform_power = uniform_power
        self.encoder_layernorm = encoder_layernorm
        # Attention Blocks
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        # Strong view conditioning (rope_view_embedding.md): when enabled
        # AND view_map is non-empty, every RoPEAttention block gets a
        # zero-init nn.Embedding(num_view_embeddings=len(view_map),
        # head_dim) which is added (post q/k cat) to every visual token
        # based on its semantic camera id. The v-axis RoPE rotation is
        # bypassed in this mode.
        self.use_strong_view_conditioning = use_strong_view_conditioning
        if use_strong_view_conditioning and (view_map is None or len(view_map) == 0):
            raise ValueError("use_strong_view_conditioning=True requires a non-empty view_map")
        # Validate qk_norm_type early so misconfigurations fail before block
        # construction (Block forwards the value into RoPEAttention/Attention).
        qk_norm_type = str(qk_norm_type).lower()
        if qk_norm_type not in ("layernorm", "rmsnorm"):
            raise ValueError(f"Unknown qk_norm_type={qk_norm_type!r}; expected 'layernorm' or 'rmsnorm'.")
        self.qk_norm_type = qk_norm_type
        _num_view_embeddings = (
            len(view_map) if (use_strong_view_conditioning and view_map is not None and len(view_map) > 0) else None
        )
        self.predictor_blocks = nn.ModuleList(
            [
                Block(
                    use_rope=True,
                    grid_height=self.grid_height // self.patch_size,
                    grid_width=self.grid_width // self.patch_size,
                    grid_depth=self.grid_depth,
                    dim=predictor_embed_dim,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    num_views=num_views,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_impl=attn_impl,
                    act_layer=nn.SiLU if use_silu else nn.GELU,
                    is_causal=False,
                    use_alibi=use_alibi,
                    attn_depth=self.attn_depth,
                    wide_silu=wide_silu,
                    block_size=attn_block_size,
                    attn_drop=attn_drop_rate,
                    layer_id=i,
                    norm_layer=norm_layer,
                    rope_interleave=rope_interleave,
                    num_view_embeddings=_num_view_embeddings,
                    use_qk_norm=use_qk_norm,
                    qk_norm_type=qk_norm_type,
                    mup_attn_scale_mult=self.mup_attn_scale_mult,
                )
                for i in range(depth)
            ]
        )
        self.depth = depth
        # Normalize & project back to input dimension
        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_embed_dim = predictor_embed_dim

        if decode_state:
            self.state_proj = nn.Linear(state_tokens * predictor_embed_dim, state_dim, bias=True)

        if patch_size > 1:
            self.predictor_embed = nn.Conv2d(
                embed_dim,
                predictor_embed_dim,
                kernel_size=(patch_size, patch_size),
                stride=(patch_size, patch_size),
            )
            self.predictor_proj = nn.ConvTranspose2d(
                predictor_embed_dim,
                embed_dim,
                kernel_size=(patch_size, patch_size),
                stride=(patch_size, patch_size),
            )
        else:
            self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
            self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)
        self.action_conditioning = action_conditioning
        self.action_tokens = action_tokens
        self.use_dt = use_dt
        self.state_tokens = state_tokens
        self.decode_state = decode_state
        self.num_views = num_views
        self.use_view_tokens = use_view_tokens
        if num_views != 0 and use_view_tokens:
            self.view_tokens = nn.Parameter(torch.zeros(1, num_views, predictor_embed_dim))

        self.view_map = view_map
        if view_map is not None and len(view_map) != 0:
            self.view_assignment_tokens = nn.Parameter(torch.zeros(1, len(view_map), predictor_embed_dim))

        if not use_view_tokens:
            assert view_map is not None and len(view_map) != 0, (
                "use_view_tokens=False requires a non-empty view_map so that view_assignment_tokens "
                "is the residual-stream view-identity signal."
            )

        self.embodiment_list = embodiment_list
        if embodiment_list is not None and len(embodiment_list) != 0:
            self.embodiment_assignment_tokens = nn.Parameter(torch.zeros(1, len(embodiment_list), predictor_embed_dim))

        proprio_kwargs = dict(
            num_frames=num_frames,
            tubelet_size=1,
            use_embeddding=not (use_rope or use_alibi),
            embed_dim=predictor_embed_dim,
            shift_input=False,
        )
        self.action_dim = action_dim
        self.state_dim = state_dim
        if action_tokens > 0:
            self.action_encoder = ProprioceptiveEmbedding(
                tokens_per_step=action_tokens,
                in_chans=action_dim,
                deep=False,
                **proprio_kwargs,
                is_multiembodiment=use_multiembodiment_proj,
                num_embodiments=max(len(embodiment_list) if use_multiembodiment_proj else 1, 1),
            )
        if state_tokens > 0:
            self.state_encoder = ProprioceptiveEmbedding(
                in_chans=state_dim,
                tokens_per_step=state_tokens,
                deep=False,
                **proprio_kwargs,
                is_multiembodiment=use_multiembodiment_proj,
                num_embodiments=max(len(embodiment_list) if use_multiembodiment_proj else 1, 1),
            )
        if self.use_dt:
            self.dt_encoder = ProprioceptiveEmbedding(in_chans=1, tokens_per_step=1, deep=False, **proprio_kwargs)
        if camera_extrinsics_tokens > 0:
            self.camera_extrinsics_encoder = ProprioceptiveEmbedding(
                in_chans=camera_extrinsics_dim,
                tokens_per_step=camera_extrinsics_tokens,
                deep=True,
                **proprio_kwargs,
            )

        # ------ initialize weights
        self.init_std = init_std
        # ── μP role tagging.  Walk the just-built submodule tree and stamp
        # every Linear / Conv2d / ConvTranspose2d with `_mup_role` so the
        # role-aware `_init_weights` can dispatch on it.  Untagged Linears
        # (e.g. those inside the ProprioceptiveEmbedding stack and the top-
        # level `predictor_embed`) default to the "input" role.
        self._tag_mup_roles()
        self.apply(self._init_weights)
        # `_rescale_blocks` (the ViT-style residual 1/√(2L) re-scaling)
        # conflicts with μP's output 1/m treatment, so skip it when μP is
        # ON (gated by `mup_disable_residual_rescale`, default True).
        if not (self.mup_enabled and self.mup_disable_residual_rescale):
            self._rescale_blocks()

        self.attn_mask_cache = {}
        attn_mask = None
        if self.attn_depth > 0 or self.attn_height > 0 or self.attn_width > 0:
            grid_depth = self.num_frames // self.tubelet_size
            grid_height = self.grid_height // self.patch_size
            grid_width = self.grid_width // self.patch_size
            blockwise = self.attn_impl in ["fmha", "flex"]
            # Masks live on GPU in production; fall back to CPU when no CUDA is
            # available (e.g. CPU unit tests of the TP/SP sequence logic).
            _mask_device = "cuda" if torch.cuda.is_available() else "cpu"
            # Build the union of V values to pre-warm the mask cache for. This
            # supports DDP across mixed-V ranks, where the predictor is built
            # with num_views = max_v across groups but ranks may run forward
            # passes with V < max_v.
            if view_groups_v is not None and len(view_groups_v) != 0:
                v_values = sorted({int(v) for v in view_groups_v} | {int(self.num_views)})
            else:
                v_values = [int(self.num_views)]
            # Default attn_mask used when num_views matches the construction-time num_views.
            attn_mask = self.generate_attn_mask(
                grid_depth,
                grid_height,
                grid_width,
                self.num_views,
                self.attn_depth,
                self.attn_height,
                self.attn_width,
                blockwise=blockwise,
                device=_mask_device,
            )
            self.attn_mask_cache[
                (
                    grid_depth,
                    grid_height,
                    grid_width,
                    self.num_views,
                    self.attn_depth,
                    self.attn_height,
                    self.attn_width,
                )
            ] = attn_mask

            # Pre-warm the mask cache for all possible (V, T) combinations
            # that may appear during training and autoregressive rollout. This
            # avoids GPU memory allocation (from generate_attn_mask) during
            # training, which can race with in-flight NCCL all-reduce
            # operations at multi-node DDP scale.
            #
            # `view_groups_hw` lists the distinct (grid_h, grid_w) resolutions
            # used by any view-group when ranks train at mixed resolutions. We
            # pre-warm the cross product {(H,W)} x {V} x {T} so no rank ever
            # allocates a mask mid-forward, regardless of its crop size. The
            # construction grid (grid_height, grid_width) is always included.
            hw_pairs = {(int(grid_height), int(grid_width))}
            if view_groups_hw is not None:
                for pair in view_groups_hw:
                    hw_pairs.add((int(pair[0]), int(pair[1])))
            for gh, gw in sorted(hw_pairs):
                for v in v_values:
                    for t in range(2, grid_depth + 1):
                        cache_key = (t, gh, gw, v, self.attn_depth, self.attn_height, self.attn_width)
                        if cache_key not in self.attn_mask_cache:
                            self.attn_mask_cache[cache_key] = self.generate_attn_mask(
                                t,
                                gh,
                                gw,
                                v,
                                self.attn_depth,
                                self.attn_height,
                                self.attn_width,
                                blockwise=blockwise,
                                device=_mask_device,
                            )
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
        device="cuda",
    ):
        if blockwise:
            # Build the block-sparse mask DIRECTLY (never materialize the dense
            # N x N mask, which is ~30 GB for a 720p/3-view/16-frame predictor
            # and OOMs host RAM when built on every rank at construction time).
            mask_blocks, mask_indices = build_blockwise_local_attention_mask(
                grid_depth,
                grid_height,
                grid_width,
                grid_views,
                local_window_time=attn_depth,
                local_window_h=attn_height,
                local_window_w=attn_width,
                action_tokens=self.action_tokens,
                state_tokens=self.state_tokens,
                block_size=self.attn_block_size,
                device=device,
            )
            mask_blocks = mask_blocks.to(device)
            row_offsets, column_indices, mask_indices_new = make_sparse_mask(mask_indices.to(device))
            return mask_blocks, mask_indices_new, row_offsets, column_indices

        attn_mask = build_local_attention_mask(
            grid_depth,
            grid_height,
            grid_width,
            grid_views,
            action_tokens=self.action_tokens,
            state_tokens=self.state_tokens,
            local_window_time=attn_depth,
            local_window_h=attn_height,
            local_window_w=attn_width,
            block_size_for_padding=None,
        )
        return attn_mask.to(device)

    def _tag_mup_roles(self):
        """Stamp every Linear / Conv on the predictor with a `_mup_role`.

        Roles:
          * "head"   — output-projection-like Linear / ConvTranspose2d that
                       maps predictor_embed_dim → encoder embed_dim (or
                       state_dim).  Currently `predictor_proj` and the
                       optional `state_proj`.
          * "hidden" — every Linear inside a transformer Block (attn QKV /
                       proj / MLP fcN).  Re-scaled by 1/√m under μP.
          * "input"  — everything else (predictor_embed, all proprio /
                       action / state encoders, the `view_embed` nn.Embedding
                       inside RoPEAttention).  Init at base std, LR ×1.

        The `_mup_role` attribute is read by `_init_weights` and by the
        param-group splitter in `TrainableModel.init_opt`.  Tagging is
        unconditional (cheap, harmless) so it works the same when μP is
        OFF — `_init_weights` only changes its std when `self.mup_enabled`.
        """
        # 1. Default every leaf module to the "input" role.
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.ConvTranspose2d, nn.Embedding)):
                if not hasattr(m, "_mup_role"):
                    m._mup_role = "input"
        # 2. Output head(s).
        if hasattr(self, "predictor_proj"):
            self.predictor_proj._mup_role = "head"
        if getattr(self, "decode_state", False) and hasattr(self, "state_proj"):
            self.state_proj._mup_role = "head"
        # 3. Hidden trunk: every Linear inside any Block's attn / mlp.
        for blk in self.predictor_blocks:
            for sub_name in ("qkv", "q", "kv", "proj"):
                sub = getattr(blk.attn, sub_name, None)
                if isinstance(sub, nn.Linear):
                    sub._mup_role = "hidden"
            for sub_name in ("fc1", "fc2", "fc3"):
                sub = getattr(blk.mlp, sub_name, None)
                if isinstance(sub, nn.Linear):
                    sub._mup_role = "hidden"

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            role = getattr(m, "_mup_role", "input")
            if self.mup_enabled and role == "hidden":
                std = self.init_std / math.sqrt(self.mup_width_mult)
            elif self.mup_enabled and role == "head" and self.mup_head_mode in ("init_only", "both"):
                std = self.init_std / self.mup_width_mult
            else:
                std = self.init_std
            trunc_normal_(m.weight, std=std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.predictor_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    @property
    def is_multiembodiment(self):
        return self.embodiment_list is not None and len(self.embodiment_list) != 0

    def forward(
        self,
        video_features,
        actions,
        states=None,
        camera_extrinsics=None,
        cache=None,
        use_cache=False,
        input_position=None,
        pos_embedding_resize_t=None,
        view_map=None,
        embodiment_ids=None,
        dt=None,
        cache_size=None,
        functional_cache_tensors=None,
    ):
        """

        Args:
            video_features:
        """
        if self.view_map is not None and view_map is not None:
            view_ids = {}
            if not isinstance(view_map, torch.Tensor):
                for k, v in enumerate(view_map):
                    view_ids[k] = self.view_map.index(v.item())
                view_ids = [view_ids[k] for k in range(len(view_ids))]
                view_ids = torch.from_numpy(np.array(view_ids)).to(video_features.device)
            else:
                view_ids = view_map.to(video_features.device)
        elif self.view_map is not None and view_map is None:
            raise ValueError("View mapping is not provided")
        else:
            view_ids = None

        D = video_features.shape[-1]
        x = video_features

        if self.patch_size > 1:
            b, t, v = x.shape[:3]
            x = self.predictor_embed(einops.rearrange(x, "b t v h w c -> (b t v) c h w"))
            x = einops.rearrange(x, "(b t v) c h w -> b t v h w c", b=b, t=t, v=v)
        else:
            x = self.predictor_embed(x)

        B, T, V, H, W, C = x.shape

        # TP+SP sequence padding: the residual stream is sharded on the flat
        # T*N sequence dim, which must be divisible by tp. N (=V*H*W +
        # action_tokens + state_tokens) is fixed, but T varies during
        # autoregressive rollout, so for some T, T*N is not divisible by tp.
        # Pad T up to T_pad (whole dummy frames) so T_pad*N divides tp. The
        # temporal attention is strictly causal, so the trailing dummy frames are
        # never attended by real queries; their outputs are sliced off before
        # predictor_norm. No-op when tp is off or T*N already divides tp.
        T_pad = T
        pad_frames = 0
        if getattr(self, "_tp_mesh", None) is not None and not use_cache:
            _tp_size = int(self._tp_size)
            _n_total = V * H * W + self.action_tokens + self.state_tokens
            _step = _tp_size // math.gcd(_n_total, _tp_size)
            T_pad = ((T + _step - 1) // _step) * _step
            pad_frames = T_pad - T

        is_full_sequence_no_cache = self.attn_mask is not None and cache is None and not use_cache and T_pad > 1
        is_multistep_with_cache = use_cache and (T > 1)
        if is_full_sequence_no_cache or is_multistep_with_cache:
            if is_multistep_with_cache and cache is not None:
                raise ValueError("Multistep autoregressive inference with non-empty cache is not supported yet.")
            # `is_full_sequence_no_cache` means we have T>1 latent features AND we want to initialize
            # the kv-cache with them.
            if (
                T_pad,
                H,
                W,
                V,
                self.attn_depth,
                self.attn_height,
                self.attn_width,
            ) in self.attn_mask_cache:
                attn_mask = self.attn_mask_cache[(T_pad, H, W, V, self.attn_depth, self.attn_height, self.attn_width)]
            else:
                grid_depth = T_pad
                grid_height = H
                grid_width = W
                grid_views = V
                attn_mask = self.generate_attn_mask(
                    grid_depth,
                    grid_height,
                    grid_width,
                    grid_views,
                    self.attn_depth,
                    self.attn_height,
                    self.attn_width,
                    blockwise=self.attn_impl == "fmha",
                    device=x.device,
                )
                self.attn_mask_cache[
                    (
                        grid_depth,
                        grid_height,
                        grid_width,
                        grid_views,
                        self.attn_depth,
                        self.attn_height,
                        self.attn_width,
                    )
                ] = attn_mask
        else:
            attn_mask = None

        if pad_frames > 0 and attn_mask is None:
            # Dummy frames are only safe to append because a causal temporal
            # mask prevents real queries from attending to them. Without a mask
            # (full bidirectional attention) padding would corrupt real tokens.
            raise RuntimeError(
                "TP sequence padding needs a causal attn_mask but got attn_mask=None "
                f"(T*N not divisible by tp={self._tp_size}). Set a local_attn_window so a "
                "causal mask is built, or use a tp size that divides the sequence."
            )

        if self.camera_extrinsics_tokens > 0:
            if camera_extrinsics is None:
                raise ValueError("camera extrinsics are not provided!")
            camera_extrinsics_tokens = self.camera_extrinsics_encoder(
                camera_extrinsics,
                starting_step=input_position,
                global_length=pos_embedding_resize_t,
            )
            # [B T 1 C]
            camera_extrinsics_tokens = camera_extrinsics_tokens.unsqueeze(3).unsqueeze(3)  # [B T 1 1 1 C]
        else:
            camera_extrinsics_tokens = 0.0

        if self.num_views != 0 and self.use_view_tokens:
            view_tokens = self.view_tokens.view(1, 1, self.num_views, 1, 1, C)
            # When V < self.num_views (mixed-V across DDP ranks), only use the
            # leading V slots; the remaining slots are unused on this rank.
            view_tokens = view_tokens[:, :, :V]
            view_tokens = view_tokens.repeat(B, T, 1, H, W, 1)
            x = x + view_tokens + camera_extrinsics_tokens
        else:
            x = x + camera_extrinsics_tokens
        if view_ids is not None:
            # TODO: for simultaneous views this needs to be changed
            view_assignment_tokens = self.view_assignment_tokens.repeat(B, 1, 1)
            view_assignment_tokens = view_assignment_tokens.gather(1, view_ids.unsqueeze(2).repeat(1, 1, C))
            # view_assignment_tokens = self.view_assignment_tokens[:, view_ids]
            view_assignment_tokens = view_assignment_tokens.view(B, 1, V, 1, 1, C)
            x = x + view_assignment_tokens
        if embodiment_ids is not None and self.is_multiembodiment:
            embodiment_assignment_tokens = self.embodiment_assignment_tokens[:, embodiment_ids]
            embodiment_assignment_tokens = embodiment_assignment_tokens.view(B, 1, 1, 1, 1, C)
            x = x + embodiment_assignment_tokens
        N = V * H * W
        tokens_per_step = N
        x = x.reshape(B, T, N, C)

        if self.action_tokens > 0:
            if actions is None:
                raise ValueError("actions are not provided!")
            a = self.action_encoder(
                actions,
                embodiment_ids,
                starting_step=input_position,
                global_length=pos_embedding_resize_t,
            )
            x = torch.cat([x, a], dim=2)
            N += self.action_tokens

        if self.state_tokens > 0:
            if states is None:
                raise ValueError("states (proprio) are not provided!")
            s = self.state_encoder(
                states,
                embodiment_ids,
                starting_step=input_position,
                global_length=pos_embedding_resize_t,
            )
            x = torch.cat([x, s], dim=2)
            N += self.state_tokens

        if self.use_dt:
            if dt is None:
                raise ValueError("dt is not provided!")
            dt_emb = self.dt_encoder(dt.unsqueeze(-1))  # [b t c]
            x = x + dt_emb

        # if self.camera_extrinsics_tokens > 0:
        #     if camera_extrinsics is None:
        #         raise ValueError('camera extrinsics are not provided!')
        #     ce = self.camera_extrinsics_encoder(camera_extrinsics, starting_step=input_position, global_length=pos_embedding_resize_t)  # noqa: E501
        #     x = torch.cat([x, ce], dim=2)
        #     N += self.camera_extrinsics_tokens

        if cache is None and use_cache:
            # Under TP, attention is head-sharded (apply_tp_sp localizes
            # attn.num_heads to global//tp) but predictor.num_heads/num_kv_heads
            # stay global. The cache stores each rank's LOCAL head shard of K,V, so
            # allocate it with the per-rank head count; otherwise the cache reshape
            # in KeyValueCache.update mismatches the sharded K,V. frame_tokens=N and
            # head_dim are unchanged by TP (SP all-gathers the sequence before attn,
            # and per-head dim is fixed).
            _tp = self._tp_size if getattr(self, "_tp_mesh", None) is not None else 1
            _cache_heads = (self.num_heads if self.num_kv_heads is None else self.num_kv_heads) // _tp
            cache = KeyValueCache(
                layers=self.depth,
                batch=B,
                frame_tokens=N,
                cache_len=self.attn_depth,
                num_heads=_cache_heads,
                head_dim=self.predictor_embed_dim // self.num_heads,
                dtype=a.dtype,
                device=a.device,  # NOTE: `a` is used as a tmp workaround. Since `x` is always fp32
            )
            cache_size = 0

        x = x.reshape((B, T * N, C))
        if pad_frames > 0:
            # Append `pad_frames` whole dummy frames (zeros) so the flat sequence
            # length becomes T_pad*N, divisible by tp for even SP sharding.
            x = torch.cat([x, x.new_zeros(B, pad_frames * N, C)], dim=1)
        extra_tokens = N - tokens_per_step
        # Tensor+sequence parallelism: scatter the residual stream along the
        # sequence dim (T*N) across the TP group so each rank holds 1/tp of the
        # tokens. SequenceParallel norms operate per-token (over C) on the local
        # shard; the per-Block attention all-gathers the full sequence internally
        # (so RoPE / the block-sparse mask are unchanged), then reduce-scatters
        # back. Gathered to full length again before predictor_norm below.
        _tp_mesh = getattr(self, "_tp_mesh", None)
        if _tp_mesh is not None:
            from torch.distributed.tensor import DTensor, Replicate, Shard

            # Keep the residual stream as a DTensor (Shard on the sequence dim)
            # through the block stack so uneven sequence lengths (T*N not
            # divisible by tp during autoregressive rollout) shard correctly.
            x = DTensor.from_local(x, _tp_mesh, [Replicate()]).redistribute(placements=[Shard(1)])
        # Pin runtime V for RoPEAttention so it uses the actual V instead of
        # auto-inferring from sequence length (which can pick a wrong
        # decomposition when V<num_views — see RoPEAttention.forward).
        runtime_views = V
        # Pin the runtime spatial grid as well so mixed-resolution ranks
        # (different crop sizes per view-group) get the correct RoPE
        # decomposition. H, W come straight from the encoder output shape.
        runtime_grid_h = H
        runtime_grid_w = W
        # Fwd prop
        new_cache_keys_list = []
        new_cache_values_list = []
        for i, blk in enumerate(self.predictor_blocks):
            if self.use_activation_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    blk,
                    x,
                    attn_mask,
                    cache,
                    cache_size,
                    extra_tokens,
                    None,  # functional_cache
                    runtime_views,
                    runtime_grid_h,
                    runtime_grid_w,
                    view_ids,
                    use_reentrant=False,
                )
            else:
                if functional_cache_tensors is not None:
                    layer_cache = (
                        functional_cache_tensors[0][i],
                        functional_cache_tensors[1][i],
                    )
                    x, new_layer_cache = blk(
                        x,
                        attn_mask=attn_mask,
                        cache=cache,
                        extra_tokens=extra_tokens,
                        cache_size=cache_size,
                        functional_cache=layer_cache,
                        runtime_views=runtime_views,
                        runtime_grid_h=runtime_grid_h,
                        runtime_grid_w=runtime_grid_w,
                        view_ids=view_ids,
                    )
                    new_cache_keys_list.append(new_layer_cache[0])
                    new_cache_values_list.append(new_layer_cache[1])
                else:
                    x = blk(
                        x,
                        attn_mask=attn_mask,
                        cache=cache,
                        extra_tokens=extra_tokens,
                        cache_size=cache_size,
                        runtime_views=runtime_views,
                        runtime_grid_h=runtime_grid_h,
                        runtime_grid_w=runtime_grid_w,
                        view_ids=view_ids,
                    )
        # TP+SP: gather the sequence-sharded residual stream (a DTensor) back to
        # a full plain tensor before the (replicated) output norm/projection and
        # all downstream per-token post-processing.
        if _tp_mesh is not None:
            x = x.full_tensor()

        if pad_frames > 0:
            # Drop the trailing dummy frames so the output matches the real T
            # (and the external frame-level padding_mask used by the loss).
            x = x[:, : T * N]

        x = self.predictor_norm(x)

        x = x.reshape(B, T, N, -1)

        # if self.camera_extrinsics_tokens > 0:
        #     x, ce_tokens = x[:, :, :-self.camera_extrinsics_tokens], x[:, :, -self.camera_extrinsics_tokens:]

        if self.state_tokens > 0:
            x, state_tokens = (
                x[:, :, : -self.state_tokens],
                x[:, :, -self.state_tokens :],
            )
            if self.decode_state:
                state_tokens = self.state_proj(state_tokens.flatten(2))
                # μP head output multiplier (no-op when not mup_head_mode in
                # {"fwd_mult","both"}; self.mup_output_mult == 1.0 then).
                if self.mup_output_mult != 1.0:
                    state_tokens = state_tokens * self.mup_output_mult
        else:
            state_tokens = None

        if self.action_tokens > 0:
            x, action_tokens = (
                x[:, :, : -self.action_tokens],
                x[:, :, -self.action_tokens :],
            )
        else:
            action_tokens = None

        if self.patch_size > 1:
            x = einops.rearrange(x, "b t (v h w) c -> (b t v) c h w", v=V, h=H, w=W)
            x = self.predictor_proj(x)
            x = einops.rearrange(x, "(b t v) c h w -> b t v h w c", b=B, t=T, v=V)
        else:
            x = self.predictor_proj(x)
            x = x.reshape(B, T, V, H, W, D).contiguous()
        # μP head output multiplier on the main predictor output (no-op
        # unless mup_head_mode ∈ {"fwd_mult","both"}).
        if self.mup_output_mult != 1.0:
            x = x * self.mup_output_mult

        if self.encoder_layernorm:
            x = F.layer_norm(x, (x.size(-1),))

        if use_cache:
            if functional_cache_tensors is not None:
                new_cache_keys = torch.stack(new_cache_keys_list, dim=0)
                new_cache_values = torch.stack(new_cache_values_list, dim=0)
                return (
                    x,
                    action_tokens,
                    state_tokens,
                    cache,
                    new_cache_keys,
                    new_cache_values,
                )
            return x, action_tokens, state_tokens, cache
        return x, action_tokens, state_tokens, None


def vit_predictor(**kwargs):
    # Choose the per-layer norm class based on the new `norm_type` config key.
    # Defaults to LayerNorm (preserving existing behaviour); set
    # `model.transition_model.config.norm_type: rmsnorm` in the YAML to use
    # nn.RMSNorm instead.  The chosen class is wrapped with `partial(..., eps=1e-6)`
    # so call sites like `norm_layer(dim)` work unchanged for both Block and
    # the outer `predictor_norm`.
    norm_type = str(kwargs.pop("norm_type", "layernorm")).lower()
    if norm_type == "layernorm":
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
    elif norm_type == "rmsnorm":
        norm_layer = partial(nn.RMSNorm, eps=1e-6)
    else:
        raise ValueError(f"Unknown norm_type={norm_type!r}; expected 'layernorm' or 'rmsnorm'.")
    model = VisionTransformerPredictor(mlp_ratio=4, qkv_bias=True, norm_layer=norm_layer, **kwargs)
    return model
