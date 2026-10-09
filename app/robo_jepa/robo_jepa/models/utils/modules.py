# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import drop_path

from app.robo_jepa.robo_jepa.models.utils.attention import attention


def rotate_queries_or_keys(x, pos, interleave=False):
    B, num_heads, N, D = x.size()
    assert D % 2 == 0, "Embedding dimension must be a multiple of 2 for block matrix rotation"

    # -- compute angle for each position
    omega = torch.arange(D // 2, dtype=x.dtype, device=x.device)
    omega /= D / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)
    freq = torch.einsum("..., f -> ... f", pos, omega)  # (..., N, D/2), outer product

    # -- build rotation matrix and apply
    emb_sin = freq.sin()  # (..., N, D/2)
    emb_cos = freq.cos()  # (..., N, D/2)

    # Repeat each freq twice along the last dim so emb_* has shape (..., N, D)
    # which matches x's last dim. NOTE: do NOT squeeze(-1) first — that
    # silently collapses D/2==1 into a bare N dim and then repeat_interleave
    # produces (..., 2N) instead of (..., N, D), which broadcasts wrong.
    if interleave:
        emb_sin = emb_sin.squeeze(-1).repeat_interleave(2, dim=-1)
        emb_cos = emb_cos.squeeze(-1).repeat_interleave(2, dim=-1)
    else:
        emb_sin = emb_sin.squeeze(-1).repeat(1, 1, 1, 2)
        emb_cos = emb_cos.squeeze(-1).repeat(1, 1, 1, 2)

    # --
    y = x.unflatten(-1, (-1, 2))
    y1, y2 = y.unbind(
        dim=-1,
    )
    y = torch.stack((-y2, y1), dim=-1)
    y = y.flatten(-2)
    return (x * emb_cos) + (y * emb_sin)


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class MLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.SiLU,
        drop=0.0,
        wide_silu=True,
    ):
        super().__init__()
        out_features = out_features or in_features
        swiglu_hidden_features = hidden_features = hidden_features or in_features
        if wide_silu:
            swiglu_hidden_features = int(2 * hidden_features / 3)
            align_as = 8
            swiglu_hidden_features = (swiglu_hidden_features + align_as - 1) // align_as * align_as
        self.fc1 = nn.Linear(in_features, swiglu_hidden_features)
        self.fc2 = nn.Linear(in_features, swiglu_hidden_features)
        self.act = act_layer()
        self.fc3 = nn.Linear(swiglu_hidden_features, out_features)

    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        hidden = F.silu(x1) * x2
        return self.fc3(hidden)


class RoPEAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        num_kv_heads=None,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        attn_impl="fmha",
        grid_height=16,
        grid_width=16,
        grid_views=1,
        grid_depth=8,
        block_size=64,
        layer_id=None,
        is_causal=False,
        is_native=False,
        rope_interleave=False,
        num_view_embeddings=None,
        use_qk_norm=False,
        qk_norm_type="layernorm",
        mup_attn_scale_mult=1.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.layer_id = layer_id
        self.grid_views = grid_views
        self.head_dim = head_dim = dim // num_heads
        # μP softmax-scale modifier: identity (1.0) under standard
        # parameterization, `1/√head_dim` under μP so the effective
        # scale becomes `1/head_dim` (instead of `1/√head_dim`).
        # Threaded down to the `attention(...)` helper at call time.
        self.mup_attn_scale_mult = float(mup_attn_scale_mult)
        self.scale = (qk_scale or head_dim**-0.5) * self.mup_attn_scale_mult
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.rope_interleave = rope_interleave
        if num_kv_heads is None or num_kv_heads == num_heads:
            self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        else:
            self.q = nn.Linear(dim, dim, bias=qkv_bias)
            self.kv = nn.Linear(dim, 2 * head_dim * self.num_kv_heads, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop_prob = proj_drop
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_impl = attn_impl
        self.block_size = block_size
        # qk-norm (default off): per-head LayerNorm/RMSNorm applied to q and k
        # right after the qkv projection and BEFORE the KV-cache write so cached K
        # stays consistent at inference.  See https://arxiv.org/abs/2302.05442
        # (ViT-22B) and https://arxiv.org/abs/2407.21783 (Llama-3) for prior
        # art.  When off this is `nn.Identity()` and incurs zero overhead.
        # `qk_norm_type` selects between "layernorm" (default, backward-compat)
        # and "rmsnorm".
        self.use_qk_norm = use_qk_norm
        self.qk_norm_type = qk_norm_type
        if use_qk_norm:
            if qk_norm_type == "layernorm":
                self.q_norm = nn.LayerNorm(head_dim, eps=1e-6)
                self.k_norm = nn.LayerNorm(head_dim, eps=1e-6)
            elif qk_norm_type == "rmsnorm":
                self.q_norm = nn.RMSNorm(head_dim, eps=1e-6)
                self.k_norm = nn.RMSNorm(head_dim, eps=1e-6)
            else:
                raise ValueError(f"Unknown qk_norm_type={qk_norm_type!r}; expected 'layernorm' or 'rmsnorm'.")
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()
        # --
        if grid_views == 1:
            # for backward compatibility we fallback to the previous implementation
            # when working with one view
            self.d_dim = int(2 * ((head_dim // 3) // 2))
            self.h_dim = int(2 * ((head_dim // 3) // 2))
            self.w_dim = int(2 * ((head_dim // 3) // 2))
            self.v_dim = 0
        else:
            self.d_dim = int(2 * ((head_dim // 4) // 2))
            self.h_dim = int(2 * ((head_dim // 4) // 2))
            self.w_dim = int(2 * ((head_dim // 4) // 2))
            self.v_dim = int(2 * ((head_dim // 4) // 2))
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.grid_depth = grid_depth
        self.is_causal = is_causal
        # ---------------------------------------------------------------- #
        # Strong view conditioning (rope_view_embedding.md).               #
        # When ``view_ids`` is supplied at forward time, the v-axis RoPE   #
        # rotation is replaced by an additive bias indexed by the          #
        # *semantic* camera id. Zero-init so loading a legacy checkpoint   #
        # + adding this layer is a no-op forward at step 0 (only the      #
        # v-slab rotation goes away).                                      #
        # ---------------------------------------------------------------- #
        self.num_view_embeddings = num_view_embeddings
        if num_view_embeddings is not None and num_view_embeddings > 0 and grid_views > 1:
            self.view_embed = nn.Embedding(num_view_embeddings, self.head_dim)
            nn.init.zeros_(self.view_embed.weight)
        else:
            self.view_embed = None

    def _infer_grid(self, Nq, extra_tokens=0):
        """Infer actual (grid_h, grid_w, views) from the query sequence length,
        preserving the stored height/width aspect ratio. This lets RoPE
        positions be recomputed on the fly for any input resolution, just
        like the encoder does.
        Returns (grid_h, grid_w, actual_views)."""
        aspect = self.grid_height / self.grid_width
        for T_candidate in range(1, Nq + 1):
            if Nq % T_candidate != 0:
                continue
            tps = Nq // T_candidate
            tpf = tps - extra_tokens
            if tpf <= 0:
                continue
            for views in [self.grid_views, 1] if self.grid_views != 1 else [1]:
                if tpf % views != 0:
                    continue
                spatial = tpf // views
                for ar in [aspect, 1.0]:
                    w = round((spatial / ar) ** 0.5)
                    for w_try in [w, w - 1, w + 1]:
                        if w_try <= 0:
                            continue
                        h = spatial // w_try
                        if h * w_try == spatial and h > 0:
                            return h, w_try, views
        return self.grid_height, self.grid_width, self.grid_views

    def _get_frame_pos(self, ids):
        tokens_per_frame = int(self.grid_height * self.grid_width * self.grid_views)
        return ids // tokens_per_frame

    def _get_view_pos(self, ids):
        tokens_per_frame = int(self.grid_height * self.grid_width * self.grid_views)
        frame_ids = self._get_frame_pos(ids)
        tokens_per_view = int(self.grid_height * self.grid_width)
        ids = ids - frame_ids * tokens_per_frame
        return ids // tokens_per_view

    def _get_height_pos(self, ids):
        # Remove frame component from ids
        tokens_per_frame = int(self.grid_height * self.grid_width * self.grid_views)
        frame_ids = self._get_frame_pos(ids)
        ids = ids - tokens_per_frame * frame_ids
        tokens_per_view = int(self.grid_height * self.grid_width)
        view_ids = self._get_view_pos(ids)
        ids = ids - tokens_per_view * view_ids
        # --
        tokens_per_row = self.grid_width
        return ids // tokens_per_row

    def separate_positions(self, ids, grid_h=None, grid_w=None, views=None, extra_tokens=0):
        gh = grid_h if grid_h is not None else self.grid_height
        gw = grid_w if grid_w is not None else self.grid_width
        gv = views if views is not None else self.grid_views
        if extra_tokens == 0:
            tokens_per_frame = int(gh * gw * gv)
            frame_ids = ids // tokens_per_frame

            tokens_per_view = int(gh * gw)
            view_ids = (ids % tokens_per_frame) // tokens_per_view

            tokens_per_row = gw
            height_ids = ((ids % tokens_per_frame) % tokens_per_view) // tokens_per_row
            width_ids = ids - tokens_per_frame * frame_ids - tokens_per_view * view_ids - tokens_per_row * height_ids
            return frame_ids, height_ids, width_ids, view_ids
        else:
            tokens_per_frame = int(gh * gw * gv)
            frame_ids = ids // tokens_per_frame
            tokens_per_view = int(gh * gw)
            view_ids = (ids % tokens_per_frame) // tokens_per_view
            tokens_per_row = gw
            height_ids = ((ids % tokens_per_frame) % tokens_per_view) // tokens_per_row
            width_ids = ids - tokens_per_frame * frame_ids - tokens_per_view * view_ids - tokens_per_row * height_ids
            return frame_ids, height_ids, width_ids, view_ids

    def forward(
        self,
        x,
        mask=None,
        attn_mask=None,
        cache=None,
        extra_tokens=0,
        cache_size=None,
        update_cache=True,
        functional_cache=None,
        runtime_views=None,
        runtime_grid_h=None,
        runtime_grid_w=None,
        view_ids=None,
    ):
        B, N, C = x.size()

        if self.num_kv_heads == self.num_heads:
            if getattr(self, "_tp_head_major", False):
                # TP: fused qkv weight rows are re-packed head-major
                # (num_heads, 3, head_dim) so a contiguous column-parallel shard
                # splits HEADS (not q/k/v) across ranks. See parallel_utils.
                qkv = self.qkv(x).unflatten(-1, (self.num_heads, 3, self.head_dim)).permute(3, 0, 2, 1, 4)
            else:
                qkv = self.qkv(x).unflatten(-1, (3, self.num_heads, -1)).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, D]
        else:
            q = self.q(x).unflatten(-1, (self.num_heads, -1)).permute(0, 2, 1, 3)
            if getattr(self, "_tp_head_major", False):
                # TP: fused kv weight rows re-packed head-major
                # (num_kv_heads, 2, head_dim) so the column-parallel shard splits
                # KV HEADS across ranks (not k/v).
                kv = self.kv(x).unflatten(-1, (self.num_kv_heads, 2, self.head_dim)).permute(3, 0, 2, 1, 4)
            else:
                kv = self.kv(x).unflatten(-1, (2, self.num_kv_heads, -1)).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]  # [B, num_heads, N, D]

        # qk-norm: per-head RMSNorm on q and k.  Applied BEFORE the KV-cache
        # write so cached K matches freshly-projected K at inference time.
        # No-op when use_qk_norm=False (q_norm / k_norm are nn.Identity).
        # Cast back to v.dtype because RMSNorm/LayerNorm with fp32 weights
        # promotes bf16 inputs to fp32; the attention kernels expect q, k, v
        # to share dtype.
        q = self.q_norm(q).to(v.dtype)
        k = self.k_norm(k).to(v.dtype)

        new_layer_cache = None
        if functional_cache is not None:
            # Functional (mutation-free) cache path for CUDA graph compatibility.
            # functional_cache = (keys_in, values_in) for this layer,
            # shapes: (B, n_heads, cache_len, frame_tokens, head_dim)
            keys_in, values_in = functional_cache
            new_k = k.view(B, self.num_kv_heads, -1, keys_in.shape[3], self.head_dim)
            new_v = v.view(B, self.num_kv_heads, -1, values_in.shape[3], self.head_dim)
            keys_out, values_out, k, v = cache.update_functional(
                keys_in,
                values_in,
                new_k,
                new_v,
                layer=self.layer_id,
                cache_size=cache_size,
            )
            new_layer_cache = (keys_out, values_out)
            if q.dtype != k.dtype:
                q = q.to(k.dtype)
        elif cache is not None:
            k, v = cache.update(
                k,
                v,
                layer=self.layer_id,
                cache_size=cache_size,
                update_cache=update_cache,
            )
            # Ensure Q matches K dtype for flash attention (FP8 fix)
            if q.dtype != k.dtype:
                q = q.to(k.dtype)

        _, Hq, Nq, dH = q.shape
        _, Hk, Nk, _ = k.shape

        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1)
            d_mask, h_mask, w_mask, v_mask = self.separate_positions(mask)
        else:
            # Infer actual grid dims from the sequence length when stored
            # dims don't match (cross-resolution). Preserves aspect ratio.
            #
            # `runtime_views` lets the caller (e.g. the predictor on a
            # mixed-V DDP rank) pass the *actual* runtime number of views,
            # which is needed when V_runtime < self.grid_views: in that
            # case the auto-inference can pick a wrong (gh, gw, gv)
            # decomposition because the spatial layout (h*w) becomes
            # ambiguous (e.g. 16 spatial tokens could be 4×2×2 OR 4×4×1).
            # The actual flat layout in the predictor is V·H·W. When
            # ``runtime_views`` is provided we honor it; ``runtime_grid_h`` /
            # ``runtime_grid_w`` (when also provided) pin the runtime spatial
            # grid so mixed-V composes with mixed-resolution training
            # (different ranks at different crop sizes). They default to the
            # construction-time grid_height/width, preserving prior behavior.
            stored_tpf = int(self.grid_height * self.grid_width * self.grid_views)
            stored_tps = stored_tpf + extra_tokens
            if runtime_views is not None:
                gh = int(runtime_grid_h) if runtime_grid_h is not None else self.grid_height
                gw = int(runtime_grid_w) if runtime_grid_w is not None else self.grid_width
                gv = int(runtime_views)
                tokens_per_frame = gh * gw * gv
                tokens_per_step = tokens_per_frame + extra_tokens
            elif Nq % stored_tps == 0:
                tokens_per_frame = stored_tpf
                tokens_per_step = stored_tps
                gh, gw, gv = self.grid_height, self.grid_width, self.grid_views
            else:
                gh, gw, gv = self._infer_grid(Nq, extra_tokens)
                tokens_per_frame = gh * gw * gv
                tokens_per_step = tokens_per_frame + extra_tokens
            # TODO: stopped here
            Tq = Nq // tokens_per_step
            Tk = Nk // tokens_per_step
            frame_tokens_k = Tk * tokens_per_frame
            if Tq != 1 and Tq == Tk and attn_mask is None:
                raise ValueError(
                    "non-autoregressive inference without attention mask detected!!! "
                    "Switch to autoregressive inference or use the attention mask!"
                )
            mask_q = torch.arange((Tk - Tq) * tokens_per_frame, Tk * tokens_per_frame).to(x.device)
            # Absolute frame index of the first key (KeyValueCache.key_start). The time RoPE
            # (rope_interleave=False) is not shift-invariant, so cached frames keep the positions an uncached
            # forward over the same sequence gives them.
            _t_off = cache.key_start[self.layer_id] if cache is not None else 0
            mask_k = torch.arange(int(frame_tokens_k)).to(x.device)
            if extra_tokens == 0:
                d_mask_q, h_mask_q, w_mask_q, v_mask_q = self.separate_positions(
                    mask_q, grid_h=gh, grid_w=gw, views=gv
                )
                d_mask_k, h_mask_k, w_mask_k, v_mask_k = self.separate_positions(
                    mask_k, grid_h=gh, grid_w=gw, views=gv
                )
            else:
                d_mask_q, h_mask_q, w_mask_q, v_mask_q = self.separate_positions(
                    mask_q, grid_h=gh, grid_w=gw, views=gv, extra_tokens=extra_tokens
                )
                d_mask_k, h_mask_k, w_mask_k, v_mask_k = self.separate_positions(
                    mask_k, grid_h=gh, grid_w=gw, views=gv, extra_tokens=extra_tokens
                )

        if extra_tokens != 0:
            q_ = q.reshape(B, Hq, Tq, tokens_per_step, dH)
            q, q_extra = q_[:, :, :, :-extra_tokens], q_[:, :, :, -extra_tokens:]
            q = q.reshape(B, Hq, Tq * tokens_per_frame, dH)
            q_extra = q_extra.reshape(B, Hq, Tq * extra_tokens, dH)

            k_ = k.reshape(B, Hk, Tk, tokens_per_step, dH)
            k, k_extra = k_[:, :, :, :-extra_tokens], k_[:, :, :, -extra_tokens:]
            k = k.reshape(B, Hk, Tk * tokens_per_frame, dH)
            k_extra = k_extra.reshape(B, Hk, Tk * extra_tokens, dH)

        _il = self.rope_interleave
        s = 0
        # Rotate depth
        _shift = mask is None and (torch.is_tensor(_t_off) or _t_off != 0)
        if _shift:
            d_mask_q, d_mask_k = d_mask_q + _t_off, d_mask_k + _t_off
        qd = rotate_queries_or_keys(q[..., s : s + self.d_dim], pos=d_mask_q, interleave=_il)
        kd = rotate_queries_or_keys(k[..., s : s + self.d_dim], pos=d_mask_k, interleave=_il)

        if extra_tokens != 0:
            extra_tokens_mask_q = (
                torch.arange((Tk - Tq) * extra_tokens, Tk * extra_tokens).to(x.device) // extra_tokens
            )
            extra_tokens_mask_k = torch.arange(int(Tk * extra_tokens)).to(x.device) // extra_tokens
            if _shift:
                extra_tokens_mask_q, extra_tokens_mask_k = extra_tokens_mask_q + _t_off, extra_tokens_mask_k + _t_off
            q_extra_d = rotate_queries_or_keys(
                q_extra[..., s : s + self.d_dim], pos=extra_tokens_mask_q, interleave=_il
            )
            k_extra_d = rotate_queries_or_keys(
                k_extra[..., s : s + self.d_dim], pos=extra_tokens_mask_k, interleave=_il
            )
            q_extra = torch.cat([q_extra_d, q_extra[..., s + self.d_dim :]], dim=-1)
            k_extra = torch.cat([k_extra_d, k_extra[..., s + self.d_dim :]], dim=-1)
        s += self.d_dim
        # Rotate height dim
        qh = rotate_queries_or_keys(q[..., s : s + self.h_dim], pos=h_mask_q, interleave=_il)
        kh = rotate_queries_or_keys(k[..., s : s + self.h_dim], pos=h_mask_k, interleave=_il)
        s += self.h_dim
        # Rotate width dim
        qw = rotate_queries_or_keys(q[..., s : s + self.w_dim], pos=w_mask_q, interleave=_il)
        kw = rotate_queries_or_keys(k[..., s : s + self.w_dim], pos=w_mask_k, interleave=_il)
        s += self.w_dim

        # Rotate view dimensions
        # Strong view conditioning: when ``view_ids`` is supplied AND we have
        # a ``view_embed`` layer (i.e. ``num_view_embeddings`` was set at
        # construction), the v-axis RoPE rotation is REPLACED by an additive
        # learned bias indexed by the *semantic* camera id (applied after
        # the q/k cat below). Otherwise the legacy v-slab rotation runs.
        use_view_embed = self.view_embed is not None and view_ids is not None and self.grid_views != 1
        if self.grid_views != 1:
            qv = rotate_queries_or_keys(q[..., s : s + self.v_dim], pos=v_mask_q, interleave=_il)
            kv = rotate_queries_or_keys(k[..., s : s + self.v_dim], pos=v_mask_k, interleave=_il)
            s += self.v_dim
        else:
            qv = q[..., s : s + self.v_dim]
            kv = k[..., s : s + self.v_dim]

        # Combine rotated dimension
        if s < self.head_dim:
            qr = q[..., s:]
            kr = k[..., s:]
            q = torch.cat([qd, qh, qw, qv, qr], dim=-1)
            k = torch.cat([kd, kh, kw, kv, kr], dim=-1)
        else:
            q = torch.cat([qd, qh, qw, qv], dim=-1)
            k = torch.cat([kd, kh, kw, kv], dim=-1)

        # Strong view conditioning: additive learned bias indexed by
        # semantic camera id. Applied to the visual-token slab before the
        # extra-tokens reattach + attention call. ``view_ids`` is the
        # per-V vector (or per-(B,V)) of camera ids returned by the
        # predictor; we look up the per-token id by indexing with the
        # per-token view position ``v_mask_q`` / ``v_mask_k``.
        if use_view_embed:
            view_ids_long = view_ids.to(device=q.device, dtype=torch.long)
            if view_ids_long.dim() == 1:
                view_ids_long = view_ids_long.unsqueeze(0).expand(B, -1)
            elif view_ids_long.dim() != 2:
                raise ValueError(f"view_ids must be 1D (V,) or 2D (B, V); got {tuple(view_ids_long.shape)}")
            # Trust ``runtime_views`` (when supplied) as the actual V for
            # this batch — that's how ``gv`` is set in the mask-None path
            # above. Falling back to ``self.grid_views`` covers the legacy
            # path where the caller does not pin V at runtime. Avoids
            # ``"gv" in locals()`` introspection which is unreliable
            # under torch.compile / Dynamo (see job 6545632).
            effective_gv = int(runtime_views) if runtime_views is not None else self.grid_views
            if view_ids_long.shape[1] != effective_gv:
                raise ValueError(f"view_ids has V={view_ids_long.shape[1]} but effective_gv={effective_gv}")
            sem_q = view_ids_long[:, v_mask_q.to(torch.long)]  # (B, N_q_visual)
            sem_k = view_ids_long[:, v_mask_k.to(torch.long)]  # (B, N_k_visual)
            bias_q = self.view_embed(sem_q).unsqueeze(1).to(q.dtype)
            bias_k = self.view_embed(sem_k).unsqueeze(1).to(k.dtype)
            q = q + bias_q
            k = k + bias_k

        if extra_tokens != 0:
            q = q.reshape(B, Hq, Tq, tokens_per_frame, dH)
            q_extra = q_extra.reshape(B, Hq, Tq, extra_tokens, dH)
            q = torch.cat([q, q_extra], dim=3)
            q = q.reshape(B, Hq, Nq, dH)

            k = k.reshape(B, Hk, Tk, tokens_per_frame, dH)
            k_extra = k_extra.reshape(B, Hk, Tk, extra_tokens, dH)
            k = torch.cat([k, k_extra], dim=3)
            k = k.reshape(B, Hk, Nk, dH)

        x = attention(
            q,
            k,
            v,
            attn_mask,
            is_causal=self.is_causal,
            block_size=self.block_size,
            num_heads=self.num_heads,
            attn_impl=self.attn_impl,
            proj_drop_prob=self.proj_drop_prob,
            scale=self.scale if self.mup_attn_scale_mult != 1.0 else None,
        )

        # `-1` (not the captured global C) infers the local head-shard width so
        # this is correct under tensor parallelism (qkv is column-sharded, so the
        # attention output here is num_heads/tp * head_dim wide); equals C when TP
        # is off.
        x = x.transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        if new_layer_cache is not None:
            return x, new_layer_cache
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        num_kv_heads=None,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        block_size=64,
        attn_impl="fmha",
        use_alibi=False,
        attn_depth=8,
        is_causal=False,
        layer_id=None,
        use_qk_norm=False,
        qk_norm_type="layernorm",
        mup_attn_scale_mult=1.0,
        **kwargs,
    ):
        super().__init__()
        self.use_alibi = use_alibi
        self.attn_depth = attn_depth
        self.num_heads = num_heads
        self.layer_id = layer_id
        head_dim = dim // num_heads
        self.mup_attn_scale_mult = float(mup_attn_scale_mult)
        self.scale = (qk_scale or head_dim**-0.5) * self.mup_attn_scale_mult
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        if num_kv_heads is None or num_kv_heads == num_heads:
            self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        else:
            self.q = nn.Linear(dim, dim, bias=qkv_bias)
            self.kv = nn.Linear(dim, 2 * head_dim * self.num_kv_heads, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop_prob = proj_drop
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_impl = attn_impl
        self.is_causal = is_causal
        self.block_size = block_size
        # qk-norm (default off): per-head LayerNorm/RMSNorm on q,k right after
        # the qkv projection.  See RoPEAttention.__init__ for rationale.
        # `qk_norm_type` selects between "layernorm" (default) and "rmsnorm".
        self.use_qk_norm = use_qk_norm
        self.qk_norm_type = qk_norm_type
        if use_qk_norm:
            if qk_norm_type == "layernorm":
                self.q_norm = nn.LayerNorm(head_dim, eps=1e-6)
                self.k_norm = nn.LayerNorm(head_dim, eps=1e-6)
            elif qk_norm_type == "rmsnorm":
                self.q_norm = nn.RMSNorm(head_dim, eps=1e-6)
                self.k_norm = nn.RMSNorm(head_dim, eps=1e-6)
            else:
                raise ValueError(f"Unknown qk_norm_type={qk_norm_type!r}; expected 'layernorm' or 'rmsnorm'.")
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def forward(
        self,
        x,
        attn_mask=None,
        cache=None,
        extra_tokens=0,
        cache_size=None,
        update_cache=True,
    ):
        B, N, C = x.shape

        if self.num_kv_heads == self.num_heads:
            qkv = self.qkv(x).unflatten(-1, (3, self.num_heads, -1)).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, D]
        else:
            q = self.q(x).unflatten(-1, (self.num_heads, -1)).permute(0, 2, 1, 3)
            kv = self.kv(x).unflatten(-1, (2, self.num_kv_heads, -1)).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]  # [B, num_heads, N, D]

        # qk-norm (no-op when use_qk_norm=False).  Same rationale as in
        # RoPEAttention: applied BEFORE the KV-cache write so cache stays
        # consistent at inference.  Cast back to v.dtype: RMSNorm/LayerNorm
        # with fp32 weights promotes bf16 inputs to fp32; the attention
        # kernels expect q, k, v to share dtype.
        q = self.q_norm(q).to(v.dtype)
        k = self.k_norm(k).to(v.dtype)

        if cache is not None:
            k, v = cache.update(
                k,
                v,
                layer=self.layer_id,
                cache_size=cache_size,
                update_cache=update_cache,
            )

        x = attention(
            q,
            k,
            v,
            attn_mask,
            is_causal=self.is_causal,
            block_size=self.block_size,
            num_heads=self.num_heads,
            attn_impl=self.attn_impl,
            proj_drop_prob=self.proj_drop_prob,
            use_alibi=self.use_alibi,
            attn_depth=self.attn_depth,
            scale=self.scale if self.mup_attn_scale_mult != 1.0 else None,
        )
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class KeyValueCache:
    def __init__(
        self,
        layers: int,
        batch: int = 1,
        frame_tokens: int = 257,
        cache_len: int = 8,
        num_heads: int = 32,
        head_dim: int = 32,
        dtype=torch.float32,
        device="cuda",
        keys=None,
        values=None,
        pointer=None,
        evicted=None,
    ):
        self.layers = layers
        self.dtype = dtype
        self.device = device
        shape = (batch, num_heads, cache_len, frame_tokens, head_dim)
        if keys is None:
            self.keys = [torch.zeros(shape, dtype=self.dtype, device=self.device) for l in range(layers)]  # noqa: E741
        else:
            self.keys = keys
        if values is None:
            self.values = [
                torch.zeros(shape, dtype=self.dtype, device=self.device)
                for l in range(layers)  # noqa: E741
            ]  # noqa: E741
        else:
            self.values = values
        # Use Python int list instead of GPU tensor to avoid GPU↔CPU sync
        # on every pointer access/increment (enables CUDA graphs).
        if pointer is None:
            self.pointer = [0] * self.layers
        elif isinstance(pointer, torch.Tensor):
            self.pointer = pointer.tolist()
        elif isinstance(pointer, list):
            self.pointer = list(pointer)
        else:
            self.pointer = [int(pointer)] * self.layers
        self.frame_tokens = frame_tokens
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.batch = batch
        self.cache_len = cache_len
        # Frames rolled out of the cache so far: the absolute frame index of slot 0. A 0-dim tensor (the
        # compiled step programs take it as a runtime input) applies to every layer.
        if evicted is None:
            evicted = [0] * self.layers
        elif torch.is_tensor(evicted):
            evicted = [evicted] * self.layers
        self.evicted = list(evicted)
        # Absolute frame index of the first key returned by the latest update() (read by RoPEAttention).
        self.key_start = list(self.evicted)

    def select_history(self, prefix_len, cache_len):
        """
        Selects a slice of history from the cache. First it selects `prefix_len` first elements and drops the rest
        and then it selects the last `cache_len` elements among the previously selectected prefix. The idea is to
        first select a partial history (prefix) and then adjust it due to local attention (suffix).

        Args:
            prefix_len:
            cache_len:
        """
        # assert prefix_len < cache_len, "cache prefix cannot be bigger than cache size"
        start_idx, end_idx = max(prefix_len - cache_len + 1, 0), prefix_len
        history_len = end_idx - start_idx
        empty = torch.zeros(
            (
                self.batch,
                self.num_heads,
                cache_len - history_len,
                self.frame_tokens,
                self.head_dim,
            ),
            dtype=self.dtype,
            device=self.device,
        )
        keys = [
            torch.cat([self.keys[l][:, :, start_idx:end_idx], empty], dim=2)
            for l in range(len(self.keys))  # noqa: E741
        ]  # noqa: E741
        values = [
            torch.cat([self.values[l][:, :, start_idx:end_idx], empty], dim=2)
            for l in range(len(self.values))  # noqa: E741
        ]  # noqa: E741
        return KeyValueCache(
            layers=self.layers,
            batch=self.batch,
            frame_tokens=self.frame_tokens,
            cache_len=cache_len,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            dtype=self.dtype,
            device=self.device,
            pointer=[history_len] * self.layers,
            keys=keys,
            values=values,
            evicted=[e + start_idx for e in self.evicted],
        )

    def update(self, new_k, new_v, layer=0, cache_size=None, update_cache=True):
        new_k_, new_v_ = new_k, new_v
        new_k = new_k.view(self.batch, self.num_heads, -1, self.frame_tokens, self.head_dim)
        new_v = new_v.view(self.batch, self.num_heads, -1, self.frame_tokens, self.head_dim)
        cache_update_len = new_k.shape[2]
        if cache_size is None:
            cache_size = self.cache_len
        if cache_update_len >= self.cache_len:  # we just overwrite the full cache
            # attention runs over the new frames only, which follow the cache_size cached ones
            self.key_start[layer] = self.evicted[layer] + cache_size
            if not update_cache:
                return new_k_, new_v_
            self.evicted[layer] += cache_size + cache_update_len - self.cache_len
            if cache_update_len != self.cache_len:
                new_k = new_k[:, :, -self.cache_len :]
                new_v = new_v[:, :, -self.cache_len :]
            self.keys[layer].copy_(new_k.detach())
            self.values[layer].copy_(new_v.detach())
            self.pointer[layer] += self.cache_len
            return new_k_, new_v_

        else:  # we append the update to the cache  # noqa: RET505
            if not update_cache:
                # a full cache returns its window shifted by one frame (the oldest one dropped)
                self.key_start[layer] = self.evicted[layer] + (1 if cache_size == self.cache_len else 0)
                if cache_size == 0:
                    return (
                        new_k.view(
                            self.batch,
                            self.num_heads,
                            cache_update_len * self.frame_tokens,
                            self.head_dim,
                        ),
                        new_v.view(
                            self.batch,
                            self.num_heads,
                            cache_update_len * self.frame_tokens,
                            self.head_dim,
                        ),
                    )
                if cache_size == self.cache_len:
                    return (
                        torch.cat([self.keys[layer][:, :, 1:], new_k], dim=2).view(
                            self.batch,
                            self.num_heads,
                            self.cache_len * self.frame_tokens,
                            self.head_dim,
                        ),
                        torch.cat([self.values[layer][:, :, 1:], new_v], dim=2).view(
                            self.batch,
                            self.num_heads,
                            self.cache_len * self.frame_tokens,
                            self.head_dim,
                        ),
                    )
                else:
                    return (
                        torch.cat([self.keys[layer][:, :, :cache_size], new_k], dim=2).view(
                            self.batch,
                            self.num_heads,
                            (cache_size + 1) * self.frame_tokens,
                            self.head_dim,
                        ),
                        torch.cat([self.values[layer][:, :, :cache_size], new_v], dim=2).view(
                            self.batch,
                            self.num_heads,
                            (cache_size + 1) * self.frame_tokens,
                            self.head_dim,
                        ),
                    )
            for t in range(cache_update_len):
                ptr = self.pointer[layer]
                if (cache_size + t) < self.cache_len:
                    # Slice assignment instead of index_copy_ with GPU tensor
                    # index — compatible with CUDA graphs.
                    self.keys[layer][:, :, ptr : ptr + 1] = new_k[:, :, t : t + 1].detach()
                    self.values[layer][:, :, ptr : ptr + 1] = new_v[:, :, t : t + 1].detach()
                    self.pointer[layer] += 1
                else:
                    roll_index = (
                        torch.arange(
                            1,
                            self.cache_len + 1,
                            device=self.device,
                        )
                        % self.cache_len
                    )
                    self.keys[layer] = self.keys[layer][:, :, roll_index]
                    self.values[layer] = self.values[layer][:, :, roll_index]
                    self.evicted[layer] += 1
                    self.keys[layer][:, :, ptr - 1 : ptr] = new_k[:, :, t : t + 1].detach()
                    self.values[layer][:, :, ptr - 1 : ptr] = new_v[:, :, t : t + 1].detach()
            new_cache_size = min(self.cache_len, cache_size + cache_update_len)
            self.key_start[layer] = self.evicted[layer]
            return (
                self.keys[layer][:, :, :new_cache_size].view(
                    self.batch,
                    self.num_heads,
                    new_cache_size * self.frame_tokens,
                    self.head_dim,
                ),
                self.values[layer][:, :, :new_cache_size].view(
                    self.batch,
                    self.num_heads,
                    new_cache_size * self.frame_tokens,
                    self.head_dim,
                ),
            )

    def update_functional(
        self,
        keys_in: torch.Tensor,
        values_in: torch.Tensor,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
        layer: int,
        cache_size: int,
    ):
        """Mutation-free cache update for CUDA graph compatibility.

        Instead of mutating self.keys/values in-place, this method:
        1. Takes keys_in, values_in as INPUT tensors (read-only)
        2. Returns NEW tensors with the update applied
        3. Does NOT mutate any tensors — pure functional

        This allows CUDA graphs to capture the forward pass since there
        are no in-place mutations to detect.

        Args:
            keys_in:    (B, n_heads, cache_len, frame_tokens, head_dim) — current cache
            values_in:  same shape
            new_k:      (B, n_heads, 1, frame_tokens, head_dim) — new key to insert
            new_v:      same shape
            layer:      which layer (for pointer lookup)
            cache_size: current number of filled slots

        Returns:
            (keys_out, values_out, k_for_attn, v_for_attn)
            - keys_out, values_out: updated cache tensors (new allocations)
            - k_for_attn, v_for_attn: (B, n_heads, seq_len, head_dim) for attention
        """
        ptr = cache_size  # ptr equals cache_size for single-step updates
        # Pure: self.evicted is not advanced. The caller passes the eviction count with the next call
        # (KeyValueCache(evicted=...)); only the key start of this call is recorded for RoPEAttention.
        self.key_start[layer] = self.evicted[layer] + (1 if cache_size >= self.cache_len else 0)
        B, n_heads, cache_len, frame_tokens, head_dim = keys_in.shape

        if cache_size < self.cache_len:
            # Growing phase: insert at ptr position using torch.cat
            # Split keys_in into [0:ptr] and [ptr+1:], insert new_k in between
            keys_out = torch.cat(
                [keys_in[:, :, :ptr], new_k, keys_in[:, :, ptr + 1 :]],
                dim=2,
            )
            values_out = torch.cat(
                [values_in[:, :, :ptr], new_v, values_in[:, :, ptr + 1 :]],
                dim=2,
            )
            new_cache_size = cache_size + 1
        else:
            # Rolling phase: shift left by 1, insert new at end
            keys_out = torch.cat([keys_in[:, :, 1:], new_k], dim=2)
            values_out = torch.cat([values_in[:, :, 1:], new_v], dim=2)
            new_cache_size = self.cache_len

        # Prepare tensors for attention (flatten cache_len × frame_tokens)
        k_for_attn = keys_out[:, :, :new_cache_size].reshape(B, n_heads, new_cache_size * frame_tokens, head_dim)
        v_for_attn = values_out[:, :, :new_cache_size].reshape(B, n_heads, new_cache_size * frame_tokens, head_dim)

        return keys_out, values_out, k_for_attn, v_for_attn


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        num_kv_heads=None,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        num_views=1,
        act_layer=nn.GELU,
        wide_silu=True,
        norm_layer=nn.LayerNorm,
        attn_impl="fmha",
        use_alibi=False,
        attn_depth=8,
        is_causal=False,
        grid_height=16,
        grid_width=16,
        grid_depth=8,
        layer_id=None,
        use_rope=False,
        block_size=64,
        rope_interleave=False,
        num_view_embeddings=None,
        use_qk_norm=False,
        qk_norm_type="layernorm",
        mup_attn_scale_mult=1.0,
        **kwargs,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if use_rope:
            self.attn = RoPEAttention(
                dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                attn_impl=attn_impl,
                grid_views=num_views,
                is_causal=is_causal,
                grid_height=grid_height,
                grid_width=grid_width,
                block_size=block_size,
                grid_depth=grid_depth,
                proj_drop=drop,
                layer_id=layer_id,
                rope_interleave=rope_interleave,
                num_view_embeddings=num_view_embeddings,
                use_qk_norm=use_qk_norm,
                qk_norm_type=qk_norm_type,
                mup_attn_scale_mult=mup_attn_scale_mult,
            )
        else:
            self.attn = Attention(
                dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                block_size=block_size,
                use_alibi=use_alibi,
                attn_depth=attn_depth,
                attn_impl=attn_impl,
                is_causal=is_causal,
                proj_drop=drop,
                layer_id=layer_id,
                use_qk_norm=use_qk_norm,
                qk_norm_type=qk_norm_type,
                mup_attn_scale_mult=mup_attn_scale_mult,
            )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
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

    def forward(
        self,
        x,
        attn_mask=None,
        cache=None,
        cache_size=None,
        extra_tokens=0,
        functional_cache=None,
        runtime_views=None,
        runtime_grid_h=None,
        runtime_grid_w=None,
        view_ids=None,
    ):
        # Only RoPEAttention understands runtime_views (used to pin V at runtime
        # when V<grid_views_built; see RoPEAttention.forward for rationale).
        attn_kwargs = dict(
            attn_mask=attn_mask,
            cache=cache,
            extra_tokens=extra_tokens,
            cache_size=cache_size,
            functional_cache=functional_cache,
        )
        if isinstance(self.attn, RoPEAttention):
            attn_kwargs["runtime_views"] = runtime_views
            # Runtime spatial grid (mixed-resolution training): pins the
            # (gh, gw) decomposition per batch so the RoPE positions match the
            # actual encoder output grid on this rank. Defaults (None) fall
            # back to the construction grid inside RoPEAttention.
            attn_kwargs["runtime_grid_h"] = runtime_grid_h
            attn_kwargs["runtime_grid_w"] = runtime_grid_w
            # Strong view conditioning (rope_view_embedding.md): only RoPEAttention
            # consumes view_ids; the legacy `Attention` class does not.
            attn_kwargs["view_ids"] = view_ids
        y_or_tuple = self.attn(self.norm1(x), **attn_kwargs)
        if functional_cache is not None:
            # Attention returns (y, new_layer_cache) in functional mode
            y, new_layer_cache = y_or_tuple
        else:
            y = y_or_tuple
            new_layer_cache = None
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if new_layer_cache is not None:
            return x, new_layer_cache
        return x
