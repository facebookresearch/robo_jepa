# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from functools import lru_cache
import math

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend
from torch.nn.attention.flex_attention import (
    create_block_mask,
    flex_attention,
)
try:
    from xformers.ops import fmha
except ImportError:
    fmha = None

# torch._dynamo.config.optimize_ddp = False
torch._dynamo.config.cache_size_limit = 1000

flex_attention = torch.compile(flex_attention, dynamic=False)


@lru_cache
def create_block_mask_cached(score_mod, B, H, M, N, device="cuda"):
    block_mask = create_block_mask(score_mod, B, H, M, N, device=device)
    return block_mask


def attention(
    q,
    k,
    v,
    attn_mask,
    is_causal=False,
    block_size=64,
    num_heads=32,
    proj_drop_prob=0.0,
    attn_impl="fmha",
    tokens_per_step=257,
    attn_depth=8,
    use_alibi=False,
    scale=None,
):
    """Backend-dispatched attention.

    `scale` (default None): when None, every backend uses its built-in
    1/√d_k softmax scale (preserves the historical numerics).  When set
    to a float, it overrides the softmax scale globally — used by μP,
    where we pass `scale = 1/d_k` (i.e. an extra factor of 1/√d_k vs the
    standard 1/√d_k) so attention logits don't grow with width.

    Backends that don't accept a `scale=` kwarg directly (currently the
    cuDNN ATen op) are
    handled by pre-multiplying `q` by `scale * √d_k` so that their
    hard-coded internal `1/√d_k` reproduces the desired logit scale.
    """
    # Pre-scale q for backends that hard-code 1/√d_k internally.  Doing it
    # once here avoids per-backend duplication.  We invert the standard
    # `1/√d_k` on the fly so that backend's internal divide reproduces
    # the desired `scale`.  Only applied for the two affected backends.
    _need_q_prescale = scale is not None and attn_impl in ("sdpa_cudnn",)
    if _need_q_prescale:
        head_dim = q.shape[-1]
        q = q * (scale * math.sqrt(head_dim))
    if attn_impl == "fmha":
        q = q.permute(0, 2, 1, 3)  # [B, num_heads, N, D] -> [B, N, num_heads, D]
        k = k.permute(0, 2, 1, 3)  # [B, num_heads, N, D] -> [B, N, num_heads, D]
        v = v.permute(0, 2, 1, 3)  # [B, num_heads, N, D] -> [B, N, num_heads, D]
        if q.shape[2] != k.shape[2]:  # group-query attention
            groups = q.shape[2] // k.shape[2]
            k = k.repeat_interleave(groups, dim=2)
            v = v.repeat_interleave(groups, dim=2)
        if is_causal:
            attn_mask = fmha.attn_bias.LowerTriangularFromBottomRightMask()
            x = fmha.memory_efficient_attention(q, k, v, attn_bias=attn_mask, scale=scale).permute(0, 2, 1, 3)
        elif attn_mask is None:
            x = fmha.memory_efficient_attention(q, k, v, scale=scale).permute(0, 2, 1, 3)
        else:
            raise ValueError('fmha is supported with custom attention mask!')
    elif attn_impl == "flex":
        assert attn_depth is not None, "provide attention depth"

        def block_causal_local_mask(b, h, q_idx, kv_idx):
            return ((q_idx // tokens_per_step) - (kv_idx // tokens_per_step) >= 0) & (
                (q_idx // tokens_per_step) - (kv_idx // tokens_per_step) < attn_depth
            )

        B, H, M, _ = q.shape
        _, _, N, _ = k.shape
        if use_alibi:
            # NOTE: kv-cache based inference with alibi is not implemented yet!
            def generate_alibi_bias():
                alibi_bias = []
                for h in range(num_heads):
                    alibi_bias.append(-((h + 1) * 8.0 / num_heads))
                alibi_bias = torch.tensor(alibi_bias, device="cuda")
                alibi_bias = torch.exp2(alibi_bias)
                return alibi_bias

            alibi_bias = generate_alibi_bias()

            def alibi_score_mod(score, b, h, q_idx, kv_idx):
                bias = alibi_bias[h] * ((kv_idx // tokens_per_step) - (q_idx // tokens_per_step))
                return score + bias

        block_mask = create_block_mask_cached(block_causal_local_mask, B, H, M, N, device=q.device)
        if q.shape[1] != k.shape[1]:  # group-query attention
            groups = q.shape[1] // k.shape[1]
            k = k.repeat_interleave(groups, dim=1)
            v = v.repeat_interleave(groups, dim=1)
        x = flex_attention(
            q,
            k,
            v,
            score_mod=alibi_score_mod if use_alibi else None,
            block_mask=block_mask,
            scale=scale,
        )
    elif attn_impl == "sdpa":
        use_gqa = q.shape[1] != k.shape[1]
        if use_gqa and attn_mask is not None:
            # The mem-efficient kernel (the only fused one that takes a mask) needs equal head counts.
            groups = q.shape[1] // k.shape[1]
            k = k.repeat_interleave(groups, dim=1)
            v = v.repeat_interleave(groups, dim=1)
            use_gqa = False
        with torch.nn.attention.sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=proj_drop_prob,
                is_causal=is_causal,
                attn_mask=attn_mask,
                enable_gqa=use_gqa,
                scale=scale,
            )
    elif attn_impl == "sdpa_cudnn":
        use_gqa = q.shape[1] != k.shape[1]
        if use_gqa:
            groups = q.shape[1] // k.shape[1]
            k = k.repeat_interleave(groups, dim=1)
            v = v.repeat_interleave(groups, dim=1)
        # Call cuDNN attention ATen op directly so it survives torch.export.
        # The op returns 9 values per its schema; index [0] is the output tensor.
        x = torch.ops.aten._scaled_dot_product_cudnn_attention(
            q, k, v, attn_mask, False, proj_drop_prob, is_causal,
        )[0]
    else:
        raise ValueError(f"Unsupported attn_impl: {attn_impl}")
    return x
