# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Analytical FLOPs estimator for the robo_jepa predictor.

Pure functions (dict in, int out) that estimate the per-iter forward + backward
FLOPs of the ``VisionTransformerPredictor`` so the training loop can maintain a
cumulative compute counter for scaling-laws plots.

Scope is limited to the predictor; the frozen V-JEPA encoder and any optional
heads are explicitly excluded.

References:
  * ``app/robo_jepa/robo_jepa/models/predictor.py`` for architecture.
  * ``app/robo_jepa/robo_jepa/models/utils/modules.py`` for ``Block`` / ``MLP`` /
    ``SwiGLUFFN`` / ``RoPEAttention`` arithmetic.
  * ``app/robo_jepa/world_model.py:1051-1102`` for the rollout loop layout.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple


def _causal_local_attended_per_query(
    T: int,
    H: int,
    W: int,
    V: int,
    attn_depth: int,
    attn_height: int,
    attn_width: int,
    extras_per_step: int,
) -> float:
    """Average number of keys each query attends to under the predictor's causal
    local attention mask.

    Mirrors ``app/robo_jepa/robo_jepa/models/utils/xformer_utils.py::build_local_attention_mask``:

      * Time axis is **causal**: a query at frame ``t1`` attends to keys at
        frames ``[max(0, t1 - W_T + 1), t1]`` -- i.e., up to ``W_T`` past
        frames including itself. Early frames see fewer past frames (so the
        first ``W_T-1`` queries have a truncated window).
      * Within each attended frame, the spatial mask_block is full when
        ``W_H<0`` and ``W_W<0`` (the wm_*.yaml case); otherwise it's a
        symmetric ``(2*(W_H//2)+1) x (2*(W_W//2)+1)`` window around the query.
      * Per-step extras (action/state/dt) are part of each frame's block so
        they get the same temporal window. Video-level extras (e.g. camera
        extrinsics) are a constant flat suffix and are folded in below.
      * ``W_T < 0`` means "no temporal limit" — degenerates to full attention
        (still causal in time) i.e. attended_frames(t)=t+1.

    Returns the *average* per-query attended-key count (float). Multiply by
    the per-pair attention cost to get block-level attention FLOPs.
    """
    tps = H * W * V + int(extras_per_step)  # tokens per step (visual + per-step extras)
    if T <= 0 or tps <= 0:
        return 0.0

    # Temporal: sum_{t=0..T-1} attended_frames(t)
    if attn_depth < 0:
        # No temporal limit but still causal in time per mask builder loop t2 in [..., t1]
        sum_attended = T * (T + 1) // 2  # 1 + 2 + ... + T
    else:
        W_T = max(1, int(attn_depth))
        if T <= W_T:
            sum_attended = T * (T + 1) // 2
        else:
            sum_attended = W_T * (W_T + 1) // 2 + (T - W_T) * W_T

    # Spatial keys-per-query within a single attended (t1,t2) frame pair.
    if attn_height < 0 and attn_width < 0:
        # Full spatial: mask_block is all-True across HxWxV plus per-step extras
        per_frame_pair_keys = tps
    else:
        H_attn = H if attn_height < 0 else min(2 * (int(attn_height) // 2) + 1, H)
        W_attn = W if attn_width < 0 else min(2 * (int(attn_width) // 2) + 1, W)
        # mask_block iterates over visual indices only when h/w are windowed
        # (see xformer_utils.build_local_attention_mask lines 64-69); extras
        # within the block remain zero. Treat visual queries with the windowed
        # count and pretend extras attend to extras only (approximation;
        # negligible for the typical case).
        per_frame_pair_keys = H_attn * W_attn * V + int(extras_per_step)

    # Sum over all queries:
    #   total_pairs = sum_t (queries_at_t) * attended_frames(t) * per_frame_pair_keys
    #               = tps * sum_attended * per_frame_pair_keys
    # Average over queries (queries_total = T * tps):
    #   avg_per_q = (sum_attended / T) * per_frame_pair_keys
    return float(sum_attended) / float(T) * float(per_frame_pair_keys)


def _swiglu_hidden(mlp_hidden: int, wide_silu: bool, align: int = 8) -> int:
    """Match the SwiGLU hidden-dim arithmetic in ``modules.SwiGLUFFN``."""
    if not wide_silu:
        return int(mlp_hidden)
    h = int(2 * mlp_hidden / 3)
    h = (h + align - 1) // align * align
    return int(h)


def compute_predictor_block_flops_per_token(
    cfg_dict: Mapping[str, Any],
    seq_len: int,
    runtime_v: int,
) -> int:
    """Forward FLOPs for **one** transformer Block per token at the given seq_len.

    The seq_len argument is the *full* per-step token count
    (``T*H*W*V + extra``) and is used (together with ``runtime_v``) to recover
    the (T, H, W, V) grid for the attention-pair count. ``cfg_dict`` is the
    predictor's ``config`` block from the training YAML.

    The result includes (per token, per block):
      * QKV projection (GQA-aware)
      * Q/K RMSNorm (when use_qk_norm)
      * Output projection
      * MLP (GELU 2-layer or SwiGLU 3-layer, wide_silu-aware)
      * 2 layer norms (RMSNorm or LayerNorm)
      * Attention QK^T and softmax@V under the predictor's **causal local**
        mask (see ``xformer_utils.build_local_attention_mask``): query at
        frame t attends to frames ``[max(0, t-W_T+1), t]``; early frames
        attend to fewer past frames. The result is an *average* per-token
        figure across the clip (so total block attention FLOPs are
        ``B * seq_len * avg_per_token``).

    Notes / approximations:
      * Extra tokens (action/state/dt/extrinsics) are treated as part of each
        frame's per-step token block; they get the same temporal window as
        visual tokens. This matches the actual mask builder for the wm_*.yaml
        case (W_H<0, W_W<0) and is a small approximation otherwise.
      * Video-level extras (camera_extrinsics_tokens) are ignored in the
        attention term (negligible) and only counted in linear projections.
      * Block-rounding overhead of block-sparse attention kernels is not
        counted (this is the theoretical, mask-aware FLOPs the math requires).
    """
    d = int(cfg_dict["predictor_embed_dim"])
    num_heads = int(cfg_dict["num_heads"])
    num_kv_heads = int(cfg_dict.get("num_kv_heads") or num_heads)
    if num_heads <= 0 or d % num_heads != 0:
        raise ValueError(f"predictor_embed_dim={d} must be divisible by num_heads={num_heads}")
    head_dim = d // num_heads
    d_kv = num_kv_heads * head_dim
    mlp_ratio = float(cfg_dict.get("mlp_ratio", 4.0))
    use_silu = bool(cfg_dict.get("use_silu", False))
    wide_silu = bool(cfg_dict.get("wide_silu", True))
    use_qk_norm = bool(cfg_dict.get("use_qk_norm", False))
    norm_type = str(cfg_dict.get("norm_type", "layernorm")).lower()

    # Grid for attention-pair count. Visual token count = T*H*W*V; the seq_len
    # passed in may include extras, so back out (T, H, W) from the config.
    H = int(cfg_dict["grid_height"]) // int(cfg_dict.get("patch_size", 1))
    W = int(cfg_dict["grid_width"]) // int(cfg_dict.get("patch_size", 1))
    V = int(runtime_v)
    if V <= 0 or H <= 0 or W <= 0:
        raise ValueError(f"invalid grid: H={H} W={W} V={V}")
    visual_per_step = H * W * V
    if seq_len <= 0 or seq_len < visual_per_step:
        raise ValueError(f"seq_len={seq_len} smaller than visual_per_step={visual_per_step}")
    # Infer T from the visual portion: seq_len = T * visual_per_step + extra.
    # extra per step = action_tokens + state_tokens + dt_tokens, plus a single
    # video-level camera_extrinsics_tokens. We just need T for the window.
    extra_per_step = (
        int(cfg_dict.get("action_tokens", 0))
        + int(cfg_dict.get("state_tokens", 0))
        + (1 if bool(cfg_dict.get("use_dt", False)) else 0)
    )
    extras_video = int(cfg_dict.get("camera_extrinsics_tokens", 0))
    # Solve T from seq_len = T*(visual_per_step + extra_per_step) + extras_video.
    denom = visual_per_step + extra_per_step
    T_inferred = (seq_len - extras_video) // denom
    if T_inferred <= 0:
        raise ValueError(f"could not infer T from seq_len={seq_len}, cfg={dict(cfg_dict)}")

    local_attn_window = cfg_dict.get("local_attn_window", (-1, -1, -1))
    if isinstance(local_attn_window, (list, tuple)):
        attn_depth, attn_height, attn_width = (int(x) for x in local_attn_window)
    else:
        attn_depth = attn_height = attn_width = -1

    avg_attended_per_q = _causal_local_attended_per_query(
        T_inferred, H, W, V, attn_depth, attn_height, attn_width, extra_per_step
    )

    # -------- linear-projection FLOPs (per token) --------
    # QKV: 2 * d * (d + 2 * d_kv). Factor of 2 is MAC->FLOP.
    flops_qkv = 2 * d * (d + 2 * d_kv)
    # Output projection
    flops_out_proj = 2 * d * d
    # QK norm cost is O(head_dim) per (head, token); negligible but included.
    flops_qk_norm = 0
    if use_qk_norm:
        # RMSNorm ~ 4*head_dim per element: x^2, mean, rsqrt, scale.
        flops_qk_norm = 4 * head_dim * (num_heads + num_kv_heads)

    # MLP
    mlp_hidden = int(d * mlp_ratio)
    if use_silu:
        h = _swiglu_hidden(mlp_hidden, wide_silu=wide_silu)
        # Three linears: fc1 (d->h), fc2 (d->h), fc3 (h->d).
        flops_mlp = 2 * (2 * d * h + h * d)
        # SiLU+gating activation: ~3 FLOPs per hidden element.
        flops_mlp += 3 * h
    else:
        # Two linears + GELU.
        flops_mlp = 2 * (d * mlp_hidden + mlp_hidden * d)
        # GELU ~ 8 FLOPs per element (approx; constant absorbed in dominant term).
        flops_mlp += 8 * mlp_hidden

    # Norm cost (2 norms per block): ~5*d per token per norm for RMS/LN.
    norm_per_elem = 5 if norm_type == "rmsnorm" else 6
    flops_norms = 2 * norm_per_elem * d

    # -------- attention (per token, averaged across the clip) --------
    # For each token (with `num_heads` queries of dim `head_dim`):
    #   QK^T: 2 * head_dim * num_heads * avg_attended_per_q = 2 * d * avg_attended_per_q
    #   softmax: ~3 FLOPs per attended pair (exp, sum, div) -> 3 * num_heads * avg_attended_per_q
    #   P @ V:  2 * head_dim * num_heads * avg_attended_per_q = 2 * d * avg_attended_per_q
    # `avg_attended_per_q` correctly accounts for the causal local window in time
    # (queries early in the clip attend to fewer past frames).
    flops_attn = 4 * d * avg_attended_per_q + 3 * num_heads * avg_attended_per_q

    per_token = flops_qkv + flops_qk_norm + flops_out_proj + flops_mlp + flops_norms + flops_attn
    return int(per_token)


def _seq_len_for_step(cfg_dict: Mapping[str, Any], T_steps: int, runtime_v: int) -> int:
    """Token count for a predictor forward at the given number of frames ``T_steps``.

    Matches the layout produced by the predictor (visual tokens + per-step
    extras + video-level extras).
    """
    H = int(cfg_dict["grid_height"]) // int(cfg_dict.get("patch_size", 1))
    W = int(cfg_dict["grid_width"]) // int(cfg_dict.get("patch_size", 1))
    extra_per_step = (
        int(cfg_dict.get("action_tokens", 0))
        + int(cfg_dict.get("state_tokens", 0))
        + (1 if bool(cfg_dict.get("use_dt", False)) else 0)
    )
    extras_video = int(cfg_dict.get("camera_extrinsics_tokens", 0))
    return int(T_steps * (H * W * int(runtime_v) + extra_per_step) + extras_video)


def _single_forward_flops(
    cfg_dict: Mapping[str, Any],
    batch_size: int,
    runtime_v: int,
    T_steps: int,
) -> int:
    """Forward FLOPs for one predictor pass at ``T_steps`` frames and batch ``B``."""
    seq_len = _seq_len_for_step(cfg_dict, T_steps, runtime_v)
    depth = int(cfg_dict["depth"])
    d = int(cfg_dict["predictor_embed_dim"])
    embed_dim = int(cfg_dict["embed_dim"])
    block_per_token = compute_predictor_block_flops_per_token(cfg_dict, seq_len, runtime_v)
    backbone = batch_size * seq_len * depth * block_per_token
    # Dominant linear projections in/out (predictor_embed + predictor_proj).
    # Computed over visual tokens only — extras have their own (negligible)
    # MLPs in ProprioceptiveEmbedding which we approximate as zero.
    H = int(cfg_dict["grid_height"]) // int(cfg_dict.get("patch_size", 1))
    W = int(cfg_dict["grid_width"]) // int(cfg_dict.get("patch_size", 1))
    visual_tokens = T_steps * H * W * int(runtime_v)
    embed_in_out = 2 * batch_size * visual_tokens * (2 * embed_dim * d)
    return int(backbone + embed_in_out)


def compute_teacher_forced_flops_per_microbatch(
    predictor_cfg: Mapping[str, Any],
    batch_size: int,
    runtime_v: int,
    include_backward: bool = True,
) -> int:
    """Teacher-forced forward (+ backward) FLOPs for a single microbatch.

    Constant per rank (no randomness). Multiply by ``grad_accum_steps`` for the
    per-iter contribution from the teacher-forced pass.
    """
    num_frames = int(predictor_cfg["num_frames"])
    tubelet = int(predictor_cfg.get("tubelet_size", 1))
    T_base = num_frames // tubelet
    fwd = _single_forward_flops(predictor_cfg, batch_size, runtime_v, T_base)
    if include_backward:
        return int(3 * fwd)
    return int(fwd)


def compute_ar_flops_for_prefix_lens(
    predictor_cfg: Mapping[str, Any],
    trainer_cfg: Mapping[str, Any],
    batch_size: int,
    runtime_v: int,
    prefix_lens,
    include_backward: bool = True,
) -> int:
    """Realized AR-rollout FLOPs for a concrete list of sampled prefix lengths.

    Each entry of ``prefix_lens`` is the prefix length used for one call to
    ``world_model.rollout`` (i.e. one element of the ``prefixes`` tensor
    sampled inside ``step_model``). Inside ``rollout``, the inner loop runs
    ``rollout_steps - 1`` iters and at iter ``k`` the forward sees
    ``T = prefix_len + suffix_len + k``.

    Use this to compute exact-realized FLOPs given the actually-sampled
    prefixes during training (rather than the expected value).
    """
    if not prefix_lens:
        return 0
    rollout_steps = int(trainer_cfg.get("rollout_steps", 1))
    if rollout_steps <= 1:
        return 0
    inner_iters = rollout_steps - 1
    suffix_len = max(1, int(trainer_cfg.get("predictor_window_steps", 1)))

    total_fwd = 0
    for pl in prefix_lens:
        for k in range(inner_iters):
            T_k = int(pl) + suffix_len + k
            total_fwd += _single_forward_flops(predictor_cfg, batch_size, runtime_v, T_k)
    if include_backward:
        return int(3 * total_fwd)
    return int(total_fwd)


def compute_predictor_iter_flops(
    predictor_cfg: Mapping[str, Any],
    trainer_cfg: Mapping[str, Any],
    batch_size: int,
    runtime_v: int,
    grad_accum_steps: int,
    include_backward: bool = True,
) -> int:
    """Per-rank, per-iter (forward + backward) predictor FLOPs (expected value).

    Captures the two predictor invocations per microbatch (see
    ``app/robo_jepa/train.py::step_model`` and ``world_model.rollout``):

      1. **Teacher-forced forward** at ``T = num_frames / tubelet_size``.
      2. **Autoregressive rollout** (only when ``rollout_steps > 1``):
         ``world_model.rollout`` is called once per microbatch with a random
         prefix. The forward+backward execute on the full sequence at each
         inner iter; loss-mask doesn't reduce FLOPs because attention couples
         all positions.

    Returns the *expected* per-iter value averaged over the uniform prefix
    distribution (for ``train_rollout_prefixes="random"``). For runtime-exact
    realized FLOPs, use ``compute_teacher_forced_flops_per_microbatch`` +
    ``compute_ar_flops_for_prefix_lens`` together with the actual sampled
    prefix values.

    Args:
      predictor_cfg: ``model.transition_model.config`` from the training YAML.
      trainer_cfg: ``model.transition_model.trainer``.
      batch_size: per-rank microbatch size; used as the fallback when a view
        group/cell does not specify its own ``batch_size``.
      runtime_v: V for this rank's view-group.
      grad_accum_steps: per-rank K (microbatches per iter).
      include_backward: True (default) ⇒ 3× forward.
    """
    num_frames = int(predictor_cfg["num_frames"])
    tubelet = int(predictor_cfg.get("tubelet_size", 1))
    T_base = num_frames // tubelet
    rollout_steps = int(trainer_cfg.get("rollout_steps", 1))
    predictor_window_steps = int(trainer_cfg.get("predictor_window_steps", 1))
    train_rollout_prefixes = str(trainer_cfg.get("train_rollout_prefixes", "random")).lower()

    # 1) Teacher-forced forward at T = T_base
    fwd_teacher = float(_single_forward_flops(predictor_cfg, batch_size, runtime_v, T_base))

    # 2) Autoregressive rollout (only when rollout_steps > 1)
    fwd_rollout_per_microbatch = 0.0
    if rollout_steps > 1:
        suffix_len = max(1, predictor_window_steps)
        inner_iters = rollout_steps - 1

        # Mirror train.py:1215-1216 -- the `high` arg of torch.randint is
        # exclusive. So t ∈ {0, ..., high_exclusive - 1}.
        high_exclusive = T_base + 1 - rollout_steps - (suffix_len - 1)
        if high_exclusive <= 0:
            # Degenerate: no room for a prefix; AR is effectively skipped.
            prefix_lens = []
        else:
            prefix_lens = [t + 1 for t in range(high_exclusive)]

        # FLOPs per prefix sample: sum across the inner_iters AR forwards.
        # At iter k, T = prefix_len + suffix_len + k.
        per_prefix_fwd = []
        for pl in prefix_lens:
            f = 0.0
            for k in range(inner_iters):
                T_k = pl + suffix_len + k
                f += float(_single_forward_flops(predictor_cfg, batch_size, runtime_v, T_k))
            per_prefix_fwd.append(f)

        if prefix_lens:
            if train_rollout_prefixes == "all":
                fwd_rollout_per_microbatch = float(sum(per_prefix_fwd))
            else:
                # "random" (default): one prefix sampled uniformly per microbatch.
                fwd_rollout_per_microbatch = float(sum(per_prefix_fwd)) / float(len(per_prefix_fwd))

    fwd_per_microbatch = fwd_teacher + fwd_rollout_per_microbatch
    fwd_per_iter = float(grad_accum_steps) * fwd_per_microbatch
    if not include_backward:
        return int(round(fwd_per_iter))
    # Standard scaling-laws convention: backward ≈ 2× forward (the "6N" rule).
    return int(round(3.0 * fwd_per_iter))


def compute_world_predictor_iter_flops(
    predictor_cfg: Mapping[str, Any],
    trainer_cfg: Mapping[str, Any],
    batch_size: int,
    view_groups_cfg: Mapping[str, Mapping[str, Any]] | None,
    world_size: int,
    fallback_num_views: int = 1,
    fallback_grad_accum: int = 1,
    include_backward: bool = True,
    view_group_grids: Mapping[str, Any] | None = None,
) -> Tuple[int, Dict[str, int]]:
    """World-level per-iter predictor FLOPs and a per-group breakdown.

    Args:
      predictor_cfg: ``model.transition_model.config``.
      trainer_cfg: ``model.transition_model.trainer``.
      batch_size: per-rank microbatch size.
      view_groups_cfg: ``data.train.view_groups`` (or None for legacy flat config).
      world_size: total number of ranks (= ``nodes * tasks_per_node``).
      fallback_num_views: V when ``view_groups_cfg`` is None.
      fallback_grad_accum: K when ``view_groups_cfg`` is None.
      include_backward: forward (False) or forward+backward (True, default).
      view_group_grids: optional ``{group_name: (grid_h, grid_w)}`` (token
        units) for mixed-resolution training. When given, each group's FLOPs
        use its own grid instead of ``predictor_cfg``'s grid.

    Returns:
      ``(world_flops_iter, per_group_flops_per_rank)`` where the dict maps
      group name -> per-rank per-iter FLOPs for that group.
    """
    per_group: Dict[str, int] = {}
    if view_groups_cfg is None or len(view_groups_cfg) == 0:
        v = int(predictor_cfg.get("num_views", fallback_num_views))
        k = int(fallback_grad_accum)
        per_rank = compute_predictor_iter_flops(
            predictor_cfg,
            trainer_cfg,
            batch_size=batch_size,
            runtime_v=v,
            grad_accum_steps=k,
            include_backward=include_backward,
        )
        per_group["default"] = int(per_rank)
        return int(per_rank) * int(world_size), per_group

    total = 0
    for gname, gcfg in view_groups_cfg.items():
        r0, r1 = int(gcfg["rank_range"][0]), int(gcfg["rank_range"][1])
        group_size = r1 - r0
        v = int(gcfg["num_views"])
        k = int(gcfg.get("grad_accum_steps", 1))
        # Per-group microbatch size: each cell may set its own batch_size
        # (mixed-resolution training packs different per-rank batches), falling
        # back to the global default passed in.
        b = int(gcfg.get("batch_size", batch_size))
        # Per-group resolution: override the predictor grid for this group's
        # FLOPs when a grid map is supplied (mixed-resolution view-groups).
        if view_group_grids is not None and gname in view_group_grids:
            _gh, _gw = view_group_grids[gname]
            group_cfg = {**predictor_cfg, "grid_height": int(_gh), "grid_width": int(_gw)}
        else:
            group_cfg = predictor_cfg
        per_rank = compute_predictor_iter_flops(
            group_cfg,
            trainer_cfg,
            batch_size=b,
            runtime_v=v,
            grad_accum_steps=k,
            include_backward=include_backward,
        )
        per_group[gname] = int(per_rank)
        total += int(per_rank) * int(group_size)
    return int(total), per_group
